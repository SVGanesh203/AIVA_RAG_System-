"""
agents/agent_indexer.py

Indexes each agent's data sources (files + URLs) into its own FAISS vector store.
Each agent gets its own subfolder:
  agents/indexes/<agent_id>/faiss_index/
  agents/indexes/<agent_id>/chunks.json      ← raw text chunks for BM25 fallback
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

AGENTS_DIR   = Path(__file__).parent
INDEXES_DIR  = AGENTS_DIR / "indexes"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_index_dir(agent_id: str) -> Path:
    d = INDEXES_DIR / agent_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scrape_url(url: str) -> str:
    """Fetch and extract plain text from a URL using requests + basic HTML stripping."""
    try:
        import requests
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts = []
                self._skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "footer", "header"):
                    self._skip = True
            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer", "header"):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self._parts.append(stripped)

        headers = {"User-Agent": "Mozilla/5.0 (AIVA-RAG-Agent/1.0)"}
        resp = requests.get(url, timeout=20, headers=headers)
        resp.raise_for_status()
        parser = _Stripper()
        parser.feed(resp.text)
        return "\n".join(parser._parts)
    except Exception as e:
        logger.warning(f"[indexer] URL scrape failed for {url}: {e}")
        return ""


def _read_file(path: str) -> str:
    """Read text from file. Supports .txt, .pdf, .csv, .json, .md."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"[indexer] File not found: {path}")
        return ""
    suffix = p.suffix.lower()
    try:
        if suffix == ".pdf":
            import fitz
            doc = fitz.open(str(p))
            return "\n".join(page.get_text() for page in doc)
        elif suffix in (".txt", ".md"):
            return p.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".csv":
            import csv
            rows = []
            with open(p, encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(", ".join(row))
            return "\n".join(rows)
        elif suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        else:
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"[indexer] File read error {path}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Simple character-level chunker with overlap."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Core indexing ─────────────────────────────────────────────────────────────

def index_agent(agent: Dict, embedding_model) -> bool:
    """
    Build (or rebuild) the FAISS index for a single agent.
    Returns True on success.
    """
    agent_id  = agent["id"]
    sources   = agent.get("sources", [])
    index_dir = _agent_index_dir(agent_id)
    faiss_path = index_dir / "faiss_index"
    chunks_path = index_dir / "chunks.json"

    logger.info(f"[indexer] Indexing agent '{agent_id}' ({len(sources)} sources)...")

    all_texts: List[str] = []
    all_meta: List[Dict] = []

    for source in sources:
        stype = source.get("type", "")
        sval  = source.get("value", "")

        if stype == "url":
            logger.info(f"[indexer]   Scraping URL: {sval}")
            raw = _scrape_url(sval)
            label = sval
        elif stype == "file":
            logger.info(f"[indexer]   Reading file: {sval}")
            raw = _read_file(sval)
            label = Path(sval).name
        else:
            logger.warning(f"[indexer]   Unknown source type: {stype}")
            continue

        chunks = _chunk_text(raw)
        logger.info(f"[indexer]   → {len(chunks)} chunks from {label}")
        for c in chunks:
            all_texts.append(c)
            all_meta.append({"source": label, "type": stype, "agent_id": agent_id})

    if not all_texts:
        logger.warning(f"[indexer] No text extracted for agent '{agent_id}'")
        return False

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(all_texts, all_meta)
        ]
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(str(faiss_path))

        # Save raw chunks for BM25 / keyword fallback
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"text": t, "meta": m} for t, m in zip(all_texts, all_meta)],
                f, ensure_ascii=False, indent=2
            )

        logger.info(f"[indexer] ✅ Agent '{agent_id}' indexed: {len(all_texts)} chunks.")
        return True

    except Exception as e:
        logger.error(f"[indexer] ❌ FAISS indexing failed for '{agent_id}': {e}")
        return False


def load_agent_index(agent_id: str, embedding_model):
    """Load the FAISS vectorstore for an agent. Returns None if not indexed yet."""
    try:
        from langchain_community.vectorstores import FAISS
        faiss_path = _agent_index_dir(agent_id) / "faiss_index"
        if not faiss_path.exists():
            return None
        return FAISS.load_local(
            str(faiss_path),
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.warning(f"[indexer] Could not load index for '{agent_id}': {e}")
        return None


def agent_is_indexed(agent_id: str) -> bool:
    return (INDEXES_DIR / agent_id / "faiss_index").exists()


def search_agent_index(
    agent_id: str,
    query: str,
    embedding_model,
    k: int = 6
) -> List[Dict]:
    """Search a single agent's FAISS index. Returns list of {text, source, score}."""
    vs = load_agent_index(agent_id, embedding_model)
    if vs is None:
        return []
    try:
        results = vs.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in results:
            out.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", agent_id),
                "agent_id": agent_id,
                "score": float(score),
            })
        return out
    except Exception as e:
        logger.warning(f"[indexer] Search error for agent '{agent_id}': {e}")
        return []
