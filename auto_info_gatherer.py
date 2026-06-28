"""
agents/auto_info_gatherer.py

The "Automatic Info Gatherer" from the workflow diagram.

Takes:
  - user_query          : raw user message
  - static_agent_ctx    : context string from matched agent indexes (Stage 2)
  - rag_doc_ctx         : context string from the main RAG document search
  - classification      : output of query_classifier.classify_query()

Does:
  1. If needs_realtime → run realtime_search()
  2. Chunk + rank all live results against the query
  3. Merge: live context + static agent context + RAG doc context
  4. If needs_verification → build a side-by-side comparison block
  5. Return a single augmented_context string for the LLM prompt

The final context string is structured so the LLM knows:
  - Which part came from uploaded documents (user's notes)
  - Which part came from live web sources (verified / authoritative)
  - Any discrepancies flagged explicitly (for chemistry notes scenario)
"""

import logging
from typing import Dict, List, Tuple, Optional

from agents.realtime_search import realtime_search
from agents.query_classifier import classify_query

logger = logging.getLogger(__name__)

MAX_LIVE_CHARS   = 4000
MAX_STATIC_CHARS = 3000
MAX_RAG_CHARS    = 3000


# ── Text chunker for live results ─────────────────────────────────────────────

def _chunk(text: str, size: int = 600, overlap: int = 80) -> List[str]:
    if not text.strip():
        return []
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ── Relevance scorer (simple keyword overlap) ─────────────────────────────────

def _score_chunk(chunk: str, query_tokens: set) -> float:
    words  = set(chunk.lower().split())
    hits   = len(words & query_tokens)
    return hits / (len(query_tokens) + 1e-9)


def _rank_chunks(chunks: List[str], query: str) -> List[str]:
    tokens = set(query.lower().split())
    scored = sorted(chunks, key=lambda c: _score_chunk(c, tokens), reverse=True)
    return scored


# ── Live context builder ───────────────────────────────────────────────────────

def _build_live_context(
    results: List[Dict],
    query: str,
    max_chars: int = MAX_LIVE_CHARS,
) -> Tuple[str, List[str]]:
    """
    From raw search results, build a ranked context string.
    Returns (context_string, list_of_source_urls).
    """
    all_chunks: List[Tuple[str, str]] = []   # (chunk_text, source_label)

    for r in results:
        source_label = f"{r['title']} ({r['url']})"
        # Use body if available, else snippet
        full_text = r.get("body", "") or r.get("snippet", "")
        if not full_text:
            continue
        for c in _chunk(full_text):
            all_chunks.append((c, source_label))

    if not all_chunks:
        return "", []

    # Rank by relevance
    tokens = set(query.lower().split())
    scored = sorted(
        all_chunks,
        key=lambda x: _score_chunk(x[0], tokens),
        reverse=True,
    )

    sections:    List[str] = []
    seen_labels: set       = set()
    sources:     List[str] = []
    total_chars            = 0

    for chunk_text, label in scored:
        block = f"[Source: {label}]\n{chunk_text}"
        if total_chars + len(block) > max_chars:
            break
        sections.append(block)
        total_chars += len(block)
        if label not in seen_labels:
            seen_labels.add(label)
            # Extract just the URL from label
            import re
            m = re.search(r'\((https?://[^)]+)\)', label)
            if m:
                sources.append(m.group(1))

    context = "\n\n---\n\n".join(sections)
    return context, sources


# ── Verification block builder ─────────────────────────────────────────────────

def _build_verification_block(
    rag_doc_ctx: str,
    live_ctx: str,
) -> str:
    """
    Builds a structured block that tells the LLM to compare the student's
    notes against live authoritative sources and flag discrepancies.
    """
    return f"""=== VERIFICATION MODE — COMPARE AND FLAG ERRORS ===

STUDENT'S UPLOADED NOTES (what their notes say):
{rag_doc_ctx[:MAX_RAG_CHARS] if rag_doc_ctx else "(no uploaded notes found)"}

AUTHORITATIVE / LIVE SOURCES (what is actually correct):
{live_ctx[:MAX_LIVE_CHARS] if live_ctx else "(no live sources retrieved)"}

INSTRUCTION FOR AI:
You are cross-checking the student's notes against authoritative sources.
For EACH piece of information in the student's notes:
  - If it matches the authoritative source → confirm it as correct.
  - If it differs → explicitly say: "Your note says [X], but the correct answer is [Y]."
  - If a concept is missing from the notes → mention what should be added.
Be specific, educational, and kind. Do not skip any errors found.

=== END VERIFICATION BLOCK ===
"""


# ── Main public function ───────────────────────────────────────────────────────

def gather_and_augment(
    user_query: str,
    static_agent_ctx: str = "",
    rag_doc_ctx: str = "",
    has_uploaded_docs: bool = True,
    max_live_results: int = 5,
) -> Tuple[str, Dict]:
    """
    The Automatic Info Gatherer.

    Args:
        user_query        : raw user message
        static_agent_ctx  : already-retrieved agent FAISS context (from AgentManager)
        rag_doc_ctx       : already-retrieved RAG document context
        has_uploaded_docs : whether user has docs in the RAG system
        max_live_results  : how many web results to fetch

    Returns:
        (final_augmented_context, metadata_dict)

        metadata_dict contains:
          - classification   : output of classify_query
          - live_sources     : list of URLs used
          - used_realtime    : bool
          - used_verification: bool
    """
    # 1. Classify
    classification = classify_query(user_query, has_uploaded_docs=has_uploaded_docs)
    logger.info(
        f"[gatherer] Query classified: "
        f"realtime={classification['needs_realtime']} "
        f"verify={classification['needs_verification']} "
        f"domain={classification['domain_hint']} "
        f"time={classification['time_constraint']}"
    )

    live_ctx     = ""
    live_sources: List[str] = []
    used_rt      = False

    # 2. Real-time search if needed
    if classification["needs_realtime"]:
        search_q = classification["search_query"]
        logger.info(f"[gatherer] Running real-time search: '{search_q}'")
        try:
            results  = realtime_search(search_q, max_results=max_live_results)
            live_ctx, live_sources = _build_live_context(results, user_query)
            used_rt  = True
            logger.info(
                f"[gatherer] Live context built: "
                f"{len(live_ctx)} chars from {len(live_sources)} sources"
            )
        except Exception as e:
            logger.warning(f"[gatherer] Real-time search failed: {e}")

    # 3. Build final context block
    used_verify = classification["needs_verification"] and has_uploaded_docs

    if used_verify:
        # Verification mode: side-by-side comparison with explicit error flagging
        final_ctx = _build_verification_block(rag_doc_ctx, live_ctx)

        # Also prepend any agent static context before the verification block
        if static_agent_ctx:
            final_ctx = (
                "=== DOMAIN KNOWLEDGE (from agents) ===\n"
                + static_agent_ctx[:MAX_STATIC_CHARS]
                + "\n=== END DOMAIN KNOWLEDGE ===\n\n"
                + final_ctx
            )

    else:
        # Standard augmentation: live > agent static > rag docs
        parts: List[str] = []

        if live_ctx:
            parts.append(
                "=== REAL-TIME WEB CONTEXT ===\n"
                + live_ctx
                + "\n=== END REAL-TIME CONTEXT ==="
            )

        if static_agent_ctx:
            parts.append(
                "=== AGENT KNOWLEDGE BASE ===\n"
                + static_agent_ctx[:MAX_STATIC_CHARS]
                + "\n=== END AGENT KNOWLEDGE ==="
            )

        if rag_doc_ctx and not used_verify:
            parts.append(
                "=== YOUR UPLOADED DOCUMENTS ===\n"
                + rag_doc_ctx[:MAX_RAG_CHARS]
                + "\n=== END UPLOADED DOCUMENTS ==="
            )

        final_ctx = "\n\n".join(parts)

    # 4. Append time-constraint instruction if detected
    if classification["time_constraint"] and not used_verify:
        final_ctx += (
            f"\n\n[TIME CONSTRAINT NOTE: The user needs to complete this "
            f"within {classification['time_constraint']}. "
            f"Prioritise fast, simplified techniques over traditional long methods.]"
        )

    meta = {
        "classification":    classification,
        "live_sources":      live_sources,
        "used_realtime":     used_rt,
        "used_verification": used_verify,
    }

    return final_ctx, meta
