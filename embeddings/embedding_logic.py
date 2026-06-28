#!/usr/bin/env python3
"""
embedding_logic.py  (KBase-aware, Markdown image tagging; LOCAL-IMAGE-FIRST)

- Reads a KBase export folder: page.md + attachments/ or images/
- Parses Markdown to find ![](…) image references
- Copies every resolved image to {output_dir}/images and writes images_metadata.json
- Builds FAISS with text chunks + image "documents" that point to the COPIED files
"""

import os
import re
import time
import json
import mimetypes
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from unified_document_processor import (
    MultiFormatDocumentProcessor,
    FileTypeDetector,
    get_document_name,
    validate_document_path,
    AdvancedTableProcessor,
    EnhancedImageProcessor,
)
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from keybert import KeyBERT
import spacy
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore

try:
    from easyocr import Reader
except ImportError:
    Reader = None  # type: ignore
    print("Warning: EasyOCR not available - OCR disabled")

# ----------------- Globals -----------------
kw_model = KeyBERT('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")  # ensure model is installed

easyocr_reader: Optional[Reader] = None
tesseract_available: bool = False
embedding_model: Optional[HuggingFaceEmbeddings] = None

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# ----------------- Utils -----------------
def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"}

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

# ----------------- OCR init -----------------
def initialize_ocr_engines(use_easyocr: bool = True, use_tesseract: bool = True) -> None:
    global easyocr_reader, tesseract_available
    if use_easyocr and Reader is not None:
        try:
            easyocr_reader = Reader(['en'], gpu=False)
            print("✅ EasyOCR initialized")
        except Exception as e:
            print(f"⚠️ EasyOCR init failed: {e}")
            easyocr_reader = None
    else:
        easyocr_reader = None

    if use_tesseract and pytesseract is not None:
        try:
            pytesseract.get_tesseract_version()
            tesseract_available = True
            print("✅ Tesseract available")
        except Exception as e:
            print(f"⚠️ Tesseract not available: {e}")
            tesseract_available = False
    else:
        tesseract_available = False

# ----------------- Lightweight NLP helpers -----------------
def extract_keywords_keybert(text: str) -> List[str]:
    if not text.strip():
        return []
    kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english')
    return [k[0] if isinstance(k, (list, tuple)) else k for k in kws]

def extract_entities_spacy(text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_math_equations(text: str) -> List[str]:
    return re.findall(r'(?:\\\[.*?\\\]|\\\((.*?)\\\))', text, re.DOTALL)

def extract_links_emails(text: str) -> List[str]:
    url_pattern = r'https?://[^\s)]+'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return re.findall(url_pattern, text) + re.findall(email_pattern, text)

def extract_sections(text: str) -> List[str]:
    section_pattern = r'^(?:[A-Z][A-Z ]{3,}|#+\s.*|Section \d+.*?)$'
    return re.findall(section_pattern, text, re.MULTILINE)

def extract_citations(text: str) -> List[str]:
    bracket_pattern = r'\[[0-9]{1,3}\]'
    paren_pattern = r'\([A-Z][a-z]+ et al\., \d{4}\)'
    return re.findall(bracket_pattern, text) + re.findall(paren_pattern, text)

# ----------------- OCR -----------------
def preprocess_image_for_ocr(image_path: str) -> Optional[List[np.ndarray]]:
    try:
        # If cv2 didn't import correctly, bail out early
        if cv2 is None:
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results: List[np.ndarray] = []

        # Denoise
        try:
            results.append(cv2.fastNlMeansDenoising(gray))
        except Exception:
            results.append(gray)

        # Contrast Limited Adaptive Histogram Equalization
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            results.append(clahe.apply(gray))
        except Exception:
            pass

        # Adaptive threshold
        try:
            results.append(
                cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            )
        except Exception:
            pass

        # Morph close
        try:
            results.append(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8)))
        except Exception:
            pass

        # Otsu threshold (correct constant: THRESH_OTSU)
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            otsu_flag = getattr(cv2, "THRESH_OTSU", 0)  # fallback to 0 if missing
            _, thr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + otsu_flag)
            results.append(thr)
        except Exception:
            pass

        return results or [gray]
    except Exception as e:
        print(f"⚠️ preprocess failed: {e}")
        return None


def extract_text_multi_ocr(image_path: str) -> Dict[str, Any]:
    res = {'easyocr_text':'','tesseract_text':'','combined_text':'','confidence_scores':{},'processing_time':0.0}
    start = time.time()
    imgs = preprocess_image_for_ocr(image_path)
    if imgs is None:
        return res

    if easyocr_reader:
        try:
            best_text, best_conf = "", 0.0
            for img in imgs:
                out = easyocr_reader.readtext(img)
                parts, conf = [], 0.0
                for (_b, t, c) in out:
                    if c > 0.3:
                        parts.append(t)
                        conf += c
                if parts:
                    avg = conf/len(parts)
                    txt = " ".join(parts)
                    if avg > best_conf:
                        best_text, best_conf = txt, avg
            res['easyocr_text'] = best_text
            res['confidence_scores']['easyocr'] = float(best_conf)
        except Exception as e:
            print(f"⚠️ easyocr failed: {e}")

    if tesseract_available:
        try:
            best_text, best_conf = "", 0.0
            cfgs = ['--psm 6','--psm 3','--psm 4','--psm 8']
            for img in imgs:
                for cfg in cfgs:
                    txt = pytesseract.image_to_string(img, config=cfg)
                    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
                    confs = [int(c) for c in data.get('conf',[]) if str(c).isdigit() and int(c)>0]
                    avg = (sum(confs)/len(confs)) if confs else 0.0
                    if avg > best_conf and txt.strip():
                        best_text, best_conf = txt, avg
            res['tesseract_text'] = best_text
            res['confidence_scores']['tesseract'] = float(best_conf)
        except Exception as e:
            print(f"⚠️ tesseract failed: {e}")

    e_txt = res['easyocr_text'].strip()
    t_txt = res['tesseract_text'].strip()
    res['combined_text'] = e_txt if len(e_txt) >= len(t_txt) else t_txt
    if e_txt and t_txt and abs(len(e_txt)-len(t_txt)) < max(20, 0.2*len(t_txt)):
        # merge uniques
        words, seen, uniq = (e_txt + " " + t_txt).split(), set(), []
        for w in words:
            lw = w.lower()
            if lw not in seen:
                uniq.append(w); seen.add(lw)
        res['combined_text'] = " ".join(uniq)
    res['processing_time'] = time.time()-start
    return res

# ----------------- Chunk enrichment -----------------
def enhance_chunk_metadata(chunk: Document) -> Dict[str, Any]:
    text = chunk.page_content
    md = dict(chunk.metadata) if chunk.metadata else {}

    md["contextual_keywords"] = extract_keywords_keybert(text)
    md["named_entities"] = extract_entities_spacy(text)
    md["math_equations"] = extract_math_equations(text)
    md["links_emails"] = extract_links_emails(text)
    md["sections"] = extract_sections(text)
    md["citations"] = extract_citations(text)
    md["word_count"] = len(text.split())
    md["char_count"] = len(text)

    score = 0
    if md["word_count"] > 50: score += 1
    md["content_quality_score"] = score
    return md

# ----------------- Embedding model -----------------
def initialize_embedding_model(model_options: List[str], device: str="cpu", device_name: str="cpu") -> bool:
    global embedding_model
    print(f"Loading embedding model on {device_name}...")
    for i, name in enumerate(model_options):
        try:
            print(f" Trying {name}...")
            embedding_model = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f" ✅ Loaded: {name}")
            return True
        except Exception as e:
            print(f" ❌ {name} failed: {e}")
    print(" ❌ No embedding model available.")
    return False

# ----------------- Markdown parsing -----------------
_MD_IMAGE_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)')

def _extract_md_images_with_context(md_text: str) -> List[Dict[str, Any]]:
    """
    Returns a list in doc order:
      { 'alt': str, 'url': str, 'md_index': int, 'display_page': int, 'context_text': str }
    """
    images = []
    lines = md_text.splitlines()
    joined = md_text
    for match in _MD_IMAGE_RE.finditer(joined):
        alt = (match.group('alt') or '').strip()
        url = (match.group('url') or '').strip()
        start = match.start()
        line_idx = joined[:start].count("\n")
        lo = max(0, line_idx - 3)
        hi = min(len(lines), line_idx + 4)
        window = "\n".join(l.strip() for l in lines[lo:hi] if l.strip())
        images.append({
            "alt": alt,
            "url": url,
            "md_index": len(images),          # 0-based
            "display_page": len(images)+1,    # 1-based pseudo page
            "context_text": window
        })
    return images

def _resolve_filename_from_url(url: str) -> Optional[str]:
    try:
        name = url.split("?")[0].split("/")[-1]
        return name if any(name.lower().endswith(ext) for ext in IMAGE_EXTS) else None
    except Exception:
        return None

def _load_kbase_folder(folder_path: str) -> Tuple[Document, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Reads page.md as main content; tags images based on Markdown references first.
    Falls back to enumerating attachments/images if not referenced in md.
    """
    folder = Path(folder_path)
    page_md = folder / "page.md"
    if not page_md.exists():
        raise FileNotFoundError(f"Expected page.md in {folder_path}")

    md_text = page_md.read_text(encoding="utf-8", errors="ignore")

    # Base metadata (optionally from metadata.json)
    base_meta: Dict[str, Any] = {}
    meta_json = folder / "metadata.json"
    if meta_json.exists():
        try:
            base_meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            base_meta = {}

    document_name = folder.name  # temporary; will be overridden by output_dir basename later
    title = base_meta.get("title") or document_name
    author = base_meta.get("author") or "Unknown"

    # Markdown-derived images (ordered)
    md_images = _extract_md_images_with_context(md_text)

    # Initial images_info from md refs
    images_info: List[Dict[str, Any]] = []
    for it in md_images:
        filename = _resolve_filename_from_url(it["url"]) or ""
        images_info.append({
            "filename": filename,
            "source_url": it["url"],          # keep original KBase link for reference only
            "alt": it.get("alt", ""),
            "referenced_in_md": True,
            "md_index": it["md_index"],
            "page": it["display_page"]-1,
            "display_page": it["display_page"],
            "image_type": "image",
            "extraction_method": "markdown_ref",
            "context_text": it.get("context_text","")
        })

    # Loose files in attachments/ and images/
    for sub in ["attachments", "images"]:
        p = folder / sub
        if p.exists() and p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    name = f.name
                    if any(x.get("filename")==name for x in images_info if x.get("filename")):
                        continue
                    images_info.append({
                        "filename": name,
                        "source_url": None,
                        "alt": "",
                        "referenced_in_md": False,
                        "md_index": None,
                        "page": 0,
                        "display_page": len(md_images)+1,
                        "image_type": "image",
                        "extraction_method": "folder_scan",
                        "context_text": ""
                    })

    # Attach absolute source path for each image (resolve by filename)
    def _find_source_path(fname: str) -> Optional[str]:
        for sub in ["attachments","images"]:
            p = folder / sub / fname
            if p.exists():
                return str(p)
        for sub in ["attachments","images"]:
            p = folder / sub
            if p.exists():
                for f in p.rglob(fname):
                    if f.is_file():
                        return str(f)
        return None

    for img in images_info:
        if img.get("filename"):
            sp = _find_source_path(img["filename"])
            if sp:
                img["path"] = sp
                img["content_type"] = _guess_mime(sp)
                try:
                    img["size"] = Path(sp).stat().st_size
                except Exception:
                    pass

    # Main text document (document_name will be overridden later)
    main_doc = Document(
        page_content=md_text,
        metadata={
            "title": title,
            "author": author,
            "type": "text",
            "file_category": "kbase",
            "processing_method": "kbase_folder",
            "word_count": len(md_text.split()),
            "document_name": document_name
        }
    )

    base_meta.setdefault("document_name", document_name)
    base_meta.setdefault("images", images_info)
    base_meta.setdefault("title", title)
    base_meta.setdefault("author", author)

    return main_doc, base_meta, images_info

# ----------------- Copy images into output -----------------
def _ensure_unique_filename(dest_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{base}-{i}{ext}"
        i += 1
    return candidate

def _copy_images_to_output(images_info: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    if not images_info:
        return []
    images_dir = os.path.join(output_dir, "images")
    ensure_directory(images_dir)
    out = []
    for img in images_info:
        src = img.get("path")
        updated = dict(img)
        if src and os.path.isfile(src):
            unique = _ensure_unique_filename(images_dir, os.path.basename(src))
            dst = os.path.join(images_dir, unique)
            try:
                shutil.copy2(src, dst)
                updated["path"] = dst
                updated["relative_path"] = os.path.join("images", unique).replace("\\", "/")
                updated["stored_at"] = "images"
            except Exception as e:
                updated["copy_error"] = str(e)
                updated["relative_path"] = None
                updated["stored_at"] = None
        else:
            updated["relative_path"] = None
            updated["stored_at"] = None
        out.append(updated)
    return out

# ----------------- Core: KBase → embeddings -----------------
def create_embeddings_from_kbase_folder(folder_path: str, output_dir: str, cfg: Dict[str, Any]) -> Tuple[Any, Dict]:
    if embedding_model is None:
        raise RuntimeError("Embedding model is not initialized. Call initialize_embedding_model() first.")

    emb_cfg = cfg.get("embeddings", {})
    chunk_size = int(emb_cfg.get("chunk_size", 1000))
    chunk_overlap = int(emb_cfg.get("chunk_overlap", 200))
    batch_size = int(emb_cfg.get("batch_size", 50))

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"KBase folder not found: {folder_path}")

    # Load KBase content
    print(f"📁 Loading KBase export: {folder.name}")
    main_doc, base_meta, images_info_raw = _load_kbase_folder(folder_path)

    # IMPORTANT: force document_name = output folder name (URL + serving alignment)
    document_name = Path(output_dir).name
    title = base_meta.get("title", document_name)
    author = base_meta.get("author", "Unknown")

    # 1) Copy images FIRST → now we have local relative paths like "images/<unique>"
    copied_images = _copy_images_to_output(images_info_raw, output_dir)

    # 2) Optional OCR enrichment on the **copied** files
    if copied_images and (easyocr_reader or tesseract_available):
        print("🖼️ OCR tagging images...")
        tmp = []
        for img in copied_images:
            p = img.get("path")
            if not p or not os.path.exists(p):
                tmp.append(img); continue
            ocr = extract_text_multi_ocr(p)
            enriched = dict(img)
            enriched["ocr_text"] = ocr["combined_text"]
            enriched["easyocr_text"] = ocr["easyocr_text"]
            enriched["tesseract_text"] = ocr["tesseract_text"]
            enriched["ocr_confidence"] = ocr["confidence_scores"]
            enriched["ocr_processing_time"] = ocr["processing_time"]
            tmp.append(enriched)
        copied_images = tmp

    # 3) Build docs list (text + image docs) using LOCAL relative paths
    final_docs: List[Document] = []

    # main text (reset to the serving doc_name)
    main_doc.metadata["document_name"] = document_name
    final_docs.append(main_doc)

    # image docs – point to copied files
    for img in copied_images:
        display_page = img.get("display_page", 1)
        ctx = (img.get("context_text") or "").strip()
        ocr_text = (img.get("ocr_text") or "").strip()
        alt = img.get("alt") or ""

        summary_lines = [
            f"IMAGE — display page {display_page}",
            f"Document: {title}",
        ]
        if alt: summary_lines.append(f"Alt text: {alt}")
        if ctx: summary_lines.append(f"Nearby context: {ctx}")
        if ocr_text: summary_lines.append(f"OCR: {ocr_text}")

        img_doc = Document(
            page_content="\n".join(summary_lines),
            metadata={
                "type": "image",
                "document_name": document_name,
                "display_page": display_page,
                "page": display_page-1,
                "image_filename": img.get("filename"),
                # the **local** relative path (used by /embedding/<doc>/<rel>)
                "image_relative_path": img.get("relative_path"),
                "relative_path": img.get("relative_path"),
                "image_path": img.get("path"),
                # keep original for reference only (we will NOT render this)
                "source_url": img.get("source_url"),
                "alt": alt,
                "context_text": ctx,
                "referenced_in_md": img.get("referenced_in_md", False),
                "md_index": img.get("md_index"),
            }
        )
        final_docs.append(img_doc)

    # 4) Chunking
    print("✂️ Chunking…")
    all_chunks: List[Document] = []
    splitter = MultiFormatDocumentProcessor()
    for doc in final_docs:
        cs, ov = chunk_size, chunk_overlap
        tmp = type('TempProcessed', (), {'content': doc.page_content, 'metadata': doc.metadata, 'doc_type': "kbase"})()
        chs = splitter.create_chunks(tmp, cs, ov)
        for ch in chs:
            ch.metadata = enhance_chunk_metadata(ch)
            ch.metadata.update({
                'source_document_display_page': doc.metadata.get('display_page', 1),
                'document_name': document_name
            })
        all_chunks.extend(chs)

    print(f"   Total chunks: {len(all_chunks)}")

    # 5) Build FAISS
    print("🧮 Creating embeddings…")
    if len(all_chunks) > batch_size:
        db = None
        for i in range(0, len(all_chunks), batch_size):
            b = all_chunks[i:i+batch_size]
            if i == 0:
                db = FAISS.from_documents(b, embedding_model)
            else:
                x = FAISS.from_documents(b, embedding_model)
                db.merge_from(x)  # type: ignore
    else:
        db = FAISS.from_documents(all_chunks, embedding_model)

    # 6) Save index + images metadata
    ensure_directory(output_dir)
    db.save_local(output_dir)

    # Persist copied images metadata (already includes relative_path)
    images_metadata_path = os.path.join(output_dir, "images", "images_metadata.json")
    if copied_images:
        ensure_directory(os.path.dirname(images_metadata_path))
        with open(images_metadata_path, "w", encoding="utf-8") as f:
            json.dump(copied_images, f, indent=2, ensure_ascii=False)

    # High-level metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w", encoding="utf-8") as f:
        f.write(f"Document Name (slug): {document_name}\n")
        f.write(f"Title: {title}\n")
        f.write(f"Author: {author}\n")
        f.write(f"Total Chunks: {len(all_chunks)}\n")
        f.write(f"Images: {len(copied_images)}\n")

    print(f"✅ KBase folder embedded: {document_name}")
    return db, {
        'author': author,
        'title': title,
        'file_type': 'kbase',
        'file_format': 'folder',
        'document_name': document_name,
        'images': copied_images,
        'total_chunks': len(all_chunks),
        'processing_date': datetime.now().isoformat(),
    }

# ----------------- Multi-format fallback (unchanged core) -----------------
def create_enhanced_embeddings_multi_format(file_path: str, output_dir: str, cfg: Dict[str, Any]) -> Tuple[Any, Dict]:
    p = Path(file_path)
    if p.is_dir() and (p / "page.md").exists():
        return create_embeddings_from_kbase_folder(file_path, output_dir, cfg)
    raise NotImplementedError("Non-KBase path omitted here for brevity (use your existing code).")

def create_enhanced_embeddings(pdf_path: str, output_dir: str, cfg: Dict[str, Any]) -> Tuple[Any, Dict]:
    return create_enhanced_embeddings_multi_format(pdf_path, output_dir, cfg)
