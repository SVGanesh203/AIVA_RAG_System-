
#!/usr/bin/env python3
"""
embeding_url_logic.py

URL-specific embedding logic that reuses the core pipeline from embedding_logic.py.

Behaviour:
- Reads a URL export folder produced by embed_public_url.py:
    <export_dir>/
        page.md        # CLEANED page markdown (header/footer/sidebar/cookies removed)
        metadata.json  # optional (title, source_url, etc)
        images/        # images already downloaded by embed_public_url (from cleaned body only)

TEXT
- Uses page.md content (already cleaned by embed_public_url.py) as the main
  article text.
- No additional Goose-based text extraction is done.
- No extra header/footer stripping at this stage; DOM-level stripping has
  already removed the chrome.

IMAGES
- Markdown image references get context windows around them (from page.md).
- Goose3 is used ONLY to decide which images belong to the main article.
- Only images whose filename matches Goose's main-article images are embedded.
- All other images (header, footer, sidebars, thumbnails) are treated as
  low_priority and never copied/embedded.

Chunking
- We enrich chunk metadata (keywords, entities, etc.).
- We no longer drop chunks based on low-priority heuristics; everything that
  survived DOM cleaning is embedded.
"""

import os
import re
import json
import time
import shutil
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS

from unified_document_processor import MultiFormatDocumentProcessor

# Import core helpers + GLOBAL state from embedding_logic
import embedding_logic  # we rely on its globals being initialized

IMAGE_EXTS = embedding_logic.IMAGE_EXTS
logger = __import__("logging").getLogger(__name__)

# --------------------------------------------------------------------
# Optional: Goose3 for main-article image detection
# --------------------------------------------------------------------
try:
    from goose3 import Goose
    HAS_GOOSE = True
except Exception:
    Goose = None  # type: ignore
    HAS_GOOSE = False


def _image_key(url_or_path: str) -> str:
    """
    Normalized key for matching images between Goose and our markdown:
    - strip query params
    - take just the basename
    - lowercase
    """
    if not url_or_path:
        return ""
    return url_or_path.split("?")[0].split("/")[-1].lower()


def _get_main_image_keys_from_url(source_url: str) -> Set[str]:
    """
    Use Goose3 to detect main-article images.

    Returns a set of normalized image keys (filenames), so that we can
    match against markdown image URLs or local filenames.

    NOTE: This is focused only on images. Text is taken from page.md.
    """
    keys: Set[str] = set()
    if not (HAS_GOOSE and source_url):
        return keys

    try:
        g = Goose()
        article = g.extract(source_url)
        if article is None:
            return keys

        # Top image (hero)
        top_img = getattr(article, "top_image", None)
        if top_img is not None:
            src = getattr(top_img, "src", None)
            if src:
                k = _image_key(str(src))
                if k:
                    keys.add(k)

        # All article images
        try:
            imgs = getattr(article, "images", []) or []
            for u in imgs:
                if not u:
                    continue
                k = _image_key(str(u))
                if k:
                    keys.add(k)
        except Exception:
            pass

        if keys:
            logger.info(
                "Goose detected %d main-article images for %s",
                len(keys),
                source_url,
            )
        else:
            logger.warning("Goose returned no image keys for %s", source_url)

    except Exception as e:
        logger.warning("Goose main-image detection failed for %s: %s", source_url, e)

    return keys


# --------------------------------------------------------------------
# Low-priority markers (for classification only; not for hard dropping)
# --------------------------------------------------------------------
LOW_PRIORITY_SIDEBAR_MARKERS: List[str] = [
    "required reading",
    "quick links",
    "related reading",
    "related articles",
    "you might also like",
    "more stories",
    "trending now",
    "recommended for you",
    "hr hub",
    "your hr hub",
]

LOW_PRIORITY_FOOTER_MARKERS: List[str] = [
    "privacy policy",
    "terms of use",
    "terms & conditions",
    "cookie policy",
    "cookies policy",
    "all rights reserved",
    "©",
    "copyright ",
    "follow us",
    "connect with us",
    "contact us",
    "intouch communities",
    # "answers",  # removed to avoid nuking main 'Answers' content
    "choose your campus",
]

LOW_PRIORITY_HEADER_MARKERS: List[str] = [
    "sign in",
    "log in",
    "login",
    "sign up",
    "register",
    "search",
    "menu",
    "my account",
    "profile",
    "notifications",
    "language",
    "select language",
]

_NAV_KEYWORDS: List[str] = [
    "home",
    "about",
    "about us",
    "services",
    "solutions",
    "products",
    "blog",
    "news",
    "careers",
    "jobs",
    "support",
    "help",
    "contact",
    "contact us",
    "faq",
    "privacy",
    "terms",
    "security",
    "status",
    "dashboard",
    "logout",
    "sign out",
    "twitter",
    "facebook",
    "linkedin",
    "instagram",
    "youtube",
]


def _is_nav_like(text: str) -> bool:
    """
    Heuristic to detect very short navigation/menu-like text blocks,
    e.g. "Home  About  Careers  Contact".
    """
    t = " ".join((text or "").lower().split())
    if not t:
        return False

    words = t.split()
    # If block is long, it's unlikely to be just nav/menu
    if len(words) > 15:
        return False

    hits = sum(1 for kw in _NAV_KEYWORDS if kw in t)
    return hits >= 2


def _is_low_priority_context(text: str) -> bool:
    """
    Returns True if the given text looks like sidebar/footer/ads/header/nav content.
    Used only for classification; not for hard dropping chunks anymore.
    """
    t = (text or "").lower()
    for k in (
        LOW_PRIORITY_SIDEBAR_MARKERS
        + LOW_PRIORITY_FOOTER_MARKERS
        + LOW_PRIORITY_HEADER_MARKERS
    ):
        if k in t:
            return True

    if _is_nav_like(text):
        return True

    return False


def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


# --------------------------------------------------------------------
# Markdown image parsing (similar to embedding_logic)
# --------------------------------------------------------------------
_MD_IMAGE_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)')


def _extract_md_images_with_context(md_text: str) -> List[Dict[str, Any]]:
    """
    Returns a list in doc order:
      { 'alt': str, 'url': str, 'md_index': int, 'display_page': int, 'context_text': str }

    Context = +/- 3 lines around the image reference.
    """
    images: List[Dict[str, Any]] = []
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
            "md_index": len(images),
            "display_page": len(images) + 1,
            "context_text": window,
        })
    return images


def _resolve_filename_from_url(url: str) -> str:
    try:
        return url.split("?")[0].split("/")[-1]
    except Exception:
        return ""


# --------------------------------------------------------------------
# Main heading detection (article anchor)
# --------------------------------------------------------------------
def _extract_main_heading_key(md_text: str, max_words: int = 8) -> str:
    """
    Find the first top-level heading like '# Security Advisory...' and
    build a normalized key string from its first few words.
    """
    lines = md_text.splitlines()
    heading_line = ""
    for l in lines:
        lt = l.strip()
        if lt.startswith("#"):
            heading_line = lt.lstrip("#").strip()
            if heading_line:
                break

    if not heading_line:
        return ""

    words = re.findall(r"\w+", heading_line.lower())
    if not words:
        return ""

    key_words = words[:max_words]
    return " ".join(key_words)


def _contains_main_heading_key(text: str, main_heading_key: str) -> bool:
    if not main_heading_key:
        return False
    t_words = re.findall(r"\w+", (text or "").lower())
    t_norm = " ".join(t_words)
    return main_heading_key in t_norm


# --------------------------------------------------------------------
# URL folder loader (page.md + images + metadata + Goose image keys)
# --------------------------------------------------------------------
def _load_url_folder(folder_path: str) -> Tuple[Document, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Reads page.md (already cleaned by embed_public_url.py) as the main content.

    - Uses page.md text directly as the primary content to embed.
    - Still uses Goose to detect main-article images only.
    - Applies heuristics for classification (section_type), but does not
      hard-drop text based on these labels.
    """
    folder = Path(folder_path)
    page_md = folder / "page.md"
    if not page_md.exists():
        raise FileNotFoundError(f"Expected page.md in {folder_path}")

    # Full markdown = cleaned article body (header/footer/sidebar cookie removed)
    md_text = page_md.read_text(encoding="utf-8", errors="ignore")

    # Detect main heading key from markdown
    main_heading_key = _extract_main_heading_key(md_text)

    # Base metadata (optional)
    base_meta: Dict[str, Any] = {}
    meta_json = folder / "metadata.json"
    if meta_json.exists():
        try:
            base_meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            base_meta = {}

    document_name = folder.name
    title = base_meta.get("title") or document_name
    author = base_meta.get("author") or "Unknown"
    source_url = base_meta.get("source_url") or base_meta.get("url") or ""

    # Main text is just the cleaned markdown text
    main_text = md_text

    # Goose image keys for classification
    goose_main_img_keys: Set[str] = set()
    if source_url:
        goose_main_img_keys = _get_main_image_keys_from_url(source_url)

    # Markdown-derived images (ordered, from cleaned page.md)
    md_images = _extract_md_images_with_context(md_text)

    images_info: List[Dict[str, Any]] = []
    for it in md_images:
        raw_url = it["url"]
        fname = _resolve_filename_from_url(raw_url)
        ctx = it.get("context_text", "") or ""

        key = _image_key(raw_url or fname)

        # If Goose gave us image keys, use them; else fallback to small heuristic.
        if goose_main_img_keys:
            section_type = "main" if (key and key in goose_main_img_keys) else "low_priority"
        else:
            has_main = _contains_main_heading_key(ctx, main_heading_key)
            is_low_ctx = _is_low_priority_context(ctx)
            if is_low_ctx and not has_main:
                section_type = "low_priority"
            else:
                section_type = "main"

        images_info.append({
            "filename": fname,
            "source_url": raw_url,
            "alt": it.get("alt", ""),
            "referenced_in_md": True,
            "md_index": it["md_index"],
            "page": it["display_page"] - 1,
            "display_page": it["display_page"],
            "image_type": "image",
            "extraction_method": "markdown_ref",
            "context_text": ctx,
            "section_type": section_type,
        })

    # Loose files in images/ (fallback, classified via Goose keys)
    img_root = folder / "images"
    if img_root.exists() and img_root.is_dir():
        for f in sorted(img_root.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                name = f.name
                if any(x.get("filename") == name for x in images_info if x.get("filename")):
                    continue
                key = _image_key(name)
                if goose_main_img_keys:
                    section_type = "main" if (key and key in goose_main_img_keys) else "low_priority"
                else:
                    section_type = "unknown"
                images_info.append({
                    "filename": name,
                    "source_url": None,
                    "alt": "",
                    "referenced_in_md": False,
                    "md_index": None,
                    "page": 0,
                    "display_page": len(md_images) + 1,
                    "image_type": "image",
                    "extraction_method": "folder_scan",
                    "context_text": "",
                    "section_type": section_type,
                })

    # Attach absolute paths for images
    def _find_source_path(fname: str) -> str:
        p = img_root / fname
        if p.exists():
            return str(p)
        if img_root.exists():
            for f in img_root.rglob(fname):
                if f.is_file():
                    return str(f)
        return ""

    for img in images_info:
        fname = img.get("filename")
        if not fname:
            continue
        sp = _find_source_path(fname)
        if sp:
            img["path"] = sp
            img["content_type"] = _guess_mime(sp)
            try:
                img["size"] = Path(sp).stat().st_size
            except Exception:
                pass

    # Main text document
    main_doc = Document(
        page_content=main_text,
        metadata={
            "title": title,
            "author": author,
            "type": "text",
            "file_category": "url",
            "processing_method": "url_folder",
            "word_count": len(main_text.split()),
            "document_name": document_name,
            "main_heading_key": main_heading_key,
        },
    )

    base_meta.setdefault("document_name", document_name)
    base_meta.setdefault("images", images_info)
    base_meta.setdefault("title", title)
    base_meta.setdefault("author", author)
    base_meta.setdefault("main_heading_key", main_heading_key)

    return main_doc, base_meta, images_info


# --------------------------------------------------------------------
# Image copy (skip low-priority images entirely)
# --------------------------------------------------------------------
def _ensure_unique_filename(dest_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(dest_dir, candidate)):
        candidate = f"{base}-{i}{ext}"
        i += 1
    return candidate


def _copy_images_to_output_url(images_info: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    """
    URL-specific image copy:

    - Images with section_type == "low_priority" are SKIPPED completely.
    - Only main/unknown images are copied + embedded.
    """
    if not images_info:
        return []

    images_dir = os.path.join(output_dir, "images")
    embedding_logic.ensure_directory(images_dir)

    out: List[Dict[str, Any]] = []

    for img in images_info:
        if img.get("section_type") == "low_priority":
            # completely ignore low-priority images (sidebar/footer/ads/etc.)
            continue

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


# --------------------------------------------------------------------
# Chunk enrichment (URL-aware; classification only)
# --------------------------------------------------------------------
def enhance_chunk_metadata_url(chunk: Document) -> Dict[str, Any]:
    """
    Similar to embedding_logic.enhance_chunk_metadata but URL-aware:

    - Uses main_heading_key (from metadata) + low-priority markers.
    - Marks section_type = "main" or "low_priority".
    - content_quality_score is informational only; we don't hard-drop on it.
    """
    text = chunk.page_content
    md = dict(chunk.metadata) if chunk.metadata else {}

    main_heading_key = md.get("main_heading_key", "")

    md["contextual_keywords"] = embedding_logic.extract_keywords_keybert(text)
    md["named_entities"] = embedding_logic.extract_entities_spacy(text)
    md["math_equations"] = embedding_logic.extract_math_equations(text)
    md["links_emails"] = embedding_logic.extract_links_emails(text)
    md["sections"] = embedding_logic.extract_sections(text)
    md["citations"] = embedding_logic.extract_citations(text)
    md["word_count"] = len(text.split())
    md["char_count"] = len(text)

    score = 0
    if md["word_count"] > 50:
        score += 1

    # classify (for info)
    low_ctx = _is_low_priority_context(text)
    has_main = _contains_main_heading_key(text, main_heading_key)

    if low_ctx and not has_main:
        md["section_type"] = "low_priority"
        score -= 1
    else:
        md["section_type"] = md.get("section_type", "main")

    md["content_quality_score"] = score
    return md


# --------------------------------------------------------------------
# Core: URL folder → embeddings
# --------------------------------------------------------------------
def create_embeddings_from_url_folder(
    folder_path: str,
    output_dir: str,
    cfg: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    URL-specific embedding entry point.

    - embedding_logic.initialize_embedding_model() and
      embedding_logic.initialize_ocr_engines() MUST be called beforehand.
    - Uses embedding_logic.embedding_model / easyocr_reader / tesseract_available.

    TEXT:
    - Comes from the cleaned page.md created by embed_public_url.py
      (no header/footer/sidebar/cookie chrome).

    IMAGES:
    - Filtered using Goose main-article image detection so that only
      article images are embedded; sidebar/footer/ads are skipped.
    """
    if embedding_logic.embedding_model is None:
        raise RuntimeError(
            "Embedding model is not initialized. "
            "Call initialize_embedding_model() from embedding_logic first."
        )

    emb_cfg = cfg.get("embeddings", {})
    chunk_size = int(emb_cfg.get("chunk_size", 1000))
    chunk_overlap = int(emb_cfg.get("chunk_overlap", 200))
    batch_size = int(emb_cfg.get("batch_size", 50))

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"URL export folder not found: {folder_path}")

    # Load URL content
    print(f"📁 Loading URL export: {folder.name}")
    main_doc, base_meta, images_info_raw = _load_url_folder(folder_path)

    document_name = Path(output_dir).name
    title = base_meta.get("title", document_name)
    author = base_meta.get("author", "Unknown")
    main_heading_key = base_meta.get("main_heading_key", "")

    # 1) Copy only main/unknown images (skip low-priority ones)
    copied_images = _copy_images_to_output_url(images_info_raw, output_dir)

    # 2) Optional OCR enrichment for copied images
    if copied_images and (embedding_logic.easyocr_reader or embedding_logic.tesseract_available):
        print("🖼️ OCR tagging URL images...")
        tmp: List[Dict[str, Any]] = []
        for img in copied_images:
            p = img.get("path")
            if not p or not os.path.exists(p):
                tmp.append(img)
                continue
            ocr = embedding_logic.extract_text_multi_ocr(p)
            enriched = dict(img)
            enriched["ocr_text"] = ocr["combined_text"]
            enriched["easyocr_text"] = ocr["easyocr_text"]
            enriched["tesseract_text"] = ocr["tesseract_text"]
            enriched["ocr_confidence"] = ocr["confidence_scores"]
            enriched["ocr_processing_time"] = ocr["processing_time"]
            tmp.append(enriched)
        copied_images = tmp

    # 3) Build docs (text + image docs)
    final_docs: List[Document] = []

    main_doc.metadata["document_name"] = document_name
    main_doc.metadata["file_category"] = "url"
    main_doc.metadata["main_heading_key"] = main_heading_key
    final_docs.append(main_doc)

    for img in copied_images:
        display_page = img.get("display_page", 1)
        ctx = (img.get("context_text") or "").strip()
        ocr_text = (img.get("ocr_text") or "").strip()
        alt = img.get("alt") or ""
        section_type = img.get("section_type", "main")

        summary_lines = [
            f"IMAGE — display page {display_page}",
            f"Document: {title}",
        ]
        if alt:
            summary_lines.append(f"Alt text: {alt}")
        if ctx:
            summary_lines.append(f"Nearby context: {ctx}")
        if ocr_text:
            summary_lines.append(f"OCR: {ocr_text}")

        img_doc = Document(
            page_content="\n".join(summary_lines),
            metadata={
                "type": "image",
                "file_category": "url",
                "document_name": document_name,
                "display_page": display_page,
                "page": display_page - 1,
                "image_filename": img.get("filename"),
                "image_relative_path": img.get("relative_path"),
                "relative_path": img.get("relative_path"),
                "image_path": img.get("path"),
                "source_url": img.get("source_url"),
                "alt": alt,
                "context_text": ctx,
                "referenced_in_md": img.get("referenced_in_md", False),
                "md_index": img.get("md_index"),
                "section_type": section_type,
                "main_heading_key": main_heading_key,
            },
        )
        final_docs.append(img_doc)

    # 4) Chunking: DO NOT drop low-priority text chunks anymore
    print("✂️ Chunking URL document…")
    all_chunks: List[Document] = []
    splitter = MultiFormatDocumentProcessor()

    for doc in final_docs:
        tmp = type(
            "TempProcessed",
            (),
            {"content": doc.page_content, "metadata": doc.metadata, "doc_type": "url"},
        )()
        chs = splitter.create_chunks(tmp, chunk_size, chunk_overlap)
        for ch in chs:
            meta = enhance_chunk_metadata_url(ch)
            ch.metadata = meta
            ch.metadata.update(
                {
                    "source_document_display_page": doc.metadata.get("display_page", 1),
                    "document_name": document_name,
                }
            )
            all_chunks.append(ch)

    print(f"   Total chunks (after filtering): {len(all_chunks)}")

    # 5) Build FAISS
    print("🧮 Creating URL embeddings…")
    if len(all_chunks) > batch_size:
        db = None
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            if i == 0:
                db = FAISS.from_documents(batch, embedding_logic.embedding_model)
            else:
                x = FAISS.from_documents(batch, embedding_logic.embedding_model)
                db.merge_from(x)  # type: ignore
    else:
        db = FAISS.from_documents(all_chunks, embedding_logic.embedding_model)

    # 6) Save index + images metadata
    embedding_logic.ensure_directory(output_dir)
    db.save_local(output_dir)

    images_metadata_path = os.path.join(output_dir, "images", "images_metadata.json")
    if copied_images:
        embedding_logic.ensure_directory(os.path.dirname(images_metadata_path))
        with open(images_metadata_path, "w", encoding="utf-8") as f:
            json.dump(copied_images, f, indent=2, ensure_ascii=False)

    # High-level metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w", encoding="utf-8") as f:
        f.write(f"Document Name (slug): {document_name}\n")
        f.write(f"Title: {title}\n")
        f.write(f"Author: {author}\n")
        f.write(f"Total Chunks: {len(all_chunks)}\n")
        f.write(f"Images (non-low-priority): {len(copied_images)}\n")

    print(f"✅ URL folder embedded: {document_name}")

    return db, {
        "author": author,
        "title": title,
        "file_type": "url",
        "file_format": "folder",
        "document_name": document_name,
        "images": copied_images,
        "total_chunks": len(all_chunks),
        "processing_date": __import__("datetime").datetime.now().isoformat(),
    }
