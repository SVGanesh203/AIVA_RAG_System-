
#!/usr/bin/env python3
"""
embed_public_url.py

Embed ANY public URL (or internal HTTP(S) page) using a URL-specific pipeline:

- Ask for or accept a URL.
- Download the page.
- Strip scripts/styles/etc.
- Strip template chrome for Insider-style pages:
    * header
    * footer
    * right sidebar column (Quick Links, Required Reading, HR Hub, etc.)
    * cookie-consent / privacy bar + modal
- Take the remaining <body> content as the main area.
- Download all images referenced in the remaining <body> into images/.
- Convert cleaned body HTML → Markdown and write page.md.
- Also write cleaned page.html.
- Treat the export folder like a KBase-style folder and call
  embeding_url_logic.create_embeddings_from_url_folder.
- Update access.csv with a new column for this embedding (Public URL role).
"""

import os
import sys
import time
import json
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

# ---- Reuse your existing logic ----
try:
    from embedding_logic import (
        initialize_embedding_model,
        initialize_ocr_engines,
    )
except Exception as e:
    print(f"❌ Could not import embedding_logic: {e}")
    sys.exit(1)

try:
    # URL-specific embedding logic
    from embeding_url_logic import create_embeddings_from_url_folder
except Exception as e:
    print(f"❌ Could not import embeding_url_logic: {e}")
    sys.exit(1)

try:
    from edit_access import update_access_csv
except Exception:
    update_access_csv = None

# We reuse the KBase slug sanitiser for consistent naming
try:
    from list_kbase_pages import sanitize_slug
except Exception:
    import re

    def sanitize_slug(s: str) -> str:
        s = re.sub(r"[^\w]+", "_", (s or "").strip().lower())
        return re.sub(r"_{2,}", "_", s).strip("_") or "page"

# ===========================
# INLINE CONFIG
# ===========================
_THIS_FILE = Path(__file__).resolve()
_PROJECT_DIR = _THIS_FILE.parent          # .../catapult_chatbot/embeddings
_REPO_ROOT = _PROJECT_DIR.parent          # repo root (one level up)

EMBEDDINGS_OUTPUT_DIR = _REPO_ROOT / "embedding"
URL_EXPORTS_DIR = _REPO_ROOT / "url_exports"
ACCESS_CSV_PATH = _REPO_ROOT / "access.csv"
ACCESS_CSV_FALLBACK = _PROJECT_DIR / "templates" / "access.csv"

EMBEDDING_MODELS = ["sentence-transformers/all-MiniLM-L6-v2"]
EMBED_DEVICE = "cpu"
EMBED_DEVICE_NAME = "cpu"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 50

# access.csv behaviour for PUBLIC URL embeddings
PUBLIC_URL_DESIGNATION = "Public URL"
GRANT_VALUE = 1
DEFAULT_VALUE = 0

# HTTP options
DEFAULT_TIMEOUT = 30
VERIFY_SSL = True
# ===========================


# ---- HTML → Markdown helper ----
try:
    import pypandoc  # type: ignore
    HAS_PANDOC = True
except Exception:
    HAS_PANDOC = False

try:
    from markdownify import markdownify as md_convert  # type: ignore
    HAS_MARKDOWNIFY = True
except Exception:
    HAS_MARKDOWNIFY = False


def html_to_markdown(html: str) -> str:
    """
    Convert generic HTML to Markdown:
    - Tries pypandoc (GitHub-flavoured MD) if available.
    - Else falls back to markdownify.
    - Else plain text via BeautifulSoup.
    """
    html = html or ""

    if HAS_PANDOC:
        try:
            return pypandoc.convert_text(
                html,
                "gfm",
                format="html",
                extra_args=[
                    "--wrap=none",
                    "--atx-headers",
                ],
            )
        except Exception as e:
            print(f"⚠️ Pandoc conversion failed ({e}); falling back to markdownify/plain text…")

    if HAS_MARKDOWNIFY:
        try:
            return md_convert(
                html,
                heading_style="ATX",
                bullets="-",
                strip=["style", "script"],
            ).strip()
        except Exception as e:
            print(f"⚠️ markdownify conversion failed ({e}); falling back to plain text…")

    # Last-resort: plain text
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    return soup.get_text("\n", strip=True)


# ---- access.csv helpers ----
def _pick_access_csv() -> Path:
    """
    Choose the access.csv path. Prefer repo-root/access.csv, else fallback to templates/access.csv.
    If none exist, create a minimal one at repo-root.
    """
    if ACCESS_CSV_PATH.exists():
        return ACCESS_CSV_PATH
    if ACCESS_CSV_FALLBACK.exists():
        return ACCESS_CSV_FALLBACK

    print("⚠️ access.csv not found. Creating a new one at repo root with only 'Designation' header.")
    ACCESS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCESS_CSV_PATH.write_text("Designation\n", encoding="utf-8", newline="\n")
    return ACCESS_CSV_PATH


def _update_access_csv_for_embedding(out_dir: Path, roles: list[str] = []) -> None:
    """
    After a successful embed, add a column with the embedding folder name
    and set selected roles=1, others=0.
    """
    column_name = out_dir.name
    csv_path = _pick_access_csv()

    if update_access_csv is None:
        print("⚠️ edit_access.update_access_csv not available; skipping access.csv update.")
        return

    print(f"📝 Updating access.csv → {csv_path.name} (column: {column_name}) with roles: {roles}")
    ok, msg = update_access_csv(
        csv_path=str(csv_path),
        column_name=column_name,
        allowed_roles=roles,
        grant_value=GRANT_VALUE,
        default_value=DEFAULT_VALUE,
        create_backup=False,
    )
    if ok:
        print(f"✅ access.csv updated. {msg}")
    else:
        print(f"❌ Failed to update access.csv: {msg}")


# ---- URL helpers ----
def _download_url(url: str) -> Tuple[str, str]:
    """
    Download the URL and return (html, final_url).
    """
    session = requests.Session()
    session.verify = VERIFY_SSL
    headers = {
        "User-Agent": "Catapult-Embedding-Bot/1.0 (+https://intouchcx.com)",
    }
    print(f"🌐 Fetching URL: {url}")
    resp = session.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.text, resp.url  # resp.url may be the final redirected URL


def _extract_page_title(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Untitled Page"


def _ensure_unique_filename(dest_dir: Path, base_name: str) -> str:
    """
    Ensure filename uniqueness inside dest_dir.
    """
    name = base_name
    stem, ext = os.path.splitext(base_name)
    i = 1
    while (dest_dir / name).exists():
        name = f"{stem}-{i}{ext}"
        i += 1
    return name


def _rewrite_images_and_download(body_tag: BeautifulSoup, page_url: str, images_dir: Path) -> None:
    """
    For the CLEANED <body>:
    - Find all <img src="...">,
    - Resolve to absolute URL,
    - Download them into images_dir,
    - Rewrite src to "images/<local_name>" so markdown has local references.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.verify = VERIFY_SSL

    for img in body_tag.find_all("img"):
        src = img.get("src")
        if not src:
            continue

        abs_url = urljoin(page_url, src)
        parsed = urlparse(abs_url)
        if not parsed.scheme.startswith("http"):
            continue

        basename = os.path.basename(parsed.path) or "image"
        if not os.path.splitext(basename)[1]:
            # Guess extension from mime type
            try:
                head = session.head(abs_url, timeout=DEFAULT_TIMEOUT)
                mt = head.headers.get("Content-Type", "")
                ext = mimetypes.guess_extension(mt.split(";")[0].strip()) or ""
                basename += ext
            except Exception:
                pass

        safe_name = sanitize_slug(os.path.splitext(basename)[0]) + os.path.splitext(basename)[1]
        local_name = _ensure_unique_filename(images_dir, safe_name)

        dest_path = images_dir / local_name
        try:
            print(f"   • downloading image {abs_url} → {dest_path}")
            with session.get(abs_url, stream=True, timeout=DEFAULT_TIMEOUT) as res:
                res.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in res.iter_content(8192):
                        if chunk:
                            f.write(chunk)

            img["src"] = f"{images_dir.name}/{local_name}"
        except Exception as e:
            print(f"   ⚠️ image download failed: {abs_url} ({e})")


# ---------- KEY: strip header/footer/sidebar/cookies ----------
def _strip_insider_chrome(body_tag: BeautifulSoup) -> None:
    """
    Remove header, footer, sidebar chrome, and cookie banner/modal from an Insider/Divi page:
    - Global header/footer layouts
    - Right sidebar column that contains Quick Links / Required Reading / HR Hub
    - Cookie-consent bar, overlay, and modal
    """

    # 1) Remove obvious header/footer containers
    for sel in [
        "header",          # generic <header>
        "footer",          # generic <footer>
        "#main-header",    # Divi main header
        "#main-footer",    # Divi main footer
        ".et-l--header",   # Divi Theme Builder header layout
        ".et-l--footer",   # Divi Theme Builder footer layout
        ".et_pb_section_0_tb_footer",  # specific footer section class we saw
    ]:
        for el in body_tag.select(sel):
            el.decompose()

    # 2) Remove the right-hand sidebar column by class
    #    In your sample, it's the column with class "et_pb_column_1_tb_body"
    for el in body_tag.select(".et_pb_column_1_tb_body"):
        el.decompose()

    # 3) Remove cookie-consent / privacy components
    #    The snippet contains:
    #      - <div id="wpca-bar" ...>
    #      - <div id="wpca-trans-layer">
    #      - <div id="wpca-popup-modal">
    #      - <template id="wpca-placeholer-html"> ... </template>
    #      - and lots of elements with classes starting with "wpca-"
    for sel in [
        "#wpca-bar",
        "#wpca-trans-layer",
        "#wpca-popup-modal",
        "#wpca-placeholer-html",
    ]:
        for el in body_tag.select(sel):
            el.decompose()

    # Any element whose class list contains wpca-* is cookie UI, safe to drop.
    for el in list(body_tag.find_all(attrs={"class": True})):
        classes = el.get("class") or []
        if any(str(c).startswith("wpca-") for c in classes):
            el.decompose()

    # Extra safety: if any block contains the cookie text "Your Privacy" or
    # "Cookie settings", drop the enclosing cookie layout container.
    cookie_phrases = ["your privacy", "cookie settings", "save cookie settings"]
    to_remove = set()
    for txt in body_tag.find_all(string=True):
        t = (txt or "").strip().lower()
        if not t:
            continue
        if any(p in t for p in cookie_phrases):
            parent = txt.parent
            while parent is not None and parent is not body_tag:
                classes = parent.get("class") or []
                pid = parent.get("id") or ""
                if any(str(c).startswith("wpca-") for c in classes) or str(pid).startswith("wpca-"):
                    to_remove.add(parent)
                    break
                parent = parent.parent

    for el in to_remove:
        el.decompose()

    # 4) As a safety net, remove any ancestor blocks that clearly contain
    #    Quick Links / Required Reading / HR Hub labels, just in case.
    KEYWORDS = ["quick links", "required reading", "hr hub"]

    def _contains_keywords(text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in KEYWORDS)

    sidebar_blocks = set()

    # Scan text nodes and mark their ancestor columns/sections
    for txt in body_tag.find_all(
        string=lambda t: isinstance(t, NavigableString) and _contains_keywords(str(t))
    ):
        parent = txt.parent
        while parent is not None and parent is not body_tag:
            classes = parent.get("class") or []
            if any(
                str(c).startswith("et_pb_column")
                or str(c).startswith("et_pb_section")
                for c in classes
            ):
                sidebar_blocks.add(parent)
                break
            parent = parent.parent

    for el in sidebar_blocks:
        el.decompose()


def _export_url_to_folder(url: str) -> Path:
    """
    Download URL, clean OUT header/footer/sidebar/cookie chrome, download remaining images,
    convert to Markdown, and write a URL-style folder:

      <repo_root>/url_exports/<slug>/
        - page.html      (CLEANED BODY HTML ONLY)
        - page.md        (Markdown from cleaned body)
        - metadata.json
        - images/

    Returns the export folder path.
    """
    html, final_url = _download_url(url)
    title = _extract_page_title(html)
    print(f"📄 Page title: {title}")

    soup = BeautifulSoup(html, "html.parser")

    # Strip scripts/styles/noscript globally
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    body_tag = soup.body or soup  # use visible content

    # Strip template chrome BEFORE downloading images / markdown conversion
    _strip_insider_chrome(body_tag)

    # Prepare export directory
    URL_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug_base = sanitize_slug(title) or sanitize_slug(urlparse(final_url).netloc)
    export_dir = URL_EXPORTS_DIR / slug_base
    if export_dir.exists():
        export_dir = URL_EXPORTS_DIR / f"{slug_base}_{int(time.time())}"
    export_dir.mkdir(parents=True, exist_ok=True)

    images_dir = export_dir / "images"
    _rewrite_images_and_download(body_tag, final_url, images_dir)

    # CLEANED HTML (after removing header/footer/sidebar/cookies)
    cleaned_html = str(body_tag)

    # Save cleaned HTML as page.html
    html_path = export_dir / "page.html"
    html_path.write_text(cleaned_html, encoding="utf-8")
    print(f"✅ Wrote CLEANED PAGE page.html → {html_path}")

    # Markdown from cleaned HTML
    markdown_body = html_to_markdown(cleaned_html)

    safe_title = (title or "").replace('"', "'")
    fm = [
        "---",
        f'title: "{safe_title}"',
        f"source_url: {final_url}",
        f"exported_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "labels: []",
        "file_category: url",
        "processing_method: url_folder",
        "content_scope: article_only",
        "---",
        "",
    ]
    md_full = "\n".join(fm) + markdown_body.strip() + "\n"

    md_path = export_dir / "page.md"
    md_path.write_text(md_full, encoding="utf-8")
    print(f"✅ Wrote CLEANED PAGE page.md → {md_path}")

    # metadata.json
    meta = {
        "source_url": final_url,
        "title": title,
        "labels": [],
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_category": "url",
        "processing_method": "url_folder",
        "content_scope": "article_only",
    }
    (export_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✅ Wrote metadata.json → {export_dir / 'metadata.json'}")

    return export_dir


def _embed_exported_folder(export_dir: Path) -> Path:
    """
    Call URL-specific embedding logic on the exported folder and write FAISS into ../embedding.
    """
    EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {
        "embeddings": {
            "output_dir": str(EMBEDDINGS_OUTPUT_DIR),
            "models": EMBEDDING_MODELS,
            "device": EMBED_DEVICE,
            "device_name": EMBED_DEVICE_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "batch_size": BATCH_SIZE,
            "save_images_metadata": True,
        }
    }

    out_dir_name = f"{export_dir.name}__{int(time.time())}"
    out_dir = EMBEDDINGS_OUTPUT_DIR / out_dir_name

    print(f"\n🔗 Embedding URL folder (CLEANED ARTICLE ONLY): {export_dir}")
    create_embeddings_from_url_folder(str(export_dir), str(out_dir), cfg)
    print(f"✅ Saved FAISS index → {out_dir}")
    return out_dir


def main():
    print("=" * 72)
    print("🚀 PUBLIC URL → CLEANED ARTICLE EMBEDDINGS (Export → Folder → FAISS → access.csv)")
    print("=" * 72)

    # Init OCR + embedding model (from embedding_logic)
    try:
        initialize_ocr_engines(use_easyocr=True, use_tesseract=True)
        ok = initialize_embedding_model(
            model_options=EMBEDDING_MODELS,
            device=EMBED_DEVICE,
            device_name=EMBED_DEVICE_NAME,
        )
        if not ok:
            print("❌ Failed to initialize embedding model")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Initialization failure: {e}")
        sys.exit(1)

    # URL from CLI or prompt
    if len(sys.argv) > 1:
        url = sys.argv[1].strip()
    else:
        url = input("\nEnter the URL to embed: ").strip()

    if not url.lower().startswith(("http://", "https://")):
        print("⚠️ URL must start with http:// or https://")
        sys.exit(1)

    try:
        print("\n▶️ Exporting URL to local folder (CLEANED ARTICLE)…")
        export_dir = _export_url_to_folder(url)
    except Exception as e:
        print(f"❌ Failed to export URL: {e}")
        sys.exit(1)

    print("\n▶️ Building embeddings from CLEANED ARTICLE content…")
    t0 = time.time()
    out_dir = _embed_exported_folder(export_dir)
    dt = time.time() - t0
    print(f"⏱️ Embedding complete in {dt:.1f}s")

    # Update access.csv
    _update_access_csv_for_embedding(out_dir)

    print("\n" + "=" * 60)
    print("🎉 COMPLETE")
    print("=" * 60)
    print(f"Export dir: {export_dir}")
    print(f"Embeddings output dir: {EMBEDDINGS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
