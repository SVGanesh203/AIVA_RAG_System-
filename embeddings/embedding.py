#!/usr/bin/env python3
"""
Minimal embedding.py (KBase-folder aware, writes to ../embedding + updates access.csv)

- Exports a single Confluence/KBase page (via list_kbase_pages.py)
- Immediately embeds the WHOLE exported folder (page.md + attachments/images)
- No external config loader
- After a successful embed, appends a column to access.csv and grants KBase access

Outputs FAISS indexes to: <repo root>/embedding   (sibling of catapult_chatbot)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# ---- Reuse your KBase exporter & selectors ----
try:
    from list_kbase_pages import (
        fetch_spaces,
        fetch_pages,
        get_page,
        sanitize_slug,
        export_page,
    )
except Exception as e:
    print(f"❌ Could not import from list_kbase_pages.py: {e}")
    sys.exit(1)

# ---- Embedding logic (folder-aware) ----
from embedding_logic import (
    initialize_embedding_model,
    initialize_ocr_engines,
    create_enhanced_embeddings_multi_format,   # must support folder ingestion
)

# ---- Access CSV updater ----
try:
    # We import a function so we can call it directly after embedding
    from edit_access import update_access_csv
except Exception:
    update_access_csv = None  # we'll warn later if import fails


# ===========================
# INLINE CONFIG (edit freely)
# ===========================
_THIS_FILE = Path(__file__).resolve()
_PROJECT_DIR = _THIS_FILE.parent               # .../catapult_chatbot
_REPO_ROOT  = _PROJECT_DIR.parent              # repo root (one level up)

EMBEDDINGS_OUTPUT_DIR = _REPO_ROOT / "embedding"     # <-- outside project dir
KBASE_EXPORTS_DIR     = _REPO_ROOT / "kbase_exports" # <-- outside project dir
ACCESS_CSV_PATH       = _REPO_ROOT / "access.csv"    # preferred
ACCESS_CSV_FALLBACK   = _PROJECT_DIR / "templates" / "access.csv"

EMBEDDING_MODELS = ["sentence-transformers/all-MiniLM-L6-v2"]
EMBED_DEVICE = "cpu"           # "cpu" or "cuda"
EMBED_DEVICE_NAME = "cpu"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 50
SAVE_IMAGES_METADATA = True

# Access CSV defaults
KBASE_DESIGNATION = "KBase"
GRANT_VALUE = 1
DEFAULT_VALUE = 0
# ===========================


def _resolve_export_dir(out_root: Path, slug: str, page_id: str) -> Optional[Path]:
    """
    Try the common folder namings used by exporters:
      - <slug>-<id>
      - <slug>_<id>
    Fallback: any directory in out_root that starts with slug and ends with id.
    """
    candidates = [
        out_root / f"{slug}-{page_id}",
        out_root / f"{slug}_{page_id}",
    ]
    for c in candidates:
        if c.is_dir():
            return c

    lowered_slug = slug.lower()
    pid = str(page_id)
    for p in out_root.iterdir():
        if p.is_dir():
            name = p.name.lower()
            if name.startswith(lowered_slug) and name.endswith(pid):
                return p
    return None


def _embed_exported_folder(export_dir: Path) -> Path:
    """
    Hand the entire exported KBase folder to the embedding pipeline.
    The logic should read page.md + attachments/images automatically.

    Returns the output directory where FAISS was saved.
    """
    EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {
        "embeddings": {
            "output_dir": str(EMBEDDINGS_OUTPUT_DIR),  # absolute, cross-platform
            "models": EMBEDDING_MODELS,
            "device": EMBED_DEVICE,
            "device_name": EMBED_DEVICE_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "batch_size": BATCH_SIZE,
            "save_images_metadata": SAVE_IMAGES_METADATA,
        }
    }

    out_dir_name = f"{export_dir.name}__{int(time.time())}"
    out_dir = EMBEDDINGS_OUTPUT_DIR / out_dir_name

    print(f"\n🔗 Embedding KBase folder: {export_dir}")
    create_enhanced_embeddings_multi_format(str(export_dir), str(out_dir), cfg)
    print(f"✅ Saved FAISS index → {out_dir}")
    return out_dir


def _pick_access_csv() -> Path:
    """
    Choose the access.csv path. Prefer repo-root/access.csv, else fallback to templates/access.csv.
    """
    if ACCESS_CSV_PATH.exists():
        return ACCESS_CSV_PATH
    if ACCESS_CSV_FALLBACK.exists():
        return ACCESS_CSV_FALLBACK
    # If neither exists, create an empty scaffold at repo-root with just Designation
    print("⚠️ access.csv not found. Creating a new one at repo root with only 'Designation' header.")
    ACCESS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCESS_CSV_PATH.write_text("Designation\n", encoding="utf-8", newline="\n")
    return ACCESS_CSV_PATH


def _update_access_csv_for_embedding(out_dir: Path, roles: list[str] = []) -> None:
    """
    After a successful embed, add a column with the embedding folder name
    and set selected roles=1, others=0.
    """
    column_name = out_dir.name  # e.g., sop_sending_..._24743694__1762438428
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
        create_backup=True,
    )
    if ok:
        print(f"✅ access.csv updated. {msg}")
    else:
        print(f"❌ Failed to update access.csv: {msg}")


def main():
    print("=" * 72)
    print("🚀 KBASE → EMBEDDINGS (Export → Immediate Folder Embedding → access.csv update)")
    print("=" * 72)

    # OCR + model init
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

    # === Same interactive selection you already use ===
    print("\n🔍 Fetching spaces…")
    spaces = fetch_spaces()
    if not spaces:
        print("⚠️ No spaces found or access denied.")
        sys.exit(0)

    for i, (key, name, _url) in enumerate(spaces, 1):
        print(f"{i:2d}. [{key}] {name}")

    space_key = input("\nEnter SPACE KEY: ").strip()
    if not any(s[0] == space_key for s in spaces):
        print(f"⚠️ Space '{space_key}' not found.")
        sys.exit(0)

    print(f"\n📄 Listing pages in [{space_key}] …")
    pages = fetch_pages(space_key)
    if not pages:
        print("⚠️ No pages found in that space (or access denied).")
        sys.exit(0)

    for i, (pid, title, url) in enumerate(pages, 1):
        print(f"{i:3d}. {title}  ({pid})")

    choice = input("\nEnter PAGE ID (exact) to export: ").strip()
    if not any(p[0] == choice for p in pages):
        print(f"⚠️ Page ID '{choice}' not in list.")
        sys.exit(0)

    # Export selected page to repo-root/kbase_exports
    out_root = KBASE_EXPORTS_DIR.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n▶️ Exporting page…")
    export_page(choice, str(out_root))

    # Locate the export folder (hyphen/underscore tolerant)
    page_meta = get_page(choice)
    title = page_meta.get("title", f"page-{choice}")
    slug = sanitize_slug(title)
    export_dir = _resolve_export_dir(out_root, slug, choice)

    if not export_dir:
        print(f"❌ Expected export folder not found under: {out_root}")
        print(f"   Tried patterns: {slug}-{choice} and {slug}_{choice}")
        sys.exit(1)

    # Immediately embed the entire folder → repo-root/embedding
    print("\n▶️ Building embeddings from exported content…")
    t0 = time.time()
    out_dir = _embed_exported_folder(export_dir)
    dt = time.time() - t0
    print(f"⏱️ Embedding complete in {dt:.1f}s")

    # Update access.csv with the new embedding column
    _update_access_csv_for_embedding(out_dir)

    print("\n" + "=" * 60)
    print("🎉 COMPLETE")
    print("=" * 60)
    print(f"Export dir: {export_dir}")
    print(f"Embeddings output dir: {EMBEDDINGS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
