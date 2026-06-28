
import os
import sys
import time
import csv
import json
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

# Add the 'embeddings' directory to sys.path so we can import the scripts
# Assumes structure:
#   repo/
#      catapult_chatbot/
#         app.py
#         routes/
#         embeddings/  <-- we need scripts from here
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
sys.path.append(str(EMBEDDINGS_DIR))

# Import logic from the embeddings scripts
# NOTE: using try-except to avoid crashing if path is wrong, 
# but in this env we know it exists.
try:
    import embed_public_url
    import list_kbase_pages
    import embedding
    import embedding_logic
    from unified_document_processor import MultiFormatDocumentProcessor
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    print(f"Error importing embedding scripts: {e}")

embedding_bp = Blueprint('embedding_bp', __name__)
logger = logging.getLogger(__name__)

# Ensure temp upload dir
UPLOAD_FOLDER = PROJECT_ROOT / "temp_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

@embedding_bp.route('/api/kbase/spaces', methods=['GET'])
def list_spaces():
    """List all available Confluence spaces."""
    try:
        spaces = list_kbase_pages.fetch_spaces()
        # spaces is a list of tuples (key, name, url)
        result = [{"key": s[0], "name": s[1]} for s in spaces]
        return jsonify({"success": True, "spaces": result})
    except Exception as e:
        logger.error(f"Failed to fetch spaces: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@embedding_bp.route('/api/roles', methods=['GET'])
def list_roles():
    """List all available designations from access.csv."""
    try:
        csv_path = embedding._pick_access_csv()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            roles = [row["Designation"].strip() for row in reader if row.get("Designation")]
        return jsonify({"success": True, "roles": roles})
    except Exception as e:
        logger.error(f"Failed to fetch roles: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@embedding_bp.route('/api/kbase/pages/<space_key>', methods=['GET'])
def list_pages(space_key):
    """List all pages in a given space."""
    try:
        pages = list_kbase_pages.fetch_pages(space_key)
        # pages is a list of tuples (pid, title, url)
        result = [{"id": p[0], "title": p[1]} for p in pages]
        return jsonify({"success": True, "pages": result})
    except Exception as e:
        logger.error(f"Failed to fetch pages for {space_key}: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@embedding_bp.route('/api/embed/url', methods=['POST'])
def embed_url():
    """
    Input: { "url": "https://..." }
    """
    data = request.json
    url = data.get('url')
    roles = data.get('roles', []) # Expecting a list of roles
    if not url:
        return jsonify({"success": False, "message": "No URL provided"}), 400

    try:
        # Replicate main() flow from embed_public_url.py but non-interactive
        
        # 1. Initialize models (idempotent, handled by logic)
        embedding_logic.initialize_ocr_engines(use_easyocr=True, use_tesseract=True)
        embedding_logic.initialize_embedding_model(
            model_options=embed_public_url.EMBEDDING_MODELS,
            device=embed_public_url.EMBED_DEVICE,
            device_name=embed_public_url.EMBED_DEVICE_NAME
        )

        # 2. Export
        logger.info(f"Exporting URL: {url}")
        export_dir = embed_public_url._export_url_to_folder(url)

        # 3. Embed
        logger.info(f"Embedding Exported Folder: {export_dir}")
        out_dir = embed_public_url._embed_exported_folder(export_dir)

        # 4. Access Update
        embed_public_url._update_access_csv_for_embedding(out_dir, roles=roles)

        return jsonify({
            "success": True, 
            "message": f"Successfully embedded URL: {url}",
            "doc_name": out_dir.name
        })

    except Exception as e:
        logger.error(f"URL Embedding failed: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@embedding_bp.route('/api/embed/kbase', methods=['POST'])
def embed_kbase():
    """
    Input: { "space": "KEY", "page_id": "12345" }
    """
    data = request.json
    space_key = data.get('space')
    page_id = data.get('page_id')
    roles = data.get('roles', [])

    if not page_id:
        return jsonify({"success": False, "message": "Page ID is required"}), 400

    try:
        # Replicate main() flow from embedding.py
        
        # 0. Init
        embedding_logic.initialize_ocr_engines(use_easyocr=True, use_tesseract=True)
        embedding_logic.initialize_embedding_model(
            model_options=embedding.EMBEDDING_MODELS,
            device=embedding.EMBED_DEVICE,
            device_name=embedding.EMBED_DEVICE_NAME
        )
        
        # 1. Export
        out_root = embedding.KBASE_EXPORTS_DIR.resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting Page ID: {page_id}")
        list_kbase_pages.export_page(page_id, str(out_root))

        # 2. Resolve Directory
        page_meta = list_kbase_pages.get_page(page_id)
        title = page_meta.get("title", f"page-{page_id}")
        slug = list_kbase_pages.sanitize_slug(title)
        
        export_dir = embedding._resolve_export_dir(out_root, slug, page_id)
        if not export_dir:
            raise FileNotFoundError(f"Could not locate export folder for {slug}-{page_id}")

        # 3. Embed
        logger.info(f"Embedding KBase Folder: {export_dir}")
        out_dir = embedding._embed_exported_folder(export_dir)

        # 4. Access
        embedding._update_access_csv_for_embedding(out_dir, roles=roles)

        return jsonify({
            "success": True, 
            "message": f"Successfully embedded Page: {title}",
            "doc_name": out_dir.name
        })

    except Exception as e:
        logger.error(f"KBase Embedding failed: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@embedding_bp.route('/api/embed/doc', methods=['POST'])
def embed_doc():
    """
    Input: multipart form with 'file' and 'roles'
    """
    roles = request.form.get('roles', '[]')
    try:
        roles = json.loads(roles)
    except:
        roles = []

    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(str(save_path))
        
        try:
            # Custom Logic for Doc Embedding since embedding_logic.py stubs it out
            
            # 0. Init
            embedding_logic.initialize_ocr_engines(use_easyocr=True, use_tesseract=True)
            embedding_logic.initialize_embedding_model(
                model_options=embedding.EMBEDDING_MODELS,
                device=embedding.EMBED_DEVICE,
                device_name=embedding.EMBED_DEVICE_NAME
            )

            # 1. Process File
            logger.info(f"Processing Document: {filename}")
            processor = MultiFormatDocumentProcessor()
            
            # Create a mock document object for the processor
            # The processor expects a path, so we pass it.
            # We need to manually chunk it using the processor.
            
            # The processor's 'create_chunks' expects a 'ProcessedDocument' object usually,
            # or we can use 'process_document' if available. 
            # Let's check unified_document_processor usage:
            # It has 'process_document(file_path)' -> ProcessedDocument
            
            processed_doc = processor.process_file(str(save_path))
            
            # 2. Chunk
            chunk_size = 1000
            chunk_overlap = 200
            chunks = processor.create_chunks(processed_doc, chunk_size, chunk_overlap)
            
            # Enhance metadata
            for ch in chunks:
                ch.metadata = embedding_logic.enhance_chunk_metadata(ch)
                ch.metadata['document_name'] = filename

            # 3. FAISS
            logger.info(f"Creating FAISS index for {len(chunks)} chunks")
            if not chunks:
                 return jsonify({"success": False, "message": "No text could be extracted from the document."}), 400

            db = FAISS.from_documents(chunks, embedding_logic.embedding_model)
            
            # 4. Save
            # Create a folder in embeddings dir
            doc_slug = list_kbase_pages.sanitize_slug(Path(filename).stem)
            ts = int(time.time())
            out_dir_name = f"{doc_slug}__{ts}"
            out_dir = embedding.EMBEDDINGS_OUTPUT_DIR / out_dir_name
            embedding_logic.ensure_directory(out_dir)
            
            db.save_local(str(out_dir))
            
            # Save metadata
            meta = {
                "title": filename,
                "author": "Uploaded User",
                "file_type": "document",
                "original_filename": filename,
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(out_dir / "metadata.txt", "w") as f:
                f.write(f"Document Name: {out_dir_name}\nTitle: {filename}\n")
            
            # 5. Access
            # Docs get same permissions as KBase for now (or public?)
            # Let's use the standard update function
            embedding._update_access_csv_for_embedding(out_dir, roles=roles)

            return jsonify({
                "success": True, 
                "message": f"Successfully embedded Document: {filename}",
                "doc_name": out_dir.name
            })

        except Exception as e:
            logger.error(f"Doc Embedding failed: {e}", exc_info=True)
            return jsonify({"success": False, "message": str(e)}), 500
        
        finally:
            # Cleanup upload
            try:
                os.remove(save_path)
            except:
                pass

    return jsonify({"success": False, "message": "Unknown error"}), 500
