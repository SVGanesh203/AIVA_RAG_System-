from flask import Blueprint, jsonify, request, session, send_from_directory
from modules.auth import require_login, get_user_access
from app_state import get_system_state, app_config
import os
import json
import logging
import traceback
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
document_bp = Blueprint('document', __name__)

# ---------- helpers ----------
def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _load_images_meta(doc_dir: Path):
    f = doc_dir / "images" / "images_metadata.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _safe_join(base: Path, *parts: str) -> Path:
    """
    Join parts under base and ensure the result stays inside base.
    Allows nested paths like 'subdir/image.png' but forbids traversal.
    """
    final = base.joinpath(*parts).resolve()
    base_resolved = base.resolve()
    if not str(final).startswith(str(base_resolved)):
        raise PermissionError("Path traversal detected")
    return final

# ---------- routes ----------

@document_bp.route('/documents', methods=['GET'])
@require_login
def api_documents():
    """Get available documents information filtered by user access"""
    try:
        system_initialized, _, doc_manager, _, _, _, _ = get_system_state()
        if not system_initialized:
            return jsonify({'error': 'System not properly initialized'}), 500

        user_designation = session.get('user_designation')
        if not user_designation:
            return jsonify({'error': 'Not logged in'}), 401

        user_access = get_user_access(user_designation) or []
        if not os.path.exists(app_config.EMBEDDINGS_DIR):
            return jsonify({'documents': [], 'total_count': 0})

        documents = []
        for folder in sorted(os.listdir(app_config.EMBEDDINGS_DIR)):
            folder_path = Path(app_config.EMBEDDINGS_DIR) / folder
            if not folder_path.is_dir():
                continue
            if folder not in user_access:
                continue

            doc_info = {
                'name': folder,
                'title': folder.replace('_', ' ').title(),
                'author': 'Unknown',
                'pages': 0,
                'chunks': 0,
                'images': 0,
                'tables': 0
            }

            # Prefer images count from images_metadata.json
            try:
                images = _load_images_meta(folder_path)
                doc_info['images'] = len(images)
            except Exception:
                pass

            # Parse metadata.txt (optional extras)
            metadata_file = folder_path / "metadata.txt"
            if metadata_file.exists():
                try:
                    for raw in _read_text_safe(metadata_file).splitlines():
                        line = raw.strip()
                        if line.startswith("Author: "):
                            doc_info['author'] = line.replace("Author: ", "")
                        elif line.startswith("Title: "):
                            title = line.replace("Title: ", "")
                            if title in ["PowerPoint Presentation", "Microsoft PowerPoint", "Presentation"]:
                                doc_info['title'] = folder
                            else:
                                doc_info['title'] = title
                        elif line.startswith("Total Pages: "):
                            try:
                                doc_info['pages'] = int(line.replace("Total Pages: ", ""))
                            except ValueError:
                                pass
                        elif line.startswith("Total Chunks: "):
                            try:
                                doc_info['chunks'] = int(line.replace("Total Chunks: ", ""))
                            except ValueError:
                                pass
                        elif line.startswith("Tables: "):
                            try:
                                doc_info['tables'] = int(line.replace("Tables: ", ""))
                            except ValueError:
                                pass
                        # We ignore "Images:" here because we now trust images_metadata.json
                except Exception as e:
                    logger.warning(f"Error reading metadata for {folder}: {e}")

            documents.append(doc_info)

        return jsonify({
            'documents': documents,
            'total_count': len(documents),
            'user_designation': user_designation,
            'accessible_documents': user_access
        })

    except Exception as e:
        logger.error(f"Documents API error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to load documents', 'message': str(e)}), 500


@document_bp.route('/load-documents', methods=['POST'])
@require_login
def api_load_documents():
    """Load selected documents"""
    system_initialized, _, doc_manager, _, _, _, _ = get_system_state()
    if not system_initialized:
        return jsonify({'status': 'error', 'message': 'System not initialized'}), 500

    try:
        user_designation = session.get('user_designation')
        if not user_designation:
            return jsonify({'error': 'Not logged in'}), 401

        user_access = get_user_access(user_designation) or []

        data = request.get_json()
        if not data or 'documents' not in data:
            return jsonify({'error': 'No documents specified'}), 400

        document_names = data.get('documents', [])
        if not isinstance(document_names, list) or not document_names:
            return jsonify({'error': 'Invalid document list'}), 400

        unauthorized_docs = [doc for doc in document_names if doc not in user_access]
        if unauthorized_docs:
            return jsonify({
                'error': 'Access denied',
                'message': f'You do not have access to: {", ".join(unauthorized_docs)}',
                'unauthorized_documents': unauthorized_docs
            }), 403

        loaded_count = 0
        failed_docs = []

        doc_manager.loaded_documents.clear()
        doc_manager.search_cache.clear()

        for doc_name in document_names:
            folder_path = os.path.join(app_config.EMBEDDINGS_DIR, doc_name)
            if os.path.exists(folder_path):
                try:
                    db, metadata, stats = doc_manager._load_single_document(folder_path)
                    if db:
                        doc_manager.loaded_documents[doc_name] = {
                            'db': db,
                            'metadata': metadata,
                            'statistics': stats,
                            'folder_path': folder_path
                        }
                        loaded_count += 1
                        logger.info(f"✅ Loaded document: {doc_name}")
                    else:
                        failed_docs.append(doc_name)
                except Exception as e:
                    logger.error(f"❌ Failed to load {doc_name}: {e}")
                    failed_docs.append(doc_name)
            else:
                logger.warning(f"❌ Folder not found: {folder_path}")
                failed_docs.append(doc_name)

        if loaded_count == 0:
            return jsonify({
                'status': 'error',
                'message': 'No documents could be loaded',
                'failed_documents': failed_docs
            }), 400

        response_data = {
            'status': 'success',
            'message': f'Successfully loaded {loaded_count} document(s)',
            'loaded_documents': [name for name in document_names if name not in failed_docs],
            'failed_documents': failed_docs,
            'loaded_count': loaded_count
        }

        if failed_docs:
            response_data['warning'] = f'Failed to load: {", ".join(failed_docs)}'

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in api_load_documents: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Failed to load documents: {str(e)}'}), 500



@document_bp.route('/load-all-documents', methods=['POST'])
@require_login
def api_load_all_documents():
    """Load all documents accessible to the user"""
    system_initialized, _, doc_manager, _, _, _, _ = get_system_state()
    if not system_initialized:
        return jsonify({'status': 'error', 'message': 'System not initialized'}), 500

    try:
        user_designation = session.get('user_designation')
        if not user_designation:
            return jsonify({'error': 'Not logged in'}), 401

        user_access = get_user_access(user_designation) or []
        if not user_access:
            return jsonify({'status': 'error', 'message': 'No documents are accessible for your designation'}), 400

        if not os.path.exists(app_config.EMBEDDINGS_DIR):
            return jsonify({'status': 'error', 'message': 'No document embeddings directory found'}), 400

        folders = [f for f in os.listdir(app_config.EMBEDDINGS_DIR)
                   if os.path.isdir(os.path.join(app_config.EMBEDDINGS_DIR, f))]

        accessible_folders = [f for f in folders if f in user_access]
        if not accessible_folders:
            return jsonify({'status': 'error', 'message': 'No documents are accessible for your designation'}), 400

        # Fast path: all accessible documents are already loaded (e.g. from startup)
        already_loaded = [
            f for f in accessible_folders if f in doc_manager.loaded_documents
        ]
        if len(already_loaded) == len(accessible_folders):
            return jsonify({
                'status': 'success',
                'message': f'All {len(already_loaded)} accessible document(s) already loaded',
                'loaded_documents': already_loaded,
                'failed_documents': [],
                'loaded_count': len(already_loaded),
                'total_accessible': len(accessible_folders),
            })

        loaded_count = 0
        failed_docs = []

        # Only load documents that are not yet in memory; keep existing indices
        folders_to_load = [
            f for f in accessible_folders if f not in doc_manager.loaded_documents
        ]

        for doc_name in folders_to_load:
            folder_path = os.path.join(app_config.EMBEDDINGS_DIR, doc_name)
            try:
                db, metadata, stats = doc_manager._load_single_document(folder_path)
                if db:
                    doc_manager.loaded_documents[doc_name] = {
                        'db': db,
                        'metadata': metadata,
                        'statistics': stats,
                        'folder_path': folder_path
                    }
                    loaded_count += 1
                    logger.info(f"✅ Loaded document: {doc_name}")
                else:
                    failed_docs.append(doc_name)
            except Exception as e:
                logger.error(f"❌ Failed to load {doc_name}: {e}")
                failed_docs.append(doc_name)

        total_in_memory = len([
            f for f in accessible_folders if f in doc_manager.loaded_documents
        ])
        if total_in_memory == 0:
            return jsonify({
                'status': 'error',
                'message': 'No documents could be loaded',
                'failed_documents': failed_docs
            }), 400

        response_data = {
            'status': 'success',
            'message': f'{total_in_memory} accessible document(s) ready',
            'loaded_documents': [
                name for name in accessible_folders if name not in failed_docs
            ],
            'failed_documents': failed_docs,
            'loaded_count': total_in_memory,
            'total_accessible': len(accessible_folders)
        }
        if failed_docs:
            response_data['warning'] = f'Failed to load: {", ".join(failed_docs)}'
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in api_load_all_documents: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Failed to load documents: {str(e)}'}), 500

@document_bp.route('/get-image/<path:doc_name>/<path:relpath>', methods=['GET'])
@require_login
def get_document_image(doc_name, relpath):
    """
    Serve images only from:
        <EMBEDDINGS_DIR>/<doc_name>/images/<relpath>

    Accepts either:
        /api/get-image/<doc>/image-2024-11-14_17-11-31.png
    or
        /api/get-image/<doc>/images/image-2024-11-14_17-11-31.png
    """
    try:
        # --- Access check ---
        user_designation = session.get('user_designation')
        user_access = get_user_access(user_designation) or []
        if doc_name not in user_access:
            return jsonify({'error': 'Access denied to this document'}), 403

        # Normalize relpath and strip any leading slashes
        relpath = relpath.strip().lstrip("/\\")
        # If caller included "images/" prefix, drop it so we always join under images/
        if relpath.lower().startswith("images/"):
            relpath = relpath[len("images/"):]

        # Only image extensions allowed
        if not re.search(r'\.(png|jpe?g|gif|bmp|webp)$', relpath, re.IGNORECASE):
            return jsonify({'error': 'Invalid image filename'}), 400

        embeddings_dir = Path(str(app_config.EMBEDDINGS_DIR)).resolve()  # e.g. .../catapult_chatbot/embedding
        doc_images_dir = (embeddings_dir / doc_name / "images").resolve()
        target_path = (doc_images_dir / relpath).resolve()

        # Security: must stay inside the document's images directory
        if not str(target_path).startswith(str(doc_images_dir) + os.sep):
            logger.warning(f"Traversal attempt: {target_path}")
            return jsonify({'error': 'Access denied'}), 403

        if not target_path.exists() or not target_path.is_file():
            return jsonify({'error': 'Image not found'}), 404

        return send_from_directory(target_path.parent.as_posix(), target_path.name)

    except Exception as e:
        logger.error(f"Error serving image: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500