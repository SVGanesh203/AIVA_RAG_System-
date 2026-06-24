# app.py

import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import mimetypes
from pathlib import Path
from threading import Thread

from flask import Flask, jsonify, send_file, abort, make_response
from flask_session import Session
from werkzeug.exceptions import NotFound

from routes.auth_routes import auth_bp
from routes.system_routes import system_bp
from routes.document_routes import document_bp
from routes.chat_routes import chat_bp
from config import config as app_config

# -------------------------------------------------------
# Flask App
# -------------------------------------------------------
app = Flask(__name__)

# Sessions
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-stable-secret-key-for-kb-bot')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'sessions')
app.config['SESSION_FILE_THRESHOLD'] = 100
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# -------------------------------------------------------
# Public Embedding Asset Route (no auth)
# Example: /embedding/<doc_name>/images/<file.png>
# Serves from: {PROJECT_ROOT}/embedding/...
# -------------------------------------------------------
@app.route('/embedding/<path:filename>')
def serve_embedding_asset(filename: str):
    """
    Publicly serves files from EMBEDDINGS_DIR, e.g.
      /embedding/<doc_name>/images/<file.png>

    - Prevents path traversal
    - Guesses content-type
    - Adds public cache headers
    - Prints resolved absolute path on 404 for easy debugging
    """
    base_dir = Path(app_config.EMBEDDINGS_DIR).resolve()
    # normalize input and prevent traversal
    safe_rel = Path(filename.lstrip("/\\"))
    abs_path = (base_dir / safe_rel).resolve()

    # stay inside base_dir
    if not str(abs_path).startswith(str(base_dir) + os.sep):
        abort(403)

    if not abs_path.is_file():
        print(f"[embedding 404] Tried: {abs_path}")  # helpful in console
        raise NotFound()

    ctype, _ = mimetypes.guess_type(str(abs_path))
    resp = make_response(send_file(str(abs_path), mimetype=ctype or 'application/octet-stream'))
    # Cache (tune if needed)
    resp.headers['Cache-Control'] = 'public, max-age=86400, immutable'
    return resp

# -------------------------------------------------------
# Initialize session
# -------------------------------------------------------
Session(app)

# -------------------------------------------------------
# Health Check (no auth required)
# -------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint that shows system status without requiring login"""
    from app_state import system_initialized, initialization_error
    if system_initialized:
        return jsonify({'status': 'healthy', 'initialized': True})
    else:
        return jsonify({
            'status': 'initializing',
            'initialized': False,
            'error': initialization_error
        }), 503

# -------------------------------------------------------
# Blueprints
# -------------------------------------------------------
app.register_blueprint(auth_bp)
app.register_blueprint(system_bp)
app.register_blueprint(document_bp, url_prefix='/api')
app.register_blueprint(chat_bp)

# Register Embedding Routes
from routes.embedding_routes import embedding_bp
app.register_blueprint(embedding_bp)

# -------------------------------------------------------
# CORS (dev)
# -------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# -------------------------------------------------------
# Errors
# -------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# -------------------------------------------------------
# System init
# -------------------------------------------------------
from app_state import initialize_system

if __name__ == '__main__':
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 60)
    print("🤖 RAG CHATBOT SERVER")
    print("=" * 60)
    print("🌐 Starting Flask server...")
    print("📱 Open http://localhost:9072 in your browser")

    # Ensure required directories exist
    embeddings_dir = Path(app_config.EMBEDDINGS_DIR)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    Path(app.config['SESSION_FILE_DIR']).mkdir(parents=True, exist_ok=True)

    # Log where we’re serving embedding assets from
    print(f"📂 EMBEDDINGS_DIR = {embeddings_dir.resolve()}")

    # Initialize background systems
    print("📄 Initializing system components in the background...")
    init_thread = Thread(target=initialize_system, daemon=True)
    init_thread.start()

    try:
        app.run(host='0.0.0.0', port=9072, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        raise
