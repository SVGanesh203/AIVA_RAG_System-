from flask import Blueprint, jsonify, session
from modules.auth import require_login
from app_state import get_system_state
from modules.utils import get_document_statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
system_bp = Blueprint('system', __name__, url_prefix='/api')

@system_bp.route('/status', methods=['GET'])
@require_login
def api_status():
    """Get the current system status"""
    try:
        # --- FIX: Unpack 7 values ---
        system_initialized, initialization_error, doc_manager, context_manager, app_config, _, _ = get_system_state()
        
        if not system_initialized:
            return jsonify({
                'status': 'error',
                'message': initialization_error or "System initialization failed. Check logs."
            }), 503
        
        loaded_count = len(doc_manager.loaded_documents) if doc_manager else 0
        
        return jsonify({
            'status': 'ready',
            'message': f'System is ready. {loaded_count} document(s) loaded.',
            'loaded_documents_count': loaded_count,
            'llm_model': app_config.LLM_MODEL,
            'embed_model': app_config.EMBED_MODEL,
            'device': app_config.DEVICE,
            'user_designation': session.get('user_designation', 'Unknown')
        })
    except Exception as e:
        logger.error(f"Error in /api/status: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@system_bp.route('/statistics', methods=['GET'])
@require_login
def api_get_stats():
    """Get statistics about loaded documents"""
    try:
        # --- FIX: Unpack 7 values ---
        system_initialized, _, doc_manager, _, _, _, _ = get_system_state()

        if not system_initialized or not doc_manager:
            return jsonify({'error': 'System not ready'}), 503
        
        stats = get_document_statistics(doc_manager)
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get statistics'}), 500