# app_core/routes/api.py
from flask import Blueprint, jsonify, session
from ..init_system import (
    system_initialized,
    initialization_error,
    doc_manager,
    embedding_model,
    llm,
    DEVICE,
)
from ..access import get_user_access
from config import RAGConfig

api_bp = Blueprint("api", __name__, url_prefix="/api")

@api_bp.get("/status")
def status():
    """
    Returns overall system status the sidebar expects.
    """
    # Capabilities snapshot from your config
    caps = RAGConfig.get_capability_status()
    # How many documents are currently loaded
    doc_count = 0
    if doc_manager and getattr(doc_manager, "loaded_documents", None):
        doc_count = len(doc_manager.loaded_documents)

    return jsonify({
        "ok": system_initialized and initialization_error is None,
        "error": initialization_error,
        "device": DEVICE,
        "embed_model": getattr(RAGConfig, "EMBED_MODEL", None),
        "llm_model": getattr(RAGConfig, "LLM_MODEL", None),
        "has_embeddings": embedding_model is not None,
        "has_llm": llm is not None,
        "documents_loaded": doc_count,
        "capabilities": caps,
        "user": {
            "designation": session.get("designation"),
        },
    })

@api_bp.get("/documents")
def list_documents():
    """
    Returns the list of documents available to the *current* user (based on session designation).
    - Uses doc_manager.loaded_documents keys
    - Intersects with role access (from access.csv)
    """
    # No docs if system not up yet
    if not (doc_manager and getattr(doc_manager, "loaded_documents", None)):
        return jsonify({"documents": []})

    all_docs = sorted(list(doc_manager.loaded_documents.keys()))
    designation = session.get("designation")

    # If no designation in session, return empty (protected)
    if not designation:
        return jsonify({"documents": []}), 401

    # Admin → full access; otherwise intersect with CSV-defined access
    if designation.lower() == "admin":
        allowed_docs = all_docs
    else:
        allowed_from_csv = set(get_user_access(designation, filename="access.csv"))
        # Keep only those actually loaded
        allowed_docs = [d for d in all_docs if d in allowed_from_csv]

    # Return simple shape the frontend can iterate
    # You can enrich with metadata pulled from doc_manager.loaded_documents[d]['metadata'] if needed.
    return jsonify({
        "documents": [
            {"id": name, "name": name}
            for name in allowed_docs
        ]
    })
