"""
routes/agent_routes.py

Admin-only REST API for managing AIVA agents.

Endpoints:
  GET    /api/agents                    — list all agents + index status
  POST   /api/agents                    — create agent
  GET    /api/agents/<id>               — get single agent
  PUT    /api/agents/<id>               — update agent
  DELETE /api/agents/<id>               — delete agent
  POST   /api/agents/<id>/enable        — enable agent
  POST   /api/agents/<id>/disable       — disable agent
  POST   /api/agents/<id>/index         — (re)index agent now (sync)
  GET    /api/agents/<id>/status        — index + source status
  POST   /api/agents/debug/route        — show routing scores for a query
  POST   /api/agents/<id>/upload-source — upload a file as a new source
"""

import os
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request

from modules.auth import require_login
from agents.agent_registry import (
    list_agents, get_agent, create_agent, update_agent,
    delete_agent, set_enabled
)
from agents.agent_indexer import agent_is_indexed
from app_state import get_agent_manager

logger = logging.getLogger(__name__)

agent_bp = Blueprint("agents", __name__, url_prefix="/api/agents")

UPLOADS_DIR = Path(__file__).parent.parent / "agent_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _mgr():
    mgr = get_agent_manager()
    if mgr is None:
        return None, jsonify({"error": "Agent manager not ready"}), 503
    return mgr, None, None


# ── List / Create ─────────────────────────────────────────────────────────────

@agent_bp.route("", methods=["GET"])
@require_login
def list_all_agents():
    agents = list_agents(include_disabled=True)
    result = []
    for a in agents:
        result.append({
            **a,
            "is_indexed": agent_is_indexed(a["id"]),
        })
    return jsonify({"agents": result, "total": len(result)})


@agent_bp.route("", methods=["POST"])
@require_login
def create_new_agent():
    data = request.get_json(silent=True) or {}

    name        = data.get("name", "").strip()
    description = data.get("description", "").strip()
    keywords    = data.get("keywords", [])
    sources     = data.get("sources", [])          # [{type, value}]
    refresh_h   = int(data.get("refresh_hours", 24))
    enabled     = bool(data.get("enabled", True))

    if not name:
        return jsonify({"error": "Agent name is required"}), 400
    if not keywords:
        return jsonify({"error": "At least one keyword is required"}), 400

    agent = create_agent(
        name=name,
        description=description,
        keywords=keywords,
        sources=sources,
        refresh_hours=refresh_h,
        enabled=enabled,
    )

    # Kick off async indexing if sources provided
    if sources:
        mgr = get_agent_manager()
        if mgr:
            mgr.ensure_indexed(agent["id"], force=True)

    return jsonify({"agent": agent, "message": "Agent created. Indexing started in background."}), 201


# ── Single agent ──────────────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>", methods=["GET"])
@require_login
def get_one_agent(agent_id):
    agent = get_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"agent": {**agent, "is_indexed": agent_is_indexed(agent_id)}})


@agent_bp.route("/<agent_id>", methods=["PUT"])
@require_login
def update_one_agent(agent_id):
    data = request.get_json(silent=True) or {}
    updated = update_agent(agent_id, data)
    if not updated:
        return jsonify({"error": "Agent not found"}), 404

    # If sources changed, re-index
    if "sources" in data:
        mgr = get_agent_manager()
        if mgr:
            mgr.ensure_indexed(agent_id, force=True)

    return jsonify({"agent": updated, "message": "Agent updated."})


@agent_bp.route("/<agent_id>", methods=["DELETE"])
@require_login
def delete_one_agent(agent_id):
    ok = delete_agent(agent_id)
    if not ok:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"message": f"Agent '{agent_id}' deleted."})


# ── Enable / Disable ─────────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>/enable", methods=["POST"])
@require_login
def enable_agent(agent_id):
    updated = set_enabled(agent_id, True)
    if not updated:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"message": f"Agent '{agent_id}' enabled.", "agent": updated})


@agent_bp.route("/<agent_id>/disable", methods=["POST"])
@require_login
def disable_agent(agent_id):
    updated = set_enabled(agent_id, False)
    if not updated:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({"message": f"Agent '{agent_id}' disabled.", "agent": updated})


# ── Indexing ──────────────────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>/index", methods=["POST"])
@require_login
def reindex_agent(agent_id):
    agent = get_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    mgr = get_agent_manager()
    if not mgr:
        return jsonify({"error": "Agent manager not ready"}), 503

    ok = mgr.index_agent_sync(agent_id)
    if ok:
        return jsonify({"message": f"Agent '{agent_id}' indexed successfully."})
    else:
        return jsonify({"error": "Indexing failed. Check server logs."}), 500


@agent_bp.route("/<agent_id>/status", methods=["GET"])
@require_login
def agent_status(agent_id):
    agent = get_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404
    return jsonify({
        "agent_id":    agent_id,
        "name":        agent["name"],
        "enabled":     agent["enabled"],
        "is_indexed":  agent_is_indexed(agent_id),
        "last_indexed": agent.get("last_indexed"),
        "sources":     agent.get("sources", []),
    })


# ── File upload as source ─────────────────────────────────────────────────────

@agent_bp.route("/<agent_id>/upload-source", methods=["POST"])
@require_login
def upload_source_file(agent_id):
    """
    Upload a file (PDF, TXT, CSV, MD) and add it as a source for this agent.
    The file is saved to agent_uploads/<agent_id>/ and the source entry is added.
    """
    agent = get_agent(agent_id)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    save_dir = UPLOADS_DIR / agent_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f.filename
    f.save(str(save_path))

    # Append source to agent
    sources = agent.get("sources", [])
    sources.append({"type": "file", "value": str(save_path)})
    update_agent(agent_id, {"sources": sources})

    # Trigger re-index
    mgr = get_agent_manager()
    if mgr:
        mgr.ensure_indexed(agent_id, force=True)

    return jsonify({
        "message": f"File '{f.filename}' uploaded and indexing started.",
        "file_path": str(save_path),
    })


# ── Debug routing ─────────────────────────────────────────────────────────────

@agent_bp.route("/debug/route", methods=["POST"])
@require_login
def debug_routing():
    """Show which agents would be selected for a given query and why."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query field required"}), 400

    mgr = get_agent_manager()
    if not mgr:
        return jsonify({"error": "Agent manager not ready"}), 503

    explanation = mgr.get_routing_debug(query)
    return jsonify({"query": query, "routing": explanation})
