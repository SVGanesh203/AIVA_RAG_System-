"""
agents/agent_registry.py

Persistent registry for AIVA agents.
Agents are stored in agents/agents_db.json — no code changes needed to add/remove agents.
Each agent has:
  - id           : unique slug  (e.g. "italian_recipes")
  - name         : display name (e.g. "Italian Recipes")
  - description  : what this agent knows about
  - keywords     : list of trigger words/phrases for query routing
  - sources      : list of {type, value} dicts
                     type = "file"  → value = absolute path
                     type = "url"   → value = URL string
  - enabled      : bool — admin can toggle without deleting
  - refresh_hours: how often URL sources are re-indexed (0 = never)
  - created_at   : ISO timestamp
  - last_indexed : ISO timestamp or null
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

AGENTS_DIR = Path(__file__).parent
AGENTS_DB  = AGENTS_DIR / "agents_db.json"


def _now() -> str:
    return datetime.utcnow().isoformat()


def _load() -> List[Dict]:
    if not AGENTS_DB.exists():
        return []
    try:
        with open(AGENTS_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(agents: List[Dict]) -> None:
    AGENTS_DB.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENTS_DB, "w", encoding="utf-8") as f:
        json.dump(agents, f, indent=2, ensure_ascii=False)


# ── Public API ────────────────────────────────────────────────────────────────

def list_agents(include_disabled: bool = True) -> List[Dict]:
    agents = _load()
    if not include_disabled:
        agents = [a for a in agents if a.get("enabled", True)]
    return agents


def get_agent(agent_id: str) -> Optional[Dict]:
    for a in _load():
        if a["id"] == agent_id:
            return a
    return None


def create_agent(
    name: str,
    description: str,
    keywords: List[str],
    sources: List[Dict],          # [{"type": "file"|"url", "value": "..."}]
    refresh_hours: int = 24,
    enabled: bool = True,
) -> Dict:
    """Create a new agent and persist it."""
    agents = _load()

    # Build a slug id from name
    slug = name.lower().replace(" ", "_").replace("-", "_")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    # Ensure uniqueness
    existing_ids = {a["id"] for a in agents}
    candidate = slug
    counter = 2
    while candidate in existing_ids:
        candidate = f"{slug}_{counter}"
        counter += 1

    agent = {
        "id": candidate,
        "name": name,
        "description": description,
        "keywords": [k.lower().strip() for k in keywords],
        "sources": sources,
        "enabled": enabled,
        "refresh_hours": refresh_hours,
        "created_at": _now(),
        "last_indexed": None,
    }
    agents.append(agent)
    _save(agents)
    return agent


def update_agent(agent_id: str, updates: Dict) -> Optional[Dict]:
    """Update any fields of an agent."""
    agents = _load()
    for i, a in enumerate(agents):
        if a["id"] == agent_id:
            # Protect id and created_at
            updates.pop("id", None)
            updates.pop("created_at", None)
            if "keywords" in updates:
                updates["keywords"] = [k.lower().strip() for k in updates["keywords"]]
            agents[i] = {**a, **updates}
            _save(agents)
            return agents[i]
    return None


def delete_agent(agent_id: str) -> bool:
    agents = _load()
    new = [a for a in agents if a["id"] != agent_id]
    if len(new) == len(agents):
        return False
    _save(new)
    return True


def set_enabled(agent_id: str, enabled: bool) -> Optional[Dict]:
    return update_agent(agent_id, {"enabled": enabled})


def mark_indexed(agent_id: str) -> None:
    update_agent(agent_id, {"last_indexed": _now()})


def needs_refresh(agent: Dict) -> bool:
    """Return True if agent's URL sources are due for re-indexing."""
    hours = agent.get("refresh_hours", 0)
    if hours <= 0:
        return False
    last = agent.get("last_indexed")
    if last is None:
        return True
    delta = (datetime.utcnow() - datetime.fromisoformat(last)).total_seconds() / 3600
    return delta >= hours
