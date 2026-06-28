"""
agents/agent_manager.py

High-level orchestrator for the AIVA agent system.

Workflow per query:
  1. route_query()          → find matching enabled agents
  2. For each matched agent → search_agent_index()
  3. Merge all results      → deduplicate + rank by score
  4. Return augmented_context string ready to inject into RAG prompt

Also handles:
  - On-demand indexing (if agent not yet indexed)
  - Background refresh scheduler (checks refresh_hours)
"""

import logging
import threading
from typing import List, Dict, Optional, Tuple

from agents.agent_registry import (
    list_agents, get_agent, mark_indexed, needs_refresh, agent_is_indexed
)
from agents.agent_indexer  import index_agent, search_agent_index
from agents.query_router   import route_query

logger = logging.getLogger(__name__)

# How many top chunks to take per agent
CHUNKS_PER_AGENT = 5
# Hard cap on total context characters fed to RAG
MAX_CONTEXT_CHARS = 6000


class AgentManager:
    """
    Singleton-style manager. Instantiate once at app startup and share the instance.
    Requires the same embedding_model used by the main RAG system.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self._lock = threading.Lock()
        self._index_threads: Dict[str, threading.Thread] = {}
        logger.info("[AgentManager] Initialised.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def ensure_indexed(self, agent_id: str, force: bool = False) -> bool:
        """
        Index an agent if it hasn't been indexed yet (or force=True).
        Non-blocking — starts a background thread, returns immediately.
        Returns True if indexing was triggered, False if already up-to-date.
        """
        agent = get_agent(agent_id)
        if agent is None:
            logger.warning(f"[AgentManager] Agent not found: {agent_id}")
            return False

        if not force and agent_is_indexed(agent_id) and not needs_refresh(agent):
            return False

        with self._lock:
            if agent_id in self._index_threads and self._index_threads[agent_id].is_alive():
                logger.info(f"[AgentManager] Already indexing '{agent_id}', skipping duplicate.")
                return False

            def _run():
                logger.info(f"[AgentManager] Background indexing: '{agent_id}'")
                ok = index_agent(agent, self.embedding_model)
                if ok:
                    mark_indexed(agent_id)
                    logger.info(f"[AgentManager] Indexing complete: '{agent_id}'")
                else:
                    logger.warning(f"[AgentManager] Indexing failed: '{agent_id}'")

            t = threading.Thread(target=_run, daemon=True, name=f"idx-{agent_id}")
            self._index_threads[agent_id] = t
            t.start()
            return True

    def index_agent_sync(self, agent_id: str) -> bool:
        """Synchronous indexing — blocks until done. Used by admin routes."""
        agent = get_agent(agent_id)
        if agent is None:
            return False
        ok = index_agent(agent, self.embedding_model)
        if ok:
            mark_indexed(agent_id)
        return ok

    def refresh_all(self):
        """Check all agents and re-index those that are due. Called by scheduler."""
        agents = list_agents(include_disabled=False)
        for agent in agents:
            if needs_refresh(agent):
                logger.info(f"[AgentManager] Refresh due for '{agent['id']}'")
                self.ensure_indexed(agent["id"], force=True)

    # ── Query pipeline ────────────────────────────────────────────────────────

    def get_augmented_context(
        self,
        query: str,
        max_agents: int = 5,
        threshold: float = 0.15,
    ) -> Tuple[str, List[Dict]]:
        """
        Main entry point called from chat_routes.

        Returns:
            (augmented_context_str, agent_metadata_list)

            augmented_context_str — plain text ready to prepend to the RAG prompt.
            agent_metadata_list   — list of {agent_id, agent_name, source, score}
                                    for transparency / source attribution.
        """
        enabled_agents = list_agents(include_disabled=False)
        if not enabled_agents:
            return "", []

        # 1. Route
        matched = route_query(
            query,
            enabled_agents,
            embedding_model=self.embedding_model,
            threshold=threshold,
            max_agents=max_agents,
        )

        if not matched:
            logger.debug(f"[AgentManager] No agents matched query: {query[:60]}")
            return "", []

        logger.info(
            f"[AgentManager] Matched agents: "
            + ", ".join(f"{a['id']}({s:.2f})" for a, s in matched)
        )

        # 2. Search each matched agent
        all_chunks: List[Dict] = []
        for agent, route_score in matched:
            agent_id = agent["id"]

            # Auto-index if missing (async; won't have results this call)
            if not agent_is_indexed(agent_id):
                logger.info(f"[AgentManager] Agent '{agent_id}' not indexed — triggering background index.")
                self.ensure_indexed(agent_id)
                continue

            chunks = search_agent_index(
                agent_id,
                query,
                self.embedding_model,
                k=CHUNKS_PER_AGENT,
            )
            for c in chunks:
                c["agent_name"]   = agent["name"]
                c["route_score"]  = route_score
                c["final_score"]  = route_score * 0.4 + (1 - c["score"]) * 0.6
            all_chunks.extend(chunks)

        if not all_chunks:
            return "", []

        # 3. Deduplicate (by text fingerprint) and rank
        seen: set = set()
        unique_chunks: List[Dict] = []
        for c in sorted(all_chunks, key=lambda x: x.get("final_score", 0), reverse=True):
            fp = c["text"][:120]
            if fp not in seen:
                seen.add(fp)
                unique_chunks.append(c)

        # 4. Build context string (respecting MAX_CONTEXT_CHARS)
        sections: List[str] = []
        total_chars = 0
        meta_list: List[Dict] = []

        for c in unique_chunks:
            block = (
                f"[Agent: {c['agent_name']} | Source: {c['source']}]\n"
                f"{c['text']}"
            )
            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                break
            sections.append(block)
            total_chars += len(block)
            meta_list.append({
                "agent_id":   c["agent_id"],
                "agent_name": c["agent_name"],
                "source":     c["source"],
                "score":      round(c.get("final_score", 0), 3),
            })

        if not sections:
            return "", []

        context = (
            "=== AGENT-RETRIEVED CONTEXT ===\n"
            + "\n\n---\n\n".join(sections)
            + "\n=== END AGENT CONTEXT ===\n"
        )
        return context, meta_list

    def get_routing_debug(self, query: str) -> List[Dict]:
        """Return routing explanation for admin debug view."""
        from agents.query_router import explain_routing
        enabled = list_agents(include_disabled=False)
        return explain_routing(query, enabled, self.embedding_model)

    def status(self) -> Dict:
        """Return status of all agents (for admin panel)."""
        from agents.agent_indexer import agent_is_indexed
        agents = list_agents()
        result = []
        for a in agents:
            result.append({
                **a,
                "is_indexed": agent_is_indexed(a["id"]),
            })
        return {"agents": result, "total": len(result)}
