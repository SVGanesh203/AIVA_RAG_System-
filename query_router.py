"""
agents/query_router.py

Routes an incoming user query to one or more relevant agents.

Routing strategy (layered, fast → accurate):
  1. Keyword match   — agent.keywords vs query tokens           (fast)
  2. Description     — embedding cosine similarity              (semantic)
  3. Merge & rank    — combine scores, return all above threshold

The router returns a ranked list of (agent, score) tuples.
Callers can take the top-N or all above a threshold.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# Minimum combined score to include an agent in routing results
KEYWORD_WEIGHT    = 0.6
SEMANTIC_WEIGHT   = 0.4
MATCH_THRESHOLD   = 0.15   # very permissive — better to over-retrieve than miss


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer, strips punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_score(query_tokens: List[str], agent: Dict) -> float:
    """
    Fraction of the agent's keywords that appear in the query.
    Also checks if query tokens appear in keywords (bidirectional).
    """
    agent_keywords = agent.get("keywords", [])
    if not agent_keywords:
        return 0.0

    query_set = set(query_tokens)
    hits = 0
    for kw in agent_keywords:
        kw_tokens = set(_tokenize(kw))
        # full keyword phrase match OR any token from keyword in query
        if kw_tokens.issubset(query_set) or kw_tokens & query_set:
            hits += 1

    return hits / len(agent_keywords)


def _semantic_score(query: str, agent: Dict, embedding_model) -> float:
    """
    Cosine similarity between query embedding and agent description embedding.
    Returns 0.0 if embedding model not available.
    """
    if embedding_model is None:
        return 0.0
    try:
        import numpy as np
        desc = agent.get("description", "") + " " + " ".join(agent.get("keywords", []))
        q_emb  = embedding_model.embed_query(query)
        d_emb  = embedding_model.embed_query(desc)
        q_arr  = np.array(q_emb)
        d_arr  = np.array(d_emb)
        cosine = float(np.dot(q_arr, d_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(d_arr) + 1e-9))
        # Normalise from [-1,1] to [0,1]
        return (cosine + 1) / 2
    except Exception as e:
        logger.warning(f"[router] Semantic scoring failed: {e}")
        return 0.0


def route_query(
    query: str,
    agents: List[Dict],
    embedding_model=None,
    threshold: float = MATCH_THRESHOLD,
    max_agents: int = 10,
) -> List[Tuple[Dict, float]]:
    """
    Given a user query and list of enabled agents, return a ranked list of
    (agent, combined_score) tuples above `threshold`, best first.

    Args:
        query          : raw user message
        agents         : list of agent dicts (from registry, already filtered to enabled)
        embedding_model: HuggingFace embeddings instance (can be None)
        threshold      : minimum score to include agent
        max_agents     : cap on number of agents returned

    Returns:
        Sorted list of (agent_dict, score) — highest score first.
    """
    if not agents or not query.strip():
        return []

    tokens = _tokenize(query)
    results: List[Tuple[Dict, float]] = []

    for agent in agents:
        if not agent.get("enabled", True):
            continue

        kw_score  = _keyword_score(tokens, agent)
        sem_score = _semantic_score(query, agent, embedding_model)
        combined  = KEYWORD_WEIGHT * kw_score + SEMANTIC_WEIGHT * sem_score

        logger.debug(
            f"[router] agent='{agent['id']}' kw={kw_score:.2f} sem={sem_score:.2f} → {combined:.2f}"
        )

        if combined >= threshold:
            results.append((agent, combined))

    # Sort best first
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_agents]


def explain_routing(query: str, agents: List[Dict], embedding_model=None) -> List[Dict]:
    """
    Debug helper — returns routing explanation for each agent.
    """
    tokens = _tokenize(query)
    out = []
    for agent in agents:
        kw_score  = _keyword_score(tokens, agent)
        sem_score = _semantic_score(query, agent, embedding_model)
        combined  = KEYWORD_WEIGHT * kw_score + SEMANTIC_WEIGHT * sem_score
        out.append({
            "agent_id":    agent["id"],
            "agent_name":  agent["name"],
            "keyword_score":  round(kw_score, 3),
            "semantic_score": round(sem_score, 3),
            "combined_score": round(combined, 3),
            "would_route":    combined >= MATCH_THRESHOLD,
        })
    out.sort(key=lambda x: x["combined_score"], reverse=True)
    return out
