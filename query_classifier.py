"""
agents/query_classifier.py

Classifies an incoming query to determine:
  1. Does it need real-time web data at all?
  2. If yes — what search query should be sent?
  3. Does it need fact-verification (e.g. checking student notes)?

Classification is rule-based + keyword-driven (fast, no extra LLM call needed).
Can be extended with an LLM call for edge cases.

Output:
  {
    "needs_realtime":    bool,
    "reason":            str,          ← why real-time was triggered
    "search_query":      str,          ← refined query to send to search
    "needs_verification": bool,        ← True for "check my notes / is this correct"
    "time_constraint":   str | None,   ← e.g. "1 hour" if detected
    "domain_hint":       str | None,   ← "cooking" | "education" | "finance" | ...
  }
"""

import re
from typing import Dict, Optional, List


# ── Signal patterns ────────────────────────────────────────────────────────────

REALTIME_TRIGGERS = [
    # Time pressure
    r"\b(quick|fast|faster|quickly|rapid|speedy|shortcuts?)\b",
    r"\b(in\s+\d+\s*(min(ute)?s?|hour?s?|hr?s?))\b",
    r"\bwithin\s+\d+",
    r"\b(under|less than)\s+\d+\s*(min|hour|hr)",
    r"\btoday\b", r"\bright now\b", r"\bcurrently\b", r"\blatest\b",
    r"\brecent(ly)?\b", r"\bthis (week|month|year)\b",
    # Verification
    r"\b(verify|verif(y|ied)|check|confirm|correct|mistake|error|wrong|accurate|right)\b",
    r"\b(is this (right|correct|accurate))\b",
    r"\b(my notes?|class notes?|today'?s? class)\b",
    # Knowledge gaps
    r"\b(alternative|another way|different method|easier way|simpler)\b",
    r"\b(don'?t know|not sure|help me understand)\b",
    # Business/market/finance real-time
    r"\b(price|market|stock|rate|news|update|current)\b",
    r"\b(how much|what does .* cost)\b",
]

VERIFICATION_TRIGGERS = [
    r"\b(verify|verif(y|ied)|check|confirm)\b",
    r"\b(correct|mistake|error|wrong|accurate)\b",
    r"\b(is this right|is this correct|is this accurate)\b",
    r"\bmy notes?\b",
    r"\bclass notes?\b",
    r"\btoday'?s? (class|lecture|notes?)\b",
    r"\b(should it be|isn'?t it|are (they|these) wrong)\b",
]

TIME_PATTERN   = re.compile(r"\b(\d+)\s*(min(ute)?s?|hour?s?|hr?s?)\b", re.I)
DOMAIN_SIGNALS = {
    "cooking":   r"\b(cook|recipe|dish|food|ingredient|meal|bake|fry|boil|roast|grill)\b",
    "education": r"\b(notes?|class|chapter|subject|exam|study|chemistry|physics|math|biology|history)\b",
    "finance":   r"\b(stock|market|price|invest|fund|budget|finance|bank|crypto|forex|trade)\b",
    "health":    r"\b(health|medical|symptom|disease|medicine|doctor|treatment|diet)\b",
    "tech":      r"\b(code|software|api|programming|debug|error|install|deploy|cloud)\b",
    "travel":    r"\b(travel|flight|hotel|visa|destination|trip|tour|book)\b",
}


def _matches(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def _detect_domain(text: str) -> Optional[str]:
    t = text.lower()
    for domain, pattern in DOMAIN_SIGNALS.items():
        if re.search(pattern, t):
            return domain
    return None


def _extract_time_constraint(text: str) -> Optional[str]:
    m = TIME_PATTERN.search(text)
    if m:
        return m.group(0)
    return None


def _build_search_query(
    user_query: str,
    domain: Optional[str],
    time_constraint: Optional[str],
    needs_verification: bool,
) -> str:
    """
    Construct an optimised web search query from the user's raw question.
    """
    q = user_query.strip()

    # Append domain context if not already present
    if domain == "cooking" and time_constraint:
        # e.g. "how to make paneer kebab quick method under 30 minutes"
        if "quick" not in q.lower() and "fast" not in q.lower():
            q = q + f" quick method under {time_constraint}"

    elif domain == "education" and needs_verification:
        # e.g. "NCERT class 11 chemistry reaction equations accurate"
        if "ncert" not in q.lower():
            q = "NCERT " + q + " correct formula"

    elif domain == "finance":
        q = q + " current 2024"

    return q


def classify_query(user_query: str, has_uploaded_docs: bool = True) -> Dict:
    """
    Classify a user query and return routing metadata.

    Args:
        user_query:        The raw user message.
        has_uploaded_docs: Whether the user has documents uploaded in the RAG system.

    Returns dict with keys:
        needs_realtime, reason, search_query,
        needs_verification, time_constraint, domain_hint
    """
    text = user_query.strip()

    needs_rt     = _matches(text, REALTIME_TRIGGERS)
    needs_verify = _matches(text, VERIFICATION_TRIGGERS)
    time_c       = _extract_time_constraint(text)
    domain       = _detect_domain(text)

    # Time constraints always mean real-time needed
    if time_c:
        needs_rt = True

    # Verification of uploaded docs always needs real-time cross-check
    if needs_verify and has_uploaded_docs:
        needs_rt = True

    reason = ""
    if time_c:
        reason = f"Time constraint detected: {time_c}"
    elif needs_verify:
        reason = "Verification request — cross-checking against live sources"
    elif needs_rt:
        reason = "Real-time signal detected in query"

    search_q = _build_search_query(text, domain, time_c, needs_verify)

    return {
        "needs_realtime":     needs_rt,
        "reason":             reason,
        "search_query":       search_q,
        "needs_verification": needs_verify,
        "time_constraint":    time_c,
        "domain_hint":        domain,
    }
