"""
agents/realtime_search.py

Real-time web search for AIVA agents.

Strategy:
  1. DuckDuckGo (free, no API key)
  2. Fallback → Google Custom Search API  (if GOOGLE_API_KEY + GOOGLE_CSE_ID in .env)
  3. Fallback → Bing Search API           (if BING_API_KEY in .env)

Returns a list of SearchResult dicts:
  {
    "title":   str,
    "url":     str,
    "snippet": str,
    "body":    str,   ← full scraped page text (truncated)
    "source":  "ddg" | "google" | "bing",
  }
"""

import os
import re
import time
import logging
import json
from typing import List, Dict, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

MAX_BODY_CHARS  = 3000   # per result
MAX_RESULTS     = 5
REQUEST_TIMEOUT = 12


# ── HTML text extractor (no BS4 dependency) ───────────────────────────────────

def _extract_text(html: str) -> str:
    """Strip HTML tags and return readable plain text."""
    from html.parser import HTMLParser

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self._buf: List[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "footer", "header", "noscript", "aside"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "footer", "header", "noscript", "aside"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                t = data.strip()
                if t:
                    self._buf.append(t)

    p = _Parser()
    p.feed(html)
    text = "\n".join(p._buf)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_page_body(url: str, session) -> str:
    """Scrape a URL and return plain text body."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (AIVA-RAG/2.0; +https://aivafreelancia.in/AI)"
        }
        r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        r.raise_for_status()
        text = _extract_text(r.text)
        return text[:MAX_BODY_CHARS]
    except Exception as e:
        logger.debug(f"[search] page fetch failed {url}: {e}")
        return ""


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def _search_ddg(query: str, max_results: int, session) -> List[Dict]:
    """
    Uses DuckDuckGo HTML interface (no API key needed).
    Falls back gracefully if blocked.
    """
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        r.raise_for_status()

        # Parse result blocks from DDG HTML
        results = []
        # DDG HTML results: <a class="result__a" href="...">title</a>
        # snippet:          <a class="result__snippet">...</a>
        title_re   = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
        snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)

        titles   = title_re.findall(r.text)
        snippets = snippet_re.findall(r.text)

        for i, (href, title) in enumerate(titles[:max_results]):
            # DDG sometimes wraps URLs — extract the real URL
            real_url = href
            if "//duckduckgo.com/l/" in href or href.startswith("/l/"):
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    from urllib.parse import unquote
                    real_url = unquote(m.group(1))

            snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
            results.append({
                "title":   _strip_tags(title),
                "url":     real_url,
                "snippet": snippet,
                "body":    "",
                "source":  "ddg",
            })

        logger.info(f"[search] DDG returned {len(results)} results for: {query[:60]}")
        return results

    except Exception as e:
        logger.warning(f"[search] DDG failed: {e}")
        return []


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


# ── Google Custom Search ──────────────────────────────────────────────────────

def _search_google(query: str, max_results: int, session) -> List[Dict]:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    cse_id  = os.getenv("GOOGLE_CSE_ID", "")
    if not api_key or not cse_id:
        return []
    try:
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={api_key}&cx={cse_id}"
            f"&q={quote_plus(query)}&num={min(max_results, 10)}"
        )
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "body":    "",
                "source":  "google",
            })
        logger.info(f"[search] Google returned {len(results)} results")
        return results
    except Exception as e:
        logger.warning(f"[search] Google failed: {e}")
        return []


# ── Bing Search ───────────────────────────────────────────────────────────────

def _search_bing(query: str, max_results: int, session) -> List[Dict]:
    api_key = os.getenv("BING_API_KEY", "")
    if not api_key:
        return []
    try:
        url = (
            f"https://api.bing.microsoft.com/v7.0/search"
            f"?q={quote_plus(query)}&count={min(max_results, 10)}"
        )
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        r = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("webPages", {}).get("value", []):
            results.append({
                "title":   item.get("name", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "body":    "",
                "source":  "bing",
            })
        logger.info(f"[search] Bing returned {len(results)} results")
        return results
    except Exception as e:
        logger.warning(f"[search] Bing failed: {e}")
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def realtime_search(
    query: str,
    max_results: int = MAX_RESULTS,
    fetch_bodies: bool = True,
) -> List[Dict]:
    """
    Search the web for `query`.
    1. Try DuckDuckGo.
    2. If DDG returns nothing → try Google.
    3. If Google also fails → try Bing.

    If fetch_bodies=True, scrapes the full page text for the top results.
    Returns list of SearchResult dicts.
    """
    try:
        import requests
        session = requests.Session()
    except ImportError:
        logger.error("[search] requests library not installed")
        return []

    results: List[Dict] = []

    # 1. DDG
    results = _search_ddg(query, max_results, session)

    # 2. Google fallback
    if not results:
        logger.info("[search] DDG empty → trying Google")
        results = _search_google(query, max_results, session)

    # 3. Bing fallback
    if not results:
        logger.info("[search] Google empty → trying Bing")
        results = _search_bing(query, max_results, session)

    if not results:
        logger.warning(f"[search] All search providers failed for: {query}")
        return []

    # Fetch page bodies in parallel for top results
    if fetch_bodies and results:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        top = results[:min(3, len(results))]   # only scrape top 3 to stay fast

        def _fetch(r):
            r["body"] = _fetch_page_body(r["url"], session)
            return r

        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(_fetch, r): i for i, r in enumerate(top)}
            for f in as_completed(futures):
                try:
                    idx = futures[f]
                    results[idx] = f.result()
                except Exception as e:
                    logger.debug(f"[search] body fetch error: {e}")

    return results[:max_results]
