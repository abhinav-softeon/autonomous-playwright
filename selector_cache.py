"""
Selector cache — persists working and failed selectors per domain across runs.

Structure of selector_cache.json:
{
  "www.youtube.com": {
    "working": {
      "search_input": "input[name='search_query']",
      "search_button": "button[aria-label='Search']"
    },
    "failed": ["input[name='q']", "#search", "ytd-searchbox #search-input"]
  },
  "www.google.com": { ... }
}
"""

import json
import os
from urllib.parse import urlparse

_CACHE_FILE = os.path.join(os.path.dirname(__file__), "selector_cache.json")


def _load() -> dict:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_failed_selectors(url: str) -> list[str]:
    """Return all selectors known to fail on this domain."""
    return _load().get(_domain(url), {}).get("failed", [])


def save_failed_selector(url: str, selector: str) -> None:
    """Record a selector that failed on this domain (won't be suggested again)."""
    cache = _load()
    d = _domain(url)
    if d not in cache:
        cache[d] = {"working": {}, "failed": []}
    if selector not in cache[d].get("failed", []):
        cache[d].setdefault("failed", []).append(selector)
        _save(cache)


def save_working_selector(url: str, purpose: str, selector: str) -> None:
    """
    Record a selector that successfully worked on this domain.

    Args:
        url: Page URL (domain is extracted automatically)
        purpose: Short description of what the selector targets,
                 e.g. "search_input", "submit_button"
        selector: The CSS selector that worked
    """
    cache = _load()
    d = _domain(url)
    if d not in cache:
        cache[d] = {"working": {}, "failed": []}
    cache[d].setdefault("working", {})[purpose] = selector
    # Remove from failed list if it was there before
    failed = cache[d].get("failed", [])
    if selector in failed:
        failed.remove(selector)
    _save(cache)


def get_domain_hints(url: str) -> str:
    """
    Return a formatted string of known working and failed selectors for this domain.
    Injected into the specialist prompt so the agent doesn't repeat past mistakes.
    """
    cache = _load()
    d = _domain(url)
    entry = cache.get(d, {})
    working = entry.get("working", {})
    failed = entry.get("failed", [])

    if not working and not failed:
        return ""

    lines = [f"SELECTOR MEMORY FOR {d}:"]
    if working:
        lines.append("  Previously worked:")
        for purpose, sel in working.items():
            lines.append(f"    {sel}  ({purpose})")
    if failed:
        lines.append("  Previously FAILED — do NOT use these:")
        for sel in failed:
            lines.append(f"    {sel}")

    return "\n".join(lines)
