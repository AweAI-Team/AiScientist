import re
from typing import Any, Iterable


def build_blocked_patterns_from_blacklist(blacklist: list[str]) -> dict[str, list[str]]:
    """
    Build blocked search patterns from a blacklist of URLs.
    
    Args:
        blacklist: List of URLs to block (e.g., ["https://github.com/user/repo"])
        
    Returns:
        dict suitable for blocked_search_patterns constraint
    """
    if not blacklist:
        return {}
    
    url_patterns = []
    for url in blacklist:
        # Strip whitespace and skip empty lines or comments
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        
        # Skip "none" marker
        if url.lower() == "none":
            continue
        
        # Convert URL to regex pattern
        # Escape special regex characters except for wildcards
        escaped_url = re.escape(url)
        # Replace escaped wildcards with regex equivalents
        escaped_url = escaped_url.replace(r'\*', '.*')
        # Make it match anywhere in the URL
        pattern = rf'.*{escaped_url}.*'
        url_patterns.append(pattern)
    
    if not url_patterns:
        return {}
    
    return {"url": url_patterns}


def is_url_blocked(url: str, blocked_patterns: dict[str, list[str]]) -> bool:
    """
    Check if a URL matches any blocked pattern.

    Args:
        url: The URL to check
        blocked_patterns: dict with 'url' key containing list of regex patterns

    Returns:
        True if URL is blocked, False otherwise
    """
    if not url or not blocked_patterns:
        return False

    url_patterns = blocked_patterns.get("url", [])
    for pattern in url_patterns:
        try:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def iter_url_like_values(value: Any) -> Iterable[str]:
    """
    Yield URL-like string values from nested data structures.

    This walks dicts/lists and returns string fields that either:
    - have key names containing 'url', or
    - start with http/https.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = key.lower()
            if isinstance(item, str):
                if "url" in key_lower or item.startswith("http"):
                    yield item
            else:
                yield from iter_url_like_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_url_like_values(item)