"""Small shared helpers (slugs, domain safety)."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

# Domains that must never be scraped or emailed (directories, franchises,
# institutions) — per requirements.md §7.2.
DOMAIN_BLOCKLIST = {
    "yelp.com",
    "yellowpages.com",
    "ybnc.com",
    "bbb.org",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "nextdoor.com",
    "tripadvisor.com",
    "foursquare.com",
    "g.page",
    "goo.gl",
}

_BLOCKED_TLDS = (".gov", ".edu")


def slugify(text: str) -> str:
    """Turn arbitrary text into a URL-safe slug ('Acme Plumbing Co.' → 'acme-plumbing-co')."""
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return text or "business"


def unique_slug(name: str, taken: set[str] | None = None) -> str:
    """Slugify *name*, appending -2, -3, ... until it is not in *taken*."""
    base = slugify(name)
    if not taken or base not in taken:
        return base
    i = 2
    while f"{base}-{i}" in taken:
        i += 1
    return f"{base}-{i}"


def domain_of(url: str | None) -> str | None:
    """Return the lowercased hostname of a URL, or None."""
    if not url:
        return None
    if "://" not in url:
        url = "https://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return None


def is_blocked_domain(domain: str | None) -> bool:
    """True for government/education domains and known directory/franchise sites."""
    if not domain:
        return False
    if domain.endswith(_BLOCKED_TLDS):
        return True
    for blocked in DOMAIN_BLOCKLIST:
        if domain == blocked or domain.endswith("." + blocked):
            return True
    return False


def city_from_address(address: str | None) -> str:
    """Best-effort city extraction from a 'Street, City, ST ZIP' address."""
    if not address:
        return ""
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 2:
        return parts[-2]
    return ""
