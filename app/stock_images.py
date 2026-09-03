"""Royalty-free stock photo lookup for generated sites.

Uses the Openverse API (https://api.openverse.org), which aggregates openly
licensed images from Flickr, Wikimedia Commons and similar sources. We only
ask for commercially-usable licenses (``license_type=commercial`` → CC BY /
CC BY-SA / public domain), so photos can be placed on real customer sites
without a subscription or attribution plumbing.

No API key is required for low-volume anonymous requests; every failure mode
(network error, HTTP error, unexpected payload) degrades to an empty list so
a stock-photo lookup can never block site generation.
"""

from __future__ import annotations

import requests

OPENVERSE_URL = "https://api.openverse.org/v1/images/"
SEARCH_TIMEOUT = 10  # seconds — a slow search is not worth blocking on

#: Category → Openverse query. Kept intentionally generic so the results read
#: as lifestyle/brand photography rather than stock-photo clichés. Categories
#: not listed here fall back to the raw category text.
_CATEGORY_QUERIES: dict[str, str] = {
    "plumbing": "plumber working on pipes and tools",
    "plumber": "plumber working on pipes and tools",
    "rooter": "drain cleaning service van",
    "sewer": "drain cleaning service van",
    "drain": "drain cleaning service van",
    "restaurant": "cozy restaurant interior with warm lighting",
    "bistro": "elegant bistro dining room table setting",
    "grill": "grilled food on outdoor barbecue grill flames",
    "bar": "craft cocktails being served in a bar",
    "brewery": "craft beer pouring into glass taps",
    "bar & grill": "grilled food and drinks at a bar",
    "pub": "cozy pub interior with wooden tables",
    "food": "freshly prepared gourmet food on a plate",
    "bowl": "colorful fresh salad bowl with vegetables",
    "kitchen": "chef plating food in a professional kitchen",
    "catering": "catering buffet table with dishes",
    "bakery": "freshly baked bread and pastries on display",
    "cafe": "specialty coffee latte art on a cafe counter",
    "coffee": "barista pouring espresso into a cup",
    "pizza": "wood fired pizza fresh from the oven",
    "taco": "tacos with fresh ingredients on a table",
    "burger": "gourmet cheeseburger with fries",
    "sushi": "sushi rolls arranged on a wooden board",
    "chinese": "chinese food stir fry in wok",
    "mexican": "mexican food tacos and guacamole",
    "italian": "italian pasta dish with fresh basil",
    "indian": "indian curry dishes with naan bread",
    "salad": "fresh healthy salad bowl with vegetables",
    "breakfast": "breakfast pancakes with berries on a plate",
    "diner": "classic diner interior with booth seating",
}


def image_query_for(category: str) -> str:
    """Map a business category to an Openverse search query.

    Falls back to the raw (trimmed) category text, or a generic local-business
    query when the category is empty.
    """
    key = (category or "").strip().lower()
    if not key:
        return "local small business storefront"
    if key in _CATEGORY_QUERIES:
        return _CATEGORY_QUERIES[key]
    # Try a leading-word match ("Plumbing & Drain Services" → plumbing).
    for known, query in _CATEGORY_QUERIES.items():
        if key.startswith(known):
            return query
    return f"{key} local business"


def search_stock_images(
    query: str, count: int = 5, session: requests.Session | None = None
) -> list[dict]:
    """Search Openverse for commercially-usable images matching *query*.

    Returns a list of ``{"url": <image url>, "license": <short name>}`` dicts,
    at most *count* entries. Any failure (network, HTTP, payload shape) yields
    an empty list — callers treat that as "no stock photos available".
    """
    if count <= 0:
        return []
    http = session or requests
    try:
        resp = http.get(
            OPENVERSE_URL,
            params={
                "q": query,
                "page_size": min(count, 25),
                "license_type": "commercial",
            },
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:  # noqa: BLE001 — stock lookup is strictly best-effort
        return []

    results: list[dict] = []
    for item in payload.get("results", []) if isinstance(payload, dict) else []:
        url = item.get("url") if isinstance(item, dict) else None
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        license_name = (item.get("license") or "").lower()
        results.append({"url": url, "license": license_name})
        if len(results) >= count:
            break
    return results
