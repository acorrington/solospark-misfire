"""Site-reference scraping for mockup generation.

When a lead already has a website, ``scrape_site_reference`` fetches it once
and extracts everything the mockup pipeline can reuse:

* **Copy basis** — headline, meta description, section headings and cleaned
  body text. This is fed to the LLM so the generated copy stays factually
  aligned with what the business already publishes (same services, same
  service area, same differentiators) instead of being invented from scratch.
* **Logo** — the most likely brand mark (``scanner.find_logo_url``).
* **Images** — an og:image / first content image for the hero background and
  one or two more candidates for the about section.

Defensive by design: any failure (unreachable site, timeout, nothing usable)
degrades to a partial or empty result and never raises — generation must keep
working even when the reference is unavailable. All network access goes
through an injectable session-like object so tests run offline.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .scanner import (
    _fetch,
    _to_https,
    extract_business_text,
    find_logo_url,
)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_IMAGE_BYTES = 1_500_000  # ~1.5 MB per asset — previews stay light
IMAGE_FETCH_TIMEOUT = 10

#: Raster extensions accepted for hero/about images (logo also allows svg).
_IMAGE_EXT_TO_SUFFIX = {
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpg",
    ".webp": "webp",
    ".gif": "gif",
    ".avif": "avif",
}

#: Substrings that mark an <img> as UI chrome rather than a usable photo.
_JUNK_IMAGE_KEYWORDS = (
    "icon",
    "sprite",
    "logo",
    "spacer",
    "pixel",
    "tracking",
    "blank",
    "arrow",
    "button",
    "checkmark",
    "social",
    "favicon",
    "flag",
)

MAX_CONTENT_IMAGES = 6


# ── Small helpers ────────────────────────────────────────────────────────────


def _meta_content(soup: BeautifulSoup, names: list[str]) -> str:
    """First non-empty ``content`` among <meta> tags with one of *names*."""
    for name in names:
        tag = soup.find("meta", attrs={"name": name}) or soup.find(
            "meta", attrs={"property": name}
        )
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    return ""


def _section_headings(soup: BeautifulSoup) -> list[str]:
    """Deduplicated h2/h3 texts — the site's own section/service names."""
    out: list[str] = []
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(" ", strip=True)
        if text and len(text) <= 80 and text not in out:
            out.append(text)
        if len(out) >= 8:
            break
    return out


def _image_suffix(url: str, ctype: str | None = None) -> str | None:
    """Map a URL (or content-type) to a safe file suffix, or None."""
    path = urlparse(url).path.lower()
    for ext, suffix in _IMAGE_EXT_TO_SUFFIX.items():
        if path.endswith(ext):
            return suffix
    if ctype:
        ctype = ctype.split(";")[0].strip().lower()
        mapping = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
            "image/gif": "gif",
            "image/avif": "avif",
        }
        if ctype in mapping:
            return mapping[ctype]
    return None


def _is_content_image(img, base_url: str) -> str | None:
    """Return the absolute URL of *img* if it looks like a usable photo."""
    src = img.get("src") or img.get("data-src") or ""
    if not src or src.startswith(("data:", "javascript:")):
        return None
    url = urljoin(base_url, src.strip())
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    path = parsed.path.lower()
    if not any(path.endswith(ext) for ext in _IMAGE_EXT_TO_SUFFIX):
        return None
    name = path.rsplit("/", 1)[-1]
    if any(keyword in name for keyword in _JUNK_IMAGE_KEYWORDS):
        return None
    # Tiny explicit dimensions ⇒ almost certainly an icon/spacer.
    for attr in ("width", "height"):
        raw = img.get(attr)
        if raw and raw.isdigit() and int(raw) <= 32:
            return None
    return url


def _collect_content_images(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Ordered, de-duplicated candidate photo URLs (max MAX_CONTENT_IMAGES)."""
    out: list[str] = []
    for img in soup.find_all("img"):
        url = _is_content_image(img, base_url)
        if url and url not in out:
            out.append(url)
        if len(out) >= MAX_CONTENT_IMAGES:
            break
    return out


# ── Public API ───────────────────────────────────────────────────────────────


def scrape_site_reference(url: str | None, http_session=None) -> dict:
    """Fetch *url* and extract a structured reference for mockup generation.

    Returns a dict with the keys ``source_url``, ``headline``,
    ``meta_description``, ``service_headings``, ``body_text``, ``logo_url``,
    ``hero_image_url``, ``about_images`` and ``content_images`` (the full
    deduplicated list of on-page photos, capped — missing pieces are empty).
    Returns ``{}`` when the site cannot be fetched at all.
    """
    if not url or not str(url).strip():
        return {}

    session = http_session or requests.Session()
    resp, err, _elapsed = _fetch(session, _to_https(str(url).strip()))
    if resp is None and err == "ssl":
        # Mirror audit_url: retry plain HTTP before giving up.
        from .scanner import _to_http

        resp, err, _elapsed = _fetch(session, _to_http(str(url).strip()))
    if resp is None or getattr(resp, "status_code", 0) >= 400:
        return {}

    if not getattr(resp, "encoding", None):
        resp.encoding = resp.apparent_encoding or "utf-8"
    html_text = resp.text or ""
    if not html_text.strip():
        return {}

    soup = BeautifulSoup(html_text, "html.parser")
    base_url = str(getattr(resp, "url", None) or url)

    content_images = _collect_content_images(soup, base_url)
    hero = (
        _meta_content(soup, ["og:image", "twitter:image"])
        or (content_images[0] if content_images else "")
    )
    if hero:
        hero = urljoin(base_url, hero.strip())

    about_images = [u for u in content_images if u != hero][:2]

    return {
        "source_url": base_url,
        "headline": soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "",
        "meta_description": _meta_content(
            soup, ["description", "og:description"]
        ),
        "service_headings": _section_headings(soup),
        "body_text": extract_business_text(soup),
        "logo_url": find_logo_url(soup, base_url) or "",
        "hero_image_url": hero,
        "about_images": about_images,
        # Full deduplicated photo list (already capped at MAX_CONTENT_IMAGES).
        # Exposed so the pipeline can build a dedicated gallery section from
        # more than the two images reserved for the about block.
        "content_images": content_images,
    }


def fetch_image_bytes(
    url: str | None,
    http_session=None,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> tuple[bytes, str] | None:
    """Download *url* and return ``(data, content_type)``.

    Returns ``None`` (never raises) when the URL is missing, unreachable,
    not an image by content-type/extension, or larger than *max_bytes*.
    """
    if not url:
        return None
    session = http_session or requests.Session()
    try:
        resp = session.get(
            url,
            timeout=IMAGE_FETCH_TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    data = resp.content or b""
    if not data or len(data) > max_bytes:
        return None
    if ctype:
        if not ctype.startswith("image/"):
            return None
    elif _image_suffix(str(resp.url)) is None:
        # No content-type header — trust the URL extension instead.
        return None
    return data, (ctype or "image/png")


def asset_filename(url: str, ctype: str | None = None, prefix: str = "asset") -> str:
    """Build a safe object filename like ``hero.jpg`` for *url*."""
    suffix = _image_suffix(url, ctype) or "png"
    return f"{prefix}.{suffix}"


def reference_copy_block(ref: dict | None) -> str:
    """Render a scraped reference into the text block fed to the LLM.

    Empty string when there is nothing usable (the prompt then falls back to
    its existing behavior).
    """
    if not ref:
        return ""
    parts: list[str] = []
    if ref.get("headline"):
        parts.append(f"Their current headline: {ref['headline']}")
    if ref.get("meta_description"):
        parts.append(f"Their site description: {ref['meta_description']}")
    if ref.get("service_headings"):
        parts.append(
            "Their site's section headings: "
            + "; ".join(ref["service_headings"])
        )
    if ref.get("body_text"):
        parts.append(f"Text from their current website:\n{ref['body_text'][:2500]}")
    return "\n\n".join(parts)
