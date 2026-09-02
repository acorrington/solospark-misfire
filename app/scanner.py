"""Phase 2 — Local business discovery & automated website auditing.

* ``discover_places`` — Google Places API (New) ``searchText`` with a strict
  field mask (keeps per-call cost minimal).
* ``audit_url`` — fetches a business website and produces a structured audit
  report: SSL/HTTPS, mobile viewport, stale copyright, HTTP status/timeout,
  plus scraped contact email and cleaned business text for the LLM.
* Asset extraction — logo URL discovery + download into ``assets/<slug>/``.
* ``ingest_places`` / ``audit_business`` — persist results to SQLite with
  place_id dedup and domain blocklist enforcement.

All network functions accept an injectable session-like object so tests can run
without touching the internet.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import get_settings
from .models import Business, DealStage
from .utils import domain_of, is_blocked_domain, unique_slug

# ── Google Places (New) ──────────────────────────────────────────────────────

PLACES_ENDPOINT = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.formattedAddress,"
    "places.nationalPhoneNumber,"
    "places.internationalPhoneNumber,"
    "places.websiteUri,"
    "places.rating,"
    "places.userRatingCount,"
    "places.primaryTypeDisplayName,"
    "places.types,"
    "nextPageToken"
)

# ── HTTP helpers ─────────────────────────────────────────────────────────────

_UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

FETCH_TIMEOUT = 8  # seconds — spec: timeout > 8s counts as a failing site


def _to_https(url: str) -> str:
    if url.lower().startswith("http://"):
        return "https://" + url[len("http://"):]
    return url if "://" in url else "https://" + url


def _to_http(url: str) -> str:
    if url.lower().startswith("https://"):
        return "http://" + url[len("https://"):]
    return url if "://" in url else "http://" + url


def _fetch(session, url: str):
    """GET *url*; returns (response | None, error | None, elapsed_ms).

    ``error`` is one of: None, "ssl", "timeout", "unreachable:<ExcName>".
    """
    started = time.monotonic()
    try:
        resp = session.get(url, timeout=FETCH_TIMEOUT, headers=_UA_HEADERS)
        return resp, None, int((time.monotonic() - started) * 1000)
    except requests.exceptions.SSLError:
        return None, "ssl", int((time.monotonic() - started) * 1000)
    except requests.exceptions.Timeout:
        return None, "timeout", int((time.monotonic() - started) * 1000)
    except requests.exceptions.RequestException as exc:
        return None, f"unreachable:{type(exc).__name__}", int(
            (time.monotonic() - started) * 1000
        )


def _dedup(flags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            out.append(flag)
    return out


# ── Audit checks ─────────────────────────────────────────────────────────────

COPYRIGHT_RE = re.compile(
    r"(?:©|&copy;|&#169;|copyright)\s*\(?\s*(20\d{2})(?:\s*[-–—]\s*(20\d{2}))?",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

_IMAGE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".avif",
)


def _is_plausible_email(addr: str) -> bool:
    addr = addr.strip().rstrip(".,;")
    if not addr or "@" not in addr:
        return False
    domain = addr.rsplit("@", 1)[1].lower()
    if any(domain.endswith(sfx) for sfx in _IMAGE_SUFFIXES):
        return False
    if re.search(r"(example\.|domain\.|your[-_]?email|test@|noreply|no-reply)", addr, re.IGNORECASE):
        return False
    return True


def _scrape_emails(html: str) -> list[str]:
    found: list[str] = []
    for match in EMAIL_RE.findall(html or ""):
        if _is_plausible_email(match) and match not in found:
            found.append(match)
    return found


def extract_business_text(soup: BeautifulSoup, max_chars: int = 3000) -> str:
    """Aggregate h1–h3 + paragraph text, stripping nav/scripts/cookie banners."""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "form"]):
        tag.decompose()
    for attr in ("class", "id"):
        for el in soup.find_all(attrs={attr: re.compile(r"cookie|consent|gdpr|onetrust", re.IGNORECASE)}):
            el.decompose()
    parts: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        text = tag.get_text(" ", strip=True)
        if len(text) > 2:
            parts.append(text)
    return " ".join(parts)[:max_chars]


def _latest_copyright_year(html: str) -> int | None:
    best: int | None = None
    for match in COPYRIGHT_RE.finditer(html or ""):
        year = int(match.group(1))
        if match.group(2):
            year = max(year, int(match.group(2)))
        if best is None or year > best:
            best = year
    return best


def audit_url(url: str, http_session: requests.Session | None = None) -> dict:
    """Audit a business website and return a structured report.

    Report keys: url, reachable, status_code, elapsed_ms, has_ssl,
    has_viewport, is_outdated, copyright_year, scraped_email, raw_text, flags.
    """
    session = http_session or requests.Session()
    flags: list[str] = []
    report: dict = {
        "url": url,
        "reachable": False,
        "status_code": None,
        "elapsed_ms": 0,
        "has_ssl": True,
        "has_viewport": False,
        "is_outdated": False,
        "copyright_year": None,
        "scraped_email": None,
        "raw_text": "",
        "flags": [],
    }

    # 1) Fetch — prefer HTTPS; on certificate failure fall back to plain HTTP
    #    so the remaining content checks still run.
    resp, err, elapsed = _fetch(session, _to_https(url))
    if err == "ssl":
        report["has_ssl"] = False
        flags.append("SSL/Certificate Error")
        resp, err, elapsed = _fetch(session, _to_http(url))
    if resp is None:
        if err == "timeout":
            flags.append("Connection Timeout (>8s)")
        elif err and err.startswith("unreachable:"):
            flags.append(f"Unreachable ({err.split(':', 1)[1]})")
        report["flags"] = _dedup(flags)
        return report

    report["status_code"] = resp.status_code
    report["elapsed_ms"] = elapsed
    if resp.status_code >= 400:
        flags.append(f"HTTP Status {resp.status_code}")
        report["flags"] = _dedup(flags)
        return report

    report["reachable"] = True
    try:
        if not resp.encoding or resp.encoding.lower() in ("iso-8859-1",):
            resp.encoding = resp.apparent_encoding
    except Exception:  # chardet unavailable — fall back to declared/default
        pass
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    final_url = str(resp.url) or url

    # 2) Mobile viewport
    if not soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.IGNORECASE)}):
        flags.append("Missing Mobile Viewport")
    else:
        report["has_viewport"] = True

    # 3) Stale copyright (≥4 years old counts as outdated)
    year = _latest_copyright_year(html)
    report["copyright_year"] = year
    if year is not None and year <= time.localtime().tm_year - 4:
        report["is_outdated"] = True
        flags.append(f"Stale Copyright ({year})")

    # 4) Contact email — homepage first, then /contact
    emails = _scrape_emails(html)
    if not emails:
        parsed = urlparse(final_url)
        contact_url = f"{parsed.scheme or 'https'}://{parsed.netloc}/contact"
        c_resp, c_err, _ = _fetch(session, contact_url)
        if c_resp is not None and c_resp.status_code < 400:
            emails = _scrape_emails(c_resp.text)
    if emails:
        report["scraped_email"] = emails[0]

    # 5) Cleaned business text for the LLM prompt
    report["raw_text"] = extract_business_text(soup)

    report["flags"] = _dedup(flags)
    return report


# ── Asset extraction (logo) ──────────────────────────────────────────────────

LOGO_KEYWORDS = ("logo", "brand", "emblem", "favicon")

_CONTENT_TYPE_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}


def find_logo_url(soup: BeautifulSoup, base_url: str) -> str | None:
    """Find the most likely logo image URL on a page.

    Preference order: img in <header>/<nav> whose src/alt matches logo
    keywords → first img in <header>/<nav> → favicon / apple-touch-icon link.
    """
    for container in soup.find_all(["header", "nav"]):
        for img in container.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            alt = (img.get("alt") or "").lower()
            if not src:
                continue
            if any(k in src.lower() or k in alt for k in LOGO_KEYWORDS):
                return urljoin(base_url, src)
    for container in soup.find_all(["header", "nav"]):
        img = container.find("img")
        if img:
            src = img.get("src") or img.get("data-src")
            if src:
                return urljoin(base_url, src)
    for rel in ("apple-touch-icon", "icon"):
        link = soup.find("link", rel=rel)
        if link and link.get("href"):
            return urljoin(base_url, link["href"])
    return None


def _ext_from_url_or_ctype(url: str, ctype: str) -> str:
    if ctype in _CONTENT_TYPE_EXT:
        return _CONTENT_TYPE_EXT[ctype]
    path = urlparse(url).path.lower()
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".png"


def download_asset(
    url: str,
    dest_dir: Path | str,
    http_session: requests.Session | None = None,
) -> Path | None:
    """Download an image asset into *dest_dir*; returns saved Path or None."""
    session = http_session or requests.Session()
    try:
        resp = session.get(url, timeout=10, headers=_UA_HEADERS)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype and not ctype.startswith("image/"):
            return None
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"logo{_ext_from_url_or_ctype(url, ctype)}"
        path.write_bytes(resp.content)
        return path
    except requests.RequestException:
        return None


# ── Discovery + persistence ──────────────────────────────────────────────────


def discover_places(
    api_key: str | None = None,
    query: str = "plumbers in Eugene OR",
    lat: float = 44.0521,
    lng: float = -123.0868,
    radius: float = 15000,
    http_session: requests.Session | None = None,
) -> list[dict]:
    """Search Google Places (New) and return the raw place dicts."""
    key = api_key or get_settings().places_api_key
    if not key:
        raise ValueError("PLACES_API_KEY is not set")
    session = http_session or requests.Session()
    payload = {
        "textQuery": query,
        "locationBias": {
            "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": radius}
        },
    }
    resp = session.post(
        PLACES_ENDPOINT,
        json=payload,
        headers={
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": FIELD_MASK,
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("places", [])


def ingest_places(db, places: list[dict]) -> list[Business]:
    """Create Business rows for new place records.

    Dedup ledger: any ``place_id`` already in the database is skipped.
    Businesses without a website (or with only a blocked directory listing)
    are stored as LEAD_NO_WEBSITE and marked audited immediately.
    """
    existing = {b.place_id: b for b in db.query(Business).all()}
    taken_slugs = {b.slug for b in existing.values()}
    created: list[Business] = []

    for place in places:
        place_id = place.get("id")
        if not place_id or place_id in existing:
            continue
        name = (place.get("displayName") or {}).get("text", "Unknown Business")
        website = place.get("websiteUri")
        domain = domain_of(website)
        blocked = is_blocked_domain(domain)

        biz = Business(
            place_id=place_id,
            name=name,
            slug=unique_slug(name, taken_slugs),
            category=(place.get("primaryTypeDisplayName") or {}).get("text"),
            address=place.get("formattedAddress"),
            phone=place.get("nationalPhoneNumber")
            or place.get("internationalPhoneNumber"),
            rating=place.get("rating", 0.0) or 0.0,
            review_count=place.get("userRatingCount", 0) or 0,
        )
        if website and not blocked:
            biz.current_website = website
        else:
            biz.no_website = True
            biz.is_bad_site = True
            biz.stage = DealStage.AUDITED
            flags = ["No Website"]
            if website and blocked:
                flags.append(f"Website on blocked directory ({domain})")
            biz.set_audit_flags(flags)

        taken_slugs.add(biz.slug)
        db.add(biz)
        created.append(biz)

    db.commit()
    for biz in created:
        db.refresh(biz)
    return created


def audit_business(
    db,
    business: Business,
    http_session: requests.Session | None = None,
) -> Business:
    """Run the website audit for a lead and persist flags/email/stage."""
    if not business.current_website:
        return business  # LEAD_NO_WEBSITE — nothing to fetch

    report = audit_url(business.current_website, http_session=http_session)
    business.set_audit_flags(report["flags"])
    business.is_bad_site = bool(report["flags"])
    if report.get("scraped_email") and not business.contact_email:
        business.contact_email = report["scraped_email"]
    business.stage = DealStage.AUDITED
    db.commit()
    db.refresh(business)
    return business
