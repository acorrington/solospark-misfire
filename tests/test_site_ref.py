"""Tests for app.site_ref — site-reference scraping + asset download (offline).

All network access goes through a fake session; no internet required.
"""

from __future__ import annotations

import requests

from app.site_ref import (
    MAX_IMAGE_BYTES,
    asset_filename,
    fetch_image_bytes,
    reference_copy_block,
    scrape_site_reference,
)


# ── Fake HTTP layer (same pattern as tests/test_scanner.py) ──────────────────


class FakeResponse:
    def __init__(self, status_code=200, html="", url="https://example.com",
                 headers=None, content=None):
        self.status_code = status_code
        self._html = html
        self.url = url
        self.headers = headers or {}
        self.encoding = None
        self._content = content

    @property
    def text(self):
        return self._html

    @property
    def apparent_encoding(self):
        return "utf-8"  # mirrors requests.Response (chardet detection)

    @property
    def content(self):
        if self._content is not None:
            return self._content
        return self._html.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    """Route table: url -> FakeResponse or Exception instance."""

    def __init__(self, routes=None, default=None):
        self.routes = routes or {}
        self.default = default
        self.calls = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append(("GET", url))
        if url in self.routes:
            item = self.routes[url]
        elif self.default is not None:
            item = self.default
        else:
            item = FakeResponse(404, "<html><body>not found</body></html>")
        if isinstance(item, Exception):
            raise item
        return item


# ── scrape_site_reference ────────────────────────────────────────────────────

SITE_HTML = """
<html>
<head>
  <title>Acme Plumbing — Eugene</title>
  <meta name="description" content="24/7 emergency plumbing in Eugene, OR." />
  <meta property="og:image" content="/img/og-plumb.jpg" />
  <link rel="icon" href="/img/favicon.png" />
</head>
<body>
  <header><img src="/img/acme-logo.png" alt="Acme logo" /></header>
  <h1>Eugene's Trusted Plumbers Since 1998</h1>
  <p>We fix what others can't. Licensed, insured, local.</p>
  <h2>Emergency Repairs</h2>
  <p>Fast 24/7 response for burst pipes and leaks.</p>
  <h2>Drain Cleaning</h2>
  <p>Hydro jetting and camera inspection.</p>
  <img src="/img/team-work.jpg" alt="crew at work" />
  <img src="/img/pipes-repair.jpg" alt="pipes" width="1600" height="900" />
  <img src="/img/icon-check.png" alt="" width="16" height="16" />
  <img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" />
</body>
</html>
"""

BASE = "https://acmeplumbing.com"


def test_scrape_blank_url_returns_empty():
    for url in (None, "", "   "):
        assert scrape_site_reference(url) == {}


def test_scrape_unreachable_returns_empty():
    session = FakeSession(routes={BASE: requests.ConnectionError("boom")})
    assert scrape_site_reference(BASE, http_session=session) == {}


def test_scrape_404_returns_empty():
    session = FakeSession(routes={BASE: FakeResponse(404, "nope", url=BASE)})
    assert scrape_site_reference(BASE, http_session=session) == {}


def test_scrape_ssl_error_retries_plain_http():
    session = FakeSession(
        routes={
            "https://acmeplumbing.com": requests.exceptions.SSLError("bad cert"),
            "http://acmeplumbing.com": FakeResponse(html=SITE_HTML, url="http://acmeplumbing.com"),
        }
    )
    ref = scrape_site_reference("https://acmeplumbing.com", http_session=session)
    assert ref  # not empty — HTTP fallback succeeded
    assert ref["source_url"] == "http://acmeplumbing.com"


def test_scrape_extracts_full_reference():
    session = FakeSession(routes={BASE: FakeResponse(html=SITE_HTML, url=BASE)})
    ref = scrape_site_reference(BASE, http_session=session)

    assert ref["source_url"] == BASE
    assert ref["headline"] == "Eugene's Trusted Plumbers Since 1998"
    assert ref["meta_description"] == "24/7 emergency plumbing in Eugene, OR."
    assert "Emergency Repairs" in ref["service_headings"]
    assert "Drain Cleaning" in ref["service_headings"]
    assert "Licensed" in ref["body_text"]
    # Logo: header img whose src/alt matches logo keywords
    assert ref["logo_url"] == f"{BASE}/img/acme-logo.png"
    # Hero: og:image wins over content images and is made absolute
    assert ref["hero_image_url"] == f"{BASE}/img/og-plumb.jpg"
    # About: remaining content photos (junk icon + data-uri excluded)
    assert ref["about_images"] == [
        f"{BASE}/img/team-work.jpg",
        f"{BASE}/img/pipes-repair.jpg",
    ]


def test_scrape_without_og_image_uses_first_content_image():
    html = SITE_HTML.replace(
        '<meta property="og:image" content="/img/og-plumb.jpg" />', ""
    )
    session = FakeSession(routes={BASE: FakeResponse(html=html, url=BASE)})
    ref = scrape_site_reference(BASE, http_session=session)
    assert ref["hero_image_url"] == f"{BASE}/img/team-work.jpg"
    # hero is excluded from about candidates
    assert ref["about_images"] == [f"{BASE}/img/pipes-repair.jpg"]


def test_scrape_empty_body_returns_empty():
    session = FakeSession(routes={BASE: FakeResponse(html="   ", url=BASE)})
    assert scrape_site_reference(BASE, http_session=session) == {}


# ── fetch_image_bytes ────────────────────────────────────────────────────────

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png-data"


def test_fetch_image_bytes_ok():
    session = FakeSession(
        routes={
            f"{BASE}/img/photo.jpg": FakeResponse(
                url=f"{BASE}/img/photo.jpg",
                headers={"Content-Type": "image/jpeg"},
                content=PNG_BYTES,
            )
        }
    )
    result = fetch_image_bytes(f"{BASE}/img/photo.jpg", http_session=session)
    assert result == (PNG_BYTES, "image/jpeg")


def test_fetch_image_bytes_rejects_non_image_content_type():
    session = FakeSession(
        routes={
            f"{BASE}/img/page": FakeResponse(
                url=f"{BASE}/img/page",
                headers={"Content-Type": "text/html"},
                content=b"<html></html>",
            )
        }
    )
    assert fetch_image_bytes(f"{BASE}/img/page", http_session=session) is None


def test_fetch_image_bytes_rejects_oversized():
    big = b"x" * (MAX_IMAGE_BYTES + 1)
    session = FakeSession(
        routes={
            f"{BASE}/img/big.jpg": FakeResponse(
                url=f"{BASE}/img/big.jpg",
                headers={"Content-Type": "image/jpeg"},
                content=big,
            )
        }
    )
    assert fetch_image_bytes(f"{BASE}/img/big.jpg", http_session=session) is None


def test_fetch_image_bytes_none_on_request_error():
    session = FakeSession(routes={f"{BASE}/img/x.png": requests.ConnectionError("boom")})
    assert fetch_image_bytes(f"{BASE}/img/x.png", http_session=session) is None


def test_fetch_image_bytes_missing_url_returns_none():
    assert fetch_image_bytes(None) is None
    assert fetch_image_bytes("") is None


def test_fetch_image_bytes_no_content_type_falls_back_to_extension():
    session = FakeSession(
        routes={
            f"{BASE}/img/photo.jpg": FakeResponse(
                url=f"{BASE}/img/photo.jpg", headers={}, content=PNG_BYTES
            )
        }
    )
    result = fetch_image_bytes(f"{BASE}/img/photo.jpg", http_session=session)
    assert result == (PNG_BYTES, "image/png")  # ctype unknown → default png


def test_fetch_image_bytes_no_content_type_and_no_extension_rejected():
    session = FakeSession(
        routes={
            f"{BASE}/media/123": FakeResponse(
                url=f"{BASE}/media/123", headers={}, content=PNG_BYTES
            )
        }
    )
    assert fetch_image_bytes(f"{BASE}/media/123", http_session=session) is None


# ── asset_filename ───────────────────────────────────────────────────────────


def test_asset_filename_uses_url_extension():
    assert asset_filename("https://x.com/a/b.JPG", prefix="hero") == "hero.jpg"
    assert asset_filename("https://x.com/logo.PNG", prefix="logo") == "logo.png"


def test_asset_filename_falls_back_to_content_type():
    assert asset_filename("https://x.com/media/9", ctype="image/webp", prefix="about") == (
        "about.webp"
    )


def test_asset_filename_unknown_defaults_to_png():
    assert asset_filename("https://x.com/media/9", prefix="asset") == "asset.png"


# ── reference_copy_block ─────────────────────────────────────────────────────


def test_reference_copy_block_empty_when_nothing_usable():
    assert reference_copy_block(None) == ""
    assert reference_copy_block({}) == ""
    assert reference_copy_block({"headline": "", "body_text": ""}) == ""


def test_reference_copy_block_renders_all_sections():
    ref = {
        "headline": "Big Headline",
        "meta_description": "Site description here.",
        "service_headings": ["A", "B"],
        "body_text": "Body copy text.",
    }
    block = reference_copy_block(ref)
    assert "Their current headline: Big Headline" in block
    assert "Their site description: Site description here." in block
    assert "section headings: A; B" in block
    assert "Body copy text." in block


def test_reference_copy_block_truncates_long_body():
    ref = {"body_text": "x" * 3000}
    block = reference_copy_block(ref)
    # 2500-char cap + label overhead, but never the full 3000
    assert len(block) < 3000
    assert "x" * 2500 in block
    assert "x" * 2600 not in block
