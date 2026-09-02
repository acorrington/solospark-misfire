"""Phase 2 tests — Places discovery, website audit, asset extraction.

All network access goes through a fake session; no internet required.
"""

import datetime
from pathlib import Path

import pytest
import requests
from bs4 import BeautifulSoup

from app.models import Business, DealStage, reinit_db
from app.scanner import (
    FIELD_MASK,
    PLACES_ENDPOINT,
    audit_business,
    audit_url,
    discover_places,
    download_asset,
    extract_business_text,
    find_logo_url,
    ingest_places,
)

NOW_YEAR = datetime.datetime.now().year


# ── Fake HTTP layer ──────────────────────────────────────────────────────────


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

    def _resolve(self, url):
        if url in self.routes:
            item = self.routes[url]
        elif self.default is not None:
            item = self.default
        else:
            item = FakeResponse(404, "<html><body>not found</body></html>")
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, timeout=None, headers=None):
        self.calls.append(("GET", url))
        return self._resolve(url)

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json, headers))
        return self._resolve(url)


# ── Fixtures / helpers ───────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    engine, factory = reinit_db(f"sqlite:///{tmp_path / 'test.db'}")
    session = factory()
    yield session
    session.close()


GOOD_HTML = f"""
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Acme Plumbing</title>
  <script>var junk = "ignore me";</script>
</head>
<body>
  <nav><a href="/services">Services</a><a href="/about">About</a></nav>
  <header><img src="/img/acme-logo.png" alt="Acme Plumbing logo"></header>
  <h1>Acme Plumbing — Eugene, OR</h1>
  <p>Honest plumbing service since 1998. Call (541) 555-0100.</p>
  <p>Email us at owner@acmeplumbing.com for a free quote.</p>
  <div id="cookie-banner">We use cookies. Accept</div>
  <footer>&copy; {NOW_YEAR} Acme Plumbing</footer>
</body>
</html>
"""

BAD_HTML = """
<html>
<head><title>Old Site</title></head>
<body>
  <h1>Welcome</h1>
  <p>Best widgets in town.</p>
  <div class="cookie-consent">Cookie banner text</div>
  <footer>&copy; 2014 Widget Co.</footer>
</body>
</html>
"""

PLACES_PAYLOAD = {
    "places": [
        {
            "id": "ChIJ-1",
            "displayName": {"text": "Acme Plumbing"},
            "formattedAddress": "1 Main St, Eugene, OR 97401",
            "nationalPhoneNumber": "(541) 555-0100",
            "websiteUri": "https://acmeplumbing.com",
            "rating": 4.7,
            "userRatingCount": 123,
            "primaryTypeDisplayName": {"text": "Plumber"},
            "types": ["plumber"],
        },
        {
            "id": "ChIJ-2",
            "displayName": {"text": "No Web Plumbing"},
            "formattedAddress": "2 Main St, Eugene, OR 97401",
            "rating": 3.9,
            "userRatingCount": 40,
            "primaryTypeDisplayName": {"text": "Plumber"},
        },
        {
            "id": "ChIJ-3",
            "displayName": {"text": "Yelp Only Plumbing"},
            "formattedAddress": "3 Main St, Eugene, OR 97401",
            "websiteUri": "https://www.yelp.com/biz/plumbing-eugene",
            "rating": 4.2,
            "userRatingCount": 88,
            "primaryTypeDisplayName": {"text": "Plumber"},
        },
    ]
}


# ── discover_places ──────────────────────────────────────────────────────────


def test_discover_places_posts_correct_request():
    resp = FakeResponse(200, "")
    resp.json = lambda: PLACES_PAYLOAD  # type: ignore[attr-defined]
    session = FakeSession(routes={PLACES_ENDPOINT: resp})

    places = discover_places(api_key="test-key", query="plumbers in Eugene OR",
                             lat=44.05, lng=-123.08, http_session=session)

    assert len(places) == 3
    method, url, payload, headers = session.calls[0]
    assert method == "POST"
    assert url == PLACES_ENDPOINT
    assert headers["X-Goog-Api-Key"] == "test-key"
    assert headers["X-Goog-FieldMask"] == FIELD_MASK
    assert payload["textQuery"] == "plumbers in Eugene OR"
    circle = payload["locationBias"]["circle"]
    assert circle["center"] == {"latitude": 44.05, "longitude": -123.08}
    assert circle["radius"] == 15000


def test_discover_places_requires_api_key(monkeypatch):
    from app.config import reload_settings

    monkeypatch.setenv("PLACES_API_KEY", "")
    reload_settings()
    try:
        with pytest.raises(ValueError, match="PLACES_API_KEY"):
            discover_places(query="plumbers")
    finally:
        reload_settings()


# ── audit_url ────────────────────────────────────────────────────────────────


def test_audit_clean_site_has_no_flags():
    session = FakeSession(routes={"https://acmeplumbing.com": FakeResponse(200, GOOD_HTML)})
    report = audit_url("https://acmeplumbing.com", http_session=session)

    assert report["reachable"] is True
    assert report["status_code"] == 200
    assert report["has_ssl"] is True
    assert report["has_viewport"] is True
    assert report["is_outdated"] is False
    assert report["copyright_year"] == NOW_YEAR
    assert report["flags"] == []
    assert "Acme Plumbing" in report["raw_text"]


def test_audit_flags_missing_viewport_and_stale_copyright():
    session = FakeSession(routes={"https://widgets.com": FakeResponse(200, BAD_HTML)})
    report = audit_url("http://widgets.com", http_session=session)

    assert "Missing Mobile Viewport" in report["flags"]
    assert any(f.startswith("Stale Copyright (") for f in report["flags"])
    assert report["is_outdated"] is True
    # https upgrade was attempted first
    assert ("GET", "https://widgets.com") in session.calls


def test_audit_ssl_error_falls_back_to_http():
    good = FakeResponse(200, GOOD_HTML, url="http://acmeplumbing.com")
    session = FakeSession(routes={
        "https://acmeplumbing.com": requests.exceptions.SSLError("bad cert"),
        "http://acmeplumbing.com": good,
    })
    report = audit_url("https://acmeplumbing.com", http_session=session)

    assert report["has_ssl"] is False
    assert "SSL/Certificate Error" in report["flags"]
    # Content checks still ran over the http fallback.
    assert report["reachable"] is True
    assert report["has_viewport"] is True


def test_audit_timeout_flagged():
    session = FakeSession(routes={
        "https://slow.com": requests.exceptions.Timeout("timed out")
    })
    report = audit_url("https://slow.com", http_session=session)

    assert report["reachable"] is False
    assert "Connection Timeout (>8s)" in report["flags"]


def test_audit_http_404_flagged():
    session = FakeSession(routes={"https://gone.com": FakeResponse(404, "<html>gone</html>")})
    report = audit_url("https://gone.com", http_session=session)

    assert report["reachable"] is False
    assert "HTTP Status 404" in report["flags"]


def test_audit_unreachable_flagged():
    session = FakeSession(routes={
        "https://nowhere.test": requests.exceptions.ConnectionError("dns fail")
    })
    report = audit_url("https://nowhere.test", http_session=session)

    assert report["reachable"] is False
    assert any(f.startswith("Unreachable (") for f in report["flags"])


def test_email_scraped_from_homepage():
    html = '<html><body><p>Reach us at owner@acmeplumbing.com or fake@example.com</p></body></html>'
    session = FakeSession(routes={"https://acmeplumbing.com": FakeResponse(200, html)})
    report = audit_url("https://acmeplumbing.com", http_session=session)

    assert report["scraped_email"] == "owner@acmeplumbing.com"


def test_email_scraped_from_contact_page():
    home = "<html><body><h1>No email here</h1></body></html>"
    contact = "<html><body><p>Call us: service@widgetco.com</p></body></html>"
    session = FakeSession(routes={
        "https://widgets.com": FakeResponse(200, home, url="https://widgets.com"),
        "https://widgets.com/contact": FakeResponse(200, contact, url="https://widgets.com/contact"),
    })
    report = audit_url("https://widgets.com", http_session=session)

    assert report["scraped_email"] == "service@widgetco.com"
    assert ("GET", "https://widgets.com/contact") in session.calls


def test_email_junk_filtered():
    html = (
        '<html><body>'
        '<img src="x@y.png"><p>logo@site.jpg</p>'
        '<p>contact: real@biz.org</p>'
        '</body></html>'
    )
    session = FakeSession(routes={"https://biz.org": FakeResponse(200, html)})
    report = audit_url("https://biz.org", http_session=session)

    assert report["scraped_email"] == "real@biz.org"


# ── Text extraction ──────────────────────────────────────────────────────────


def test_extract_business_text_strips_nav_scripts_cookie():
    soup = BeautifulSoup(GOOD_HTML, "html.parser")
    text = extract_business_text(soup)

    assert "Acme Plumbing — Eugene, OR" in text
    assert "Honest plumbing service since 1998" in text
    assert "ignore me" not in text          # script stripped
    assert "Services About" not in text      # nav stripped
    assert "cookies" not in text.lower()     # cookie banner stripped


# ── Logo extraction / download ───────────────────────────────────────────────


def test_find_logo_url_by_keyword():
    soup = BeautifulSoup(GOOD_HTML, "html.parser")
    assert find_logo_url(soup, "https://acmeplumbing.com") == "https://acmeplumbing.com/img/acme-logo.png"


def test_find_logo_url_falls_back_to_first_header_img():
    html = '<html><body><header><img src="/banner.jpg"></header></body></html>'
    soup = BeautifulSoup(html, "html.parser")
    assert find_logo_url(soup, "https://x.com") == "https://x.com/banner.jpg"


def test_find_logo_url_falls_back_to_favicon():
    html = '<html><head><link rel="icon" href="/favicon.ico"></head><body></body></html>'
    soup = BeautifulSoup(html, "html.parser")
    assert find_logo_url(soup, "https://x.com") == "https://x.com/favicon.ico"


def test_find_logo_url_none_when_absent():
    soup = BeautifulSoup("<html><body><p>hi</p></body></html>", "html.parser")
    assert find_logo_url(soup, "https://x.com") is None


def test_download_asset_saves_image(tmp_path):
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    session = FakeSession(routes={
        "https://acmeplumbing.com/img/acme-logo.png": FakeResponse(
            200, content=png_bytes, headers={"Content-Type": "image/png"}
        )
    })
    dest = download_asset("https://acmeplumbing.com/img/acme-logo.png",
                          tmp_path / "assets" / "acme-plumbing", http_session=session)

    assert dest is not None
    assert dest == Path(tmp_path) / "assets" / "acme-plumbing" / "logo.png"
    assert dest.read_bytes() == png_bytes


def test_download_asset_rejects_non_image(tmp_path):
    session = FakeSession(routes={
        "https://x.com/page": FakeResponse(200, "<html></html>",
                                           headers={"Content-Type": "text/html"})
    })
    assert download_asset("https://x.com/page", tmp_path, http_session=session) is None


# ── Ingestion + audit persistence ────────────────────────────────────────────


def test_ingest_places_dedup_and_no_website(db):
    created = ingest_places(db, PLACES_PAYLOAD["places"])
    assert len(created) == 3

    by_id = {b.place_id: b for b in created}

    # Normal website lead
    acme = by_id["ChIJ-1"]
    assert acme.current_website == "https://acmeplumbing.com"
    assert acme.stage == DealStage.DISCOVERED
    assert acme.is_bad_site is False
    assert acme.rating == 4.7

    # No websiteUri → LEAD_NO_WEBSITE, audited immediately
    noweb = by_id["ChIJ-2"]
    assert noweb.no_website is True
    assert noweb.is_bad_site is True
    assert noweb.stage == DealStage.AUDITED
    assert "No Website" in noweb.audit_flags_list()

    # Blocked directory domain → treated as no usable website
    yelp = by_id["ChIJ-3"]
    assert yelp.no_website is True
    assert any("blocked directory" in f for f in yelp.audit_flags_list())

    # Second ingest of the same places → deduped, nothing new
    again = ingest_places(db, PLACES_PAYLOAD["places"])
    assert again == []
    assert db.query(Business).count() == 3


def test_ingest_places_unique_slugs_for_same_name(db):
    places = [
        {"id": "A", "displayName": {"text": "Joe's Plumbing"}, "websiteUri": "https://a.com"},
        {"id": "B", "displayName": {"text": "Joe's Plumbing"}, "websiteUri": "https://b.com"},
    ]
    created = ingest_places(db, places)
    slugs = sorted(b.slug for b in created)
    assert slugs == ["joe-s-plumbing", "joe-s-plumbing-2"]


def test_audit_business_persists_flags_and_email(db):
    biz = Business(place_id="ChIJ-X", name="Widget Co", slug="widget-co",
                   current_website="https://widgets.com")
    db.add(biz)
    db.commit()

    html = (
        '<html><head></head><body>'
        '<h1>Widgets</h1><p>Great widgets.</p>'
        '<footer>&copy; 2013 Widget Co.</footer>'
        '</body></html>'
    )
    session = FakeSession(routes={"https://widgets.com": FakeResponse(200, html)})
    audited = audit_business(db, biz, http_session=session)

    assert audited.stage == DealStage.AUDITED
    assert audited.is_bad_site is True
    flags = audited.audit_flags_list()
    assert "Missing Mobile Viewport" in flags
    assert any(f.startswith("Stale Copyright (") for f in flags)


def test_audit_business_captures_scraped_email(db):
    biz = Business(place_id="ChIJ-Y", name="Acme Plumbing", slug="acme-plumbing-2",
                   current_website="https://acmeplumbing.com")
    db.add(biz)
    db.commit()

    session = FakeSession(routes={"https://acmeplumbing.com": FakeResponse(200, GOOD_HTML)})
    audited = audit_business(db, biz, http_session=session)

    assert audited.contact_email == "owner@acmeplumbing.com"
    assert audited.is_bad_site is False
    assert audited.audit_flags_list() == []


def test_audit_business_no_website_is_noop(db):
    biz = Business(place_id="ChIJ-Z", name="Ghost Co", slug="ghost-co")
    db.add(biz)
    db.commit()

    session = FakeSession()
    result = audit_business(db, biz, http_session=session)
    assert result is biz
    assert session.calls == []
