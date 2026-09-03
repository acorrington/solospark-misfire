"""Tests for app.stock_images — Openverse lookup and category query mapping.

All HTTP is faked via an injected session; nothing in this file touches the
network.
"""

from __future__ import annotations

import requests

from app.stock_images import (
    OPENVERSE_URL,
    SEARCH_TIMEOUT,
    image_query_for,
    search_stock_images,
)


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, payload=None, fail_status=False):
        self._payload = payload if payload is not None else {}
        self._fail_status = fail_status

    def raise_for_status(self):
        if self._fail_status:
            raise RuntimeError("HTTP 502")

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for requests.Session; records calls and returns canned data."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return self.response


# ── search_stock_images ──────────────────────────────────────────────────────


def test_search_returns_commercial_licensed_urls():
    session = FakeSession(
        FakeResponse(
            {
                "results": [
                    {"url": "https://live.staticflickr.com/a.jpg", "license": "by"},
                    {"url": "https://upload.wikimedia.org/b.png", "license": "BY-SA"},
                    {"url": None, "license": "by"},  # malformed → skipped
                    {"url": "ftp://nope/c.jpg", "license": "by"},  # non-http → skipped
                ]
            }
        )
    )

    results = search_stock_images("plumber", count=5, session=session)
    assert [r["url"] for r in results] == [
        "https://live.staticflickr.com/a.jpg",
        "https://upload.wikimedia.org/b.png",
    ]
    assert results[0]["license"] == "by"

    # the request went to Openverse with a commercial-only license filter
    call = session.calls[0]
    assert call["url"] == OPENVERSE_URL
    assert call["params"]["q"] == "plumber"
    assert call["params"]["license_type"] == "commercial"
    assert call["timeout"] == SEARCH_TIMEOUT


def test_search_respects_count_cap():
    payload = {
        "results": [
            {"url": f"https://x/{i}.jpg", "license": "by"} for i in range(10)
        ]
    }
    results = search_stock_images("food", count=3, session=FakeSession(FakeResponse(payload)))
    assert len(results) == 3


def test_search_page_size_clamped_to_25():
    session = FakeSession(FakeResponse({"results": []}))
    search_stock_images("food", count=100, session=session)
    assert session.calls[0]["params"]["page_size"] == 25


def test_search_zero_count_short_circuits_without_http():
    session = FakeSession(FakeResponse({"results": []}))
    assert search_stock_images("food", count=0, session=session) == []
    assert session.calls == []


def test_search_http_error_returns_empty():
    session = FakeSession(FakeResponse(fail_status=True))
    assert search_stock_images("food", session=session) == []


def test_search_network_error_returns_empty():
    session = FakeSession(error=requests.ConnectionError("boom"))
    assert search_stock_images("food", session=session) == []


def test_search_malformed_payload_returns_empty():
    # payload without a "results" list → nothing usable, no exception
    assert search_stock_images("food", session=FakeSession(FakeResponse({"foo": 1}))) == []
    assert search_stock_images("food", session=FakeSession(FakeResponse([1, 2]))) == []


# ── image_query_for ──────────────────────────────────────────────────────────


def test_image_query_for_exact_match_is_case_insensitive():
    assert image_query_for("Plumber") == "plumber working on pipes and tools"
    assert image_query_for("RESTAURANT") == "cozy restaurant interior with warm lighting"


def test_image_query_for_leading_word_match():
    # "Plumbing & Drain Services" → the "plumbing" entry
    assert image_query_for("Plumbing & Drain Services") == "plumber working on pipes and tools"


def test_image_query_for_unknown_category_uses_raw_text():
    assert image_query_for("Fence Installer") == "fence installer local business"


def test_image_query_for_empty_category_is_generic():
    assert image_query_for("") == "local small business storefront"
    assert image_query_for(None) == "local small business storefront"
