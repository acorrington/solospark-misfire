"""Phase 7 tests — post-payment multi-page prompt builder.

Covers expansion (LLM call + validation/clamping), compilation to five
theme-aware pages with shared navigation, and R2 upload of every page.
All LLM/S3 access goes through injected fakes.
"""

from __future__ import annotations

import json

import pytest

from app import builder
from app.builder import (
    ALLOWED_SUBPAGES,
    DEFAULT_THEME,
    BuilderError,
    deploy_expanded_site,
    expand_site_with_prompt,
    render_site_pages,
)
from app.config import get_settings

BUSINESS = {
    "name": "Acme Plumbing",
    "slug": "acme-plumbing",
    "category": "plumber",
    "city": "Eugene",
    "phone": "(541) 555-0142",
    "address": "42 River St, Eugene, OR",
}

COPY = {
    "tagline": "Proudly local",
    "hero_headline": "Reliable plumbing for Eugene homes",
    "hero_subheadline": "Fast, friendly service when you need it.",
    "services": [
        {"title": "Repairs", "description": "Fix it right the first time.", "icon_name": "wrench"},
        {"title": "Installations", "description": "New systems done properly.", "icon_name": "shield"},
        {"title": "Emergency Service", "description": "24/7 response.", "icon_name": "clock"},
    ],
    "about_heading": "About Acme Plumbing",
    "about_text": "We have served Eugene for decades.\n\nFamily owned and operated.",
    "why_choose_us": ["Licensed & Insured", "Fast Response", "Local Expertise"],
    "cta_text": "Call Now",
}

SITE_PAYLOAD = json.dumps(
    {
        "theme": {"primary": "#b45309", "secondary": "#7c2d12"},
        "pages": {
            "about.html": {
                "heading": "Our Story",
                "paragraphs": ["Founded in 1998, we have kept Eugene's pipes flowing."],
            },
            "services.html": {
                "heading": "What We Offer",
                "items": [{"title": "Drain Cleaning", "text": "We clear clogs fast."}],
            },
            "gallery.html": {
                "heading": "Our Work",
                "captions": ["Water heater swap", "Bathroom repipe"],
            },
            "contact.html": {
                "heading": "Talk to Us",
                "paragraphs": ["Call or send a message — we answer fast."],
            },
        },
    }
)


# ── fakes (mirror tests/test_main.py) ────────────────────────────────────────


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeCompletions:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self.payloads, "FakeLLM ran out of scripted responses"
        return _Completion(self.payloads.pop(0))


class FakeLLMClient:
    def __init__(self, *payloads):
        self.chat = type("Chat", (), {"completions": FakeCompletions(*payloads)})()


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "R2_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()


# ── expansion ─────────────────────────────────────────────────────────────────


def test_expand_happy_path_wrapper_input():
    llm = FakeLLMClient(SITE_PAYLOAD)
    result = expand_site_with_prompt({"business": BUSINESS, "copy": COPY}, "", client=llm)

    assert result["business"] == BUSINESS
    assert result["theme"] == {"primary": "#b45309", "secondary": "#7c2d12"}
    assert set(result["pages"]) == set(ALLOWED_SUBPAGES)
    assert result["pages"]["about.html"]["heading"] == "Our Story"
    assert [i["label"] for i in result["nav"]] == ["Home", "About", "Services", "Gallery", "Contact"]
    assert result["nav"][0]["href"] == "index.html"
    assert len(llm.chat.completions.calls) == 1


def test_expand_accepts_bare_copy_dict():
    llm = FakeLLMClient(SITE_PAYLOAD)
    bare = {**COPY, **BUSINESS}
    result = expand_site_with_prompt(bare, "", client=llm)
    assert result["business"]["name"] == "Acme Plumbing"
    assert result["copy"]["hero_headline"] == COPY["hero_headline"]


def test_expand_invalid_json_raises_builder_error():
    llm = FakeLLMClient("not json at all", "still not json")
    with pytest.raises(BuilderError, match="valid site schema"):
        expand_site_with_prompt({"business": BUSINESS, "copy": COPY}, "", client=llm)
    # initial attempt + one repair retry, and the retry carries the JSON reminder
    assert len(llm.chat.completions.calls) == 2
    last_user = llm.chat.completions.calls[1]["messages"][-1]["content"]
    assert "ONLY a valid JSON object" in last_user


def test_expand_client_prompt_is_forwarded():
    llm = FakeLLMClient(SITE_PAYLOAD)
    expand_site_with_prompt(
        {"business": BUSINESS, "copy": COPY}, "Make it feel rustic and warm.", client=llm
    )
    user_prompt = llm.chat.completions.calls[0]["messages"][-1]["content"]
    assert "Additional instructions from the client:" in user_prompt
    assert "Make it feel rustic and warm." in user_prompt


def test_expand_validation_clamps_theme_and_drops_unknown_pages():
    payload = json.dumps(
        {
            "theme": {"primary": "blue", "secondary": "#123456"},
            "pages": {
                "about.html": {"heading": "Who We Are", "paragraphs": ["Locally owned since 1998."]},
                "hacker.html": {"heading": "Evil", "paragraphs": ["x"]},
            },
        }
    )
    llm = FakeLLMClient(payload)
    result = expand_site_with_prompt({"business": BUSINESS, "copy": COPY}, "", client=llm)

    assert result["theme"]["primary"] == DEFAULT_THEME["primary"]  # invalid hex → default
    assert result["theme"]["secondary"] == "#123456"  # valid hex kept
    assert set(result["pages"]) == set(ALLOWED_SUBPAGES)  # hacker.html dropped
    assert "hacker.html" not in result["pages"]
    # missing standard pages filled with defaults
    assert result["pages"]["services.html"]["heading"] == "Our Services"
    assert result["pages"]["gallery.html"]["captions"]


def test_expand_empty_llm_output_still_yields_complete_site():
    llm = FakeLLMClient(json.dumps({"theme": {}, "pages": {}}))
    result = expand_site_with_prompt({"business": BUSINESS, "copy": COPY}, "", client=llm)

    assert result["theme"] == DEFAULT_THEME
    for filename in ALLOWED_SUBPAGES:
        page = result["pages"][filename]
        assert page["heading"], f"{filename} missing heading"
    # defaults derive from the landing copy where possible
    assert result["pages"]["services.html"]["items"][0]["title"] == "Repairs"
    assert result["pages"]["about.html"]["paragraphs"]


# ── compilation ───────────────────────────────────────────────────────────────


def _expanded_schema():
    return expand_site_with_prompt(
        {"business": BUSINESS, "copy": COPY}, "", client=FakeLLMClient(SITE_PAYLOAD)
    )


def test_render_produces_all_five_pages():
    pages = render_site_pages(_expanded_schema())
    assert set(pages) == {"index.html", *ALLOWED_SUBPAGES}
    for html in pages.values():
        assert html.startswith("<!DOCTYPE html>")


def test_render_theme_and_shared_nav_on_every_page():
    pages = render_site_pages(_expanded_schema())
    for filename, html in pages.items():
        assert "--sp-primary: #b45309" in html, f"{filename} missing theme var"
        assert "--sp-secondary: #7c2d12" in html, f"{filename} missing secondary"
        for target in ("index.html", *ALLOWED_SUBPAGES):
            assert f'href="{target}"' in html, f"{filename} missing nav link to {target}"
        assert "Acme Plumbing" in html  # header brand + footer


def test_render_index_keeps_landing_sections():
    html = render_site_pages(_expanded_schema())["index.html"]
    assert "Reliable plumbing for Eugene homes" in html  # hero headline
    assert "What We Do" in html
    assert 'href="tel:+5415550142"' in html  # click-to-call
    assert "Licensed &amp; Insured" in html  # autoescaped by Jinja
    assert 'name="business_slug" value="acme-plumbing"' in html
    assert 'action="/api/forms/submit"' in html


def test_render_subpages_carry_llm_content():
    pages = render_site_pages(_expanded_schema())

    about = pages["about.html"]
    assert "Our Story" in about
    assert "Founded in 1998, we have kept Eugene&#39;s pipes flowing." in about  # autoescaped

    services = pages["services.html"]
    assert "What We Offer" in services
    assert "Drain Cleaning" in services
    assert "We clear clogs fast." in services

    gallery = pages["gallery.html"]
    assert "Our Work" in gallery
    assert "Water heater swap" in gallery
    assert "Bathroom repipe" in gallery

    contact = pages["contact.html"]
    assert "Talk to Us" in contact
    assert 'action="/api/forms/submit"' in contact  # lead form on contact page
    assert 'name="business_slug" value="acme-plumbing"' in contact


def test_render_active_nav_state():
    import re

    active_class = "bg-slate-900/5 text-slate-900"
    for filename in ("index.html", *ALLOWED_SUBPAGES):
        html = pages_for(filename)
        # exactly one nav item is marked active…
        assert html.count(active_class) == 1, f"{filename}: expected one active nav item"
        # the nav item (not the brand link, which also points at index.html)
        match = re.search(rf'<a href="{re.escape(filename)}"\s+class="(px-3[^"]*)"', html)
        assert match and active_class in match.group(1), f"{filename}: active state on wrong link"


def pages_for(filename: str) -> str:
    return render_site_pages(_expanded_schema())[filename]


def test_render_falls_back_to_defaults_for_missing_pages():
    schema = _expanded_schema()
    schema["pages"]["gallery.html"] = None  # simulate malformed/missing page
    pages = render_site_pages(schema)
    assert "Recent Work" in pages["gallery.html"]
    assert "Repairs project" in pages["gallery.html"]  # captions derived from services


# ── deployment ────────────────────────────────────────────────────────────────


def test_deploy_uploads_every_page_under_slug_prefix():
    s3 = FakeS3()
    url = deploy_expanded_site("acme-plumbing", render_site_pages(_expanded_schema()), s3_client=s3)

    settings = get_settings()
    assert len(s3.puts) == 5
    keys = {p["Key"] for p in s3.puts}
    assert keys == {f"acme-plumbing/{f}" for f in ("index.html", *ALLOWED_SUBPAGES)}
    for put in s3.puts:
        assert put["Bucket"] == settings.r2_bucket_name
        assert put["ContentType"] == "text/html; charset=utf-8"
        assert isinstance(put["Body"], bytes) and put["Body"].startswith(b"<!DOCTYPE html>")
    assert url == f"{settings.r2_public_base_url.rstrip('/')}/acme-plumbing/index.html"
