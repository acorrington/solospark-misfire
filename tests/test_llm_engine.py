"""Tests for app/llm_engine.py — no live LLM required (fake OpenAI client)."""

import json

import pytest

from app import llm_engine
from app.llm_engine import (
    ALLOWED_ICONS,
    LLMGenerationError,
    generate_landing_copy,
    generate_pitch_email,
    refine_landing_copy,
    syntax_check_copy,
    validate_landing_copy,
)

VALID_COPY = {
    "tagline": "Eugene's trusted plumbers since 1998",
    "hero_headline": "Fast, honest plumbing for Eugene homes and businesses",
    "hero_subheadline": (
        "Locally owned and operated. We answer the phone, show up on time, "
        "and stand behind every repair."
    ),
    "services": [
        {"title": "Emergency Repairs", "description": "24/7 leak response.", "icon_name": "wrench"},
        {"title": "Water Heaters", "description": "Install and service.", "icon_name": "shield"},
        {"title": "Drain Cleaning", "description": "Cleared fast.", "icon_name": "clock"},
    ],
    "about_heading": "Your neighbors' plumbers",
    "about_text": "Acme Plumbing has served Eugene for 25 years. We keep it simple: fair prices, clean work.",
    "why_choose_us": ["Licensed & Insured", "Fast Response", "Upfront Pricing"],
    "cta_text": "Get Your Free Quote",
}


# ── Fake OpenAI-compatible client ────────────────────────────────────────────


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeCompletions:
    """Scripted chat.completions.create — pops responses in order.

    Once the script is exhausted, unscripted calls (e.g. the pre-save syntax
    QA pass inside generate/refine) echo the last scripted response, so a
    single-payload test scripts only the primary call and the QA pass is a
    no-op that returns the same copy unchanged.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.last_payload = responses[-1] if responses else None
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            if self.last_payload is None:
                raise AssertionError("fake LLM ran out of scripted responses")
            return _FakeCompletion(self.last_payload)
        return _FakeCompletion(self.responses.pop(0))


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeOpenAI:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


def make_client(*responses):
    return FakeOpenAI(list(responses))


# ── generate_landing_copy ────────────────────────────────────────────────────


def test_generate_landing_copy_returns_validated_schema():
    client = make_client(json.dumps(VALID_COPY))
    result = generate_landing_copy(
        "Acme Plumbing", "Plumber", "Eugene, OR", raw_info="24/7 emergency service", client=client
    )
    assert result["tagline"] == VALID_COPY["tagline"]
    assert result["hero_headline"].count(" ") >= 5  # 6-10 words
    assert len(result["services"]) == 3
    for svc in result["services"]:
        assert set(svc) == {"title", "description", "icon_name"}
        assert svc["icon_name"] in ALLOWED_ICONS
    assert len(result["why_choose_us"]) == 3
    assert result["cta_text"] == "Get Your Free Quote"


def test_generate_landing_copy_strips_markdown_fences():
    client = make_client("```json\n" + json.dumps(VALID_COPY) + "\n```")
    result = generate_landing_copy("Acme Plumbing", "Plumber", "Eugene, OR", client=client)
    assert result["hero_headline"] == VALID_COPY["hero_headline"]


def test_generate_landing_copy_prompt_contains_business_details():
    client = make_client(json.dumps(VALID_COPY))
    generate_landing_copy(
        "Acme Plumbing", "Plumber", "Eugene, OR", raw_info="SERVICING THE VALLEY SINCE 1998",
        client=client,
    )
    user_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    system_prompt = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "Acme Plumbing" in user_prompt
    assert "Plumber" in user_prompt
    assert "Eugene, OR" in user_prompt
    assert "SERVICING THE VALLEY SINCE 1998" in user_prompt
    assert "copywriter" in system_prompt.lower()


def test_generate_landing_copy_retries_on_bad_json_then_succeeds():
    client = make_client("Sure! Here is the copy you asked for: {not json", json.dumps(VALID_COPY))
    result = generate_landing_copy("Acme Plumbing", "Plumber", "Eugene, OR", client=client)
    assert result["tagline"] == VALID_COPY["tagline"]
    # generation + JSON-repair retry + the pre-save syntax QA pass
    assert len(client.chat.completions.calls) == 3
    # the retry prompt carries the JSON-only reminder
    assert "ONLY a valid JSON" in client.chat.completions.calls[1]["messages"][1]["content"]


def test_generate_landing_copy_raises_after_retries_exhausted():
    client = make_client("no json here", "still no json")
    with pytest.raises(LLMGenerationError):
        generate_landing_copy("Acme Plumbing", "Plumber", "Eugene, OR", client=client)
    assert len(client.chat.completions.calls) == 2


def test_generate_landing_copy_rejects_non_object_json():
    # a bare array is valid JSON but not an object → retried, then raised
    client = make_client('[1, 2, 3]', '[1, 2, 3]')
    with pytest.raises(LLMGenerationError):
        generate_landing_copy("Acme Plumbing", "Plumber", "Eugene, OR", client=client)
    assert len(client.chat.completions.calls) == 2


# ── validation / normalization ───────────────────────────────────────────────


def test_validate_landing_copy_clamps_unknown_icon():
    data = dict(VALID_COPY)
    data["services"] = [
        {"title": "A", "description": "d", "icon_name": "rocket"},
        {"title": "B", "description": "d", "icon_name": "phone"},
        {"title": "C", "description": "d", "icon_name": None},
    ]
    result = validate_landing_copy(data)
    assert [s["icon_name"] for s in result["services"]] == ["star", "phone", "star"]


def test_validate_landing_copy_truncates_services_to_six():
    data = dict(VALID_COPY)
    data["services"] = [
        {"title": f"S{i}", "description": "d", "icon_name": "wrench"} for i in range(8)
    ]
    result = validate_landing_copy(data)
    assert len(result["services"]) == 6


def test_validate_landing_copy_pads_trust_badges():
    data = dict(VALID_COPY)
    data["why_choose_us"] = ["Family Owned"]
    result = validate_landing_copy(data)
    assert result["why_choose_us"][0] == "Family Owned"
    assert len(result["why_choose_us"]) == 3


def test_validate_landing_copy_requires_core_fields():
    with pytest.raises(LLMGenerationError):
        validate_landing_copy({"tagline": "x"})
    data = dict(VALID_COPY)
    del data["services"]
    with pytest.raises(LLMGenerationError):
        validate_landing_copy(data)


# ── generate_pitch_email ─────────────────────────────────────────────────────


def test_generate_pitch_email_returns_subject_and_body():
    payload = {
        "subject": "Quick question about your website",
        "body": (
            "Hi — I noticed your site is missing mobile support and the copyright "
            "says 2014. I put together a free improved version: "
            "https://preview.solospark.net/acme-plumbing/ Take a look."
        ),
    }
    client = make_client(json.dumps(payload))
    result = generate_pitch_email(
        "Acme Plumbing", ["Missing Mobile Viewport", "Stale Copyright (2014)"],
        "https://preview.solospark.net/acme-plumbing/", client=client,
    )
    assert result["subject"] == payload["subject"]
    assert "preview.solospark.net" in result["body"]


def test_generate_pitch_email_prompt_references_flaws_and_url():
    flaws = ["SSL/Certificate Error", "HTTP Status 500"]
    url = "https://preview.solospark.net/joe-s-plumbing/"
    client = make_client(json.dumps({"subject": "s", "body": "b"}))
    generate_pitch_email("Joe's Plumbing", flaws, url, client=client)
    user_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "SSL/Certificate Error" in user_prompt
    assert "HTTP Status 500" in user_prompt
    assert url in user_prompt
    assert "non-salesy" in client.chat.completions.calls[0]["messages"][0]["content"]


def test_generate_pitch_email_rejects_missing_body():
    client = make_client(json.dumps({"subject": "only subject"}))
    with pytest.raises(LLMGenerationError):
        generate_pitch_email("Joe's Plumbing", ["flag"], "https://x.example/", client=client)


# ── SDK error wrapping in _chat_json ─────────────────────────────────────────


class _RaisingCompletions:
    """chat.completions.create that always raises a scripted exception."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self._exc


def test_chat_json_wraps_openai_sdk_errors():
    from openai import OpenAIError

    client = FakeOpenAI([])
    client.chat.completions = _RaisingCompletions(OpenAIError("connection refused"))
    with pytest.raises(LLMGenerationError, match="LLM request failed"):
        llm_engine._chat_json(client, "sys", "user", 100)
    # no repair retry: a down endpoint cannot be fixed by re-asking for JSON
    assert client.chat.completions.calls == 1


def test_chat_json_re_raises_unrelated_errors():
    client = FakeOpenAI([])
    client.chat.completions = _RaisingCompletions(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        llm_engine._chat_json(client, "sys", "user", 100)


# ── client construction from settings ────────────────────────────────────────


def test_build_client_uses_settings(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen-27b")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "sk-test-123")
    llm_engine.get_settings.cache_clear()
    try:
        client = llm_engine.build_client()
        # the openai SDK normalizes base_url with a trailing slash
        assert str(client.base_url).rstrip("/") == "http://127.0.0.1:9999/v1"
        assert client.api_key == "sk-test-123"
    finally:
        llm_engine.get_settings.cache_clear()


def test_build_client_defaults_api_key_for_lm_studio(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    llm_engine.get_settings.cache_clear()
    try:
        client = llm_engine.build_client()
        assert client.api_key == "lm-studio"  # LM Studio ignores the key entirely
    finally:
        llm_engine.get_settings.cache_clear()


# ── brand colors + site reference (regenerate feature) ───────────────────────


def test_normalize_brand_validates_and_lowercases():
    assert llm_engine._normalize_brand(
        {"primary": "#1D4ED8", "secondary": "#0F172A"}
    ) == {"primary": "#1d4ed8", "secondary": "#0f172a"}


def test_normalize_brand_partial_dict_keeps_valid_key():
    assert llm_engine._normalize_brand({"primary": "#1D4ED8"}) == {
        "primary": "#1d4ed8"
    }


def test_normalize_brand_rejects_invalid_input():
    assert llm_engine._normalize_brand({"primary": "navy", "secondary": "#GGGHHH"}) is None
    assert llm_engine._normalize_brand("nope") is None
    assert llm_engine._normalize_brand(None) is None


def test_validate_landing_copy_passes_brand_through():
    data = dict(VALID_COPY, brand={"primary": "#123456", "secondary": "#654321"})
    result = validate_landing_copy(data)
    assert result["brand"] == {"primary": "#123456", "secondary": "#654321"}

    bad = dict(VALID_COPY, brand={"primary": "bogus"})
    assert validate_landing_copy(bad)["brand"] is None


def test_generate_landing_copy_with_site_reference_uses_it_as_factual_basis():
    client = make_client(json.dumps(VALID_COPY))
    ref = {
        "headline": "Eugene's Trusted Plumbers Since 1998",
        "meta_description": "24/7 emergency plumbing in Eugene, OR.",
        "service_headings": ["Emergency Repairs"],
        "body_text": "Licensed and insured since 1998.",
    }
    result = generate_landing_copy(
        "Acme Plumbing", "Plumber", "Eugene, OR", site_reference=ref, client=client
    )
    assert result["tagline"] == VALID_COPY["tagline"]

    user_msgs = [
        m for c in client.chat.completions.calls for m in c["messages"] if m["role"] == "user"
    ]
    prompt = user_msgs[0]["content"]
    # scraped content is present and framed as the factual basis
    assert "Eugene's Trusted Plumbers Since 1998" in prompt
    assert "factual basis" in prompt
    assert "(none available" not in prompt


def test_generate_landing_copy_without_reference_says_none_available():
    client = make_client(json.dumps(VALID_COPY))
    generate_landing_copy("Acme Plumbing", "Plumber", "Eugene, OR", client=client)
    user_msgs = [
        m for c in client.chat.completions.calls for m in c["messages"] if m["role"] == "user"
    ]
    assert "(none available" in user_msgs[0]["content"]


def test_generate_landing_copy_brand_schema_requested_in_prompt():
    client = make_client(json.dumps(VALID_COPY))
    generate_landing_copy("Acme Plumbing", "Plumber", "Eugene, OR", client=client)
    user_msgs = [
        m for c in client.chat.completions.calls for m in c["messages"] if m["role"] == "user"
    ]
    prompt = user_msgs[0]["content"]
    assert '"brand"' in prompt


# ── refine_landing_copy (site fine-tuning feature) ───────────────────────────


def test_refine_prompt_carries_current_copy_and_instructions():
    current = dict(VALID_COPY, brand={"primary": "#1d4ed8", "secondary": "#0f172a"})
    client = make_client(json.dumps(VALID_COPY))
    refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        "make the hero headline shorter", client=client,
    )
    user_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    system_prompt = client.chat.completions.calls[0]["messages"][0]["content"]
    # current copy is embedded as the baseline to edit in place
    assert "Eugene's trusted plumbers since 1998" in user_prompt
    assert '"hero_headline"' in user_prompt
    # operator instructions are present and framed as the only change requested
    assert "make the hero headline shorter" in user_prompt
    assert "ONLY what the operator asks" in system_prompt.lower() or (
        "only what the operator asks" in system_prompt.lower()
    )


def test_refine_returns_validated_copy_with_instructions_applied():
    refined = dict(VALID_COPY, tagline="Eugene plumbers since 1998")
    client = make_client(json.dumps(refined))
    result = refine_landing_copy(
        VALID_COPY, "Acme Plumbing", "Plumber", "Eugene, OR",
        "shorten the tagline", client=client,
    )
    assert result["tagline"] == refined["tagline"]
    assert len(result["services"]) == 3


def test_refine_carries_over_brand_when_llm_omits_it():
    current = dict(VALID_COPY, brand={"primary": "#0a3d62", "secondary": "#3c89c8"})
    client = make_client(json.dumps(VALID_COPY))  # no brand key in the reply
    result = refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        "mention 24/7 service in the subheadline", client=client,
    )
    assert result["brand"] == {"primary": "#0a3d62", "secondary": "#3c89c8"}


def test_refine_keeps_llm_brand_when_it_returns_one():
    current = dict(VALID_COPY, brand={"primary": "#0a3d62", "secondary": "#3c89c8"})
    new_brand = {"primary": "#1e3a8a", "secondary": "#111827"}
    client = make_client(json.dumps(dict(VALID_COPY, brand=new_brand)))
    result = refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        "switch the palette to navy", client=client,
    )
    assert result["brand"] == new_brand


def test_refine_raises_when_llm_returns_invalid_json():
    client = make_client("not json at all", "still not json")
    with pytest.raises(LLMGenerationError):
        refine_landing_copy(
            VALID_COPY, "Acme Plumbing", "Plumber", "Eugene, OR",
            "anything", client=client,
        )
    assert len(client.chat.completions.calls) == 2


def test_refine_excludes_asset_paths_from_baseline():
    current = dict(
        VALID_COPY,
        logo_url="assets/logo.png",
        hero_image_url="assets/hero.jpg",
        about_images=["assets/about1.webp"],
    )
    client = make_client(json.dumps(VALID_COPY))
    refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        "tighten the CTA", client=client,
    )
    user_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    # asset paths are persisted by the caller, not part of the LLM schema
    assert "assets/logo.png" not in user_prompt
    assert "assets/hero.jpg" not in user_prompt
    assert "#rrggbb" in user_prompt.lower()


# ── Named-color safety net (local models ignore explicit color requests) ─────


def test_named_color_palette_requires_intent_and_name():
    assert llm_engine._named_color_palette("") is None
    assert llm_engine._named_color_palette("tighten the CTA") is None
    # a color mentioned without change intent must not re-theme the site
    assert (
        llm_engine._named_color_palette("remove the red text from the headline")
        is None
    )
    assert llm_engine._named_color_palette("change the color scheme to red") == {
        "primary": "#b91c1c",
        "secondary": "#7f1d1d",
    }


def test_named_color_palette_prefers_specific_names():
    # "navy blue" must resolve to navy, not plain blue (dict order)
    assert llm_engine._named_color_palette("switch the palette to navy blue") == {
        "primary": "#1e3a8a",
        "secondary": "#0f172a",
    }


def test_enforce_named_color_no_op_without_request():
    copy = {"brand": {"primary": "#0f2a4a", "secondary": "#4a7fa8"}}
    result = llm_engine.enforce_named_color(copy, "tighten the CTA")
    assert result["brand"] == {"primary": "#0f2a4a", "secondary": "#4a7fa8"}


def test_enforce_named_color_overrides_unchanged_palette():
    old = {"primary": "#0f2a4a", "secondary": "#4a7fa8"}
    copy = {"brand": dict(old)}  # model echoed the baseline back
    result = llm_engine.enforce_named_color(
        copy, "change the color scheme to red", previous_brand=old
    )
    assert result["brand"] == {"primary": "#b91c1c", "secondary": "#7f1d1d"}


def test_enforce_named_color_fills_missing_brand():
    old = {"primary": "#0f2a4a", "secondary": "#4a7fa8"}
    result = llm_engine.enforce_named_color(
        {}, "change the color scheme to red", previous_brand=old
    )
    assert result["brand"] == {"primary": "#b91c1c", "secondary": "#7f1d1d"}


def test_enforce_named_color_trusts_a_changed_palette():
    old = {"primary": "#0f2a4a", "secondary": "#4a7fa8"}
    changed = {"primary": "#dc2626", "secondary": "#991b1b"}  # model picked its own red
    result = llm_engine.enforce_named_color(
        {"brand": dict(changed)},
        "change the color scheme to red",
        previous_brand=old,
    )
    assert result["brand"] == changed


def test_refine_applies_named_color_when_model_keeps_baseline():
    current = dict(VALID_COPY, brand={"primary": "#0f2a4a", "secondary": "#4a7fa8"})
    # the model ignores the request and returns the baseline palette unchanged
    client = make_client(json.dumps(current))
    result = refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        "change the color scheme to red", client=client,
    )
    assert result["brand"] == {"primary": "#b91c1c", "secondary": "#7f1d1d"}


def test_refine_applies_named_color_when_model_drops_brand():
    current = dict(VALID_COPY, brand={"primary": "#0f2a4a", "secondary": "#4a7fa8"})
    # the model omits brand entirely → carry-over restores the old palette,
    # then the named-color enforcement replaces it with the requested one
    client = make_client(json.dumps(VALID_COPY))
    result = refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        "change the color scheme to red", client=client,
    )
    assert result["brand"] == {"primary": "#b91c1c", "secondary": "#7f1d1d"}


def test_refine_named_color_trusts_a_changed_palette():
    current = dict(VALID_COPY, brand={"primary": "#0f2a4a", "secondary": "#4a7fa8"})
    changed = {"primary": "#dc2626", "secondary": "#991b1b"}
    client = make_client(json.dumps(dict(VALID_COPY, brand=changed)))
    result = refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        "change the color scheme to red", client=client,
    )
    assert result["brand"] == changed


def test_generate_prompt_marks_operator_instructions_as_overriding():
    client = make_client(json.dumps(VALID_COPY))
    generate_landing_copy(
        "Acme Plumbing", "Plumber", "Eugene, OR",
        extra_instructions="change the color scheme to red", client=client,
    )
    user_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "OVERRIDE" in user_prompt
    assert "change the color scheme to red" in user_prompt


# ── apply_explicit_edits — quoted replacement values ─────────────────────────


def test_explicit_edit_sets_cta_verbatim():
    current = dict(VALID_COPY, brand={"primary": "#0f2a4a", "secondary": "#4a7fa8"})
    # the model ignores the request and returns the baseline unchanged — the
    # quoted value must still land verbatim
    client = make_client(json.dumps(current))
    result = refine_landing_copy(
        current, "Acme Plumbing", "Plumber", "Eugene, OR",
        'change the CTA button text to "Book Now"', client=client,
    )
    assert result["cta_text"] == "Book Now"


def test_explicit_edit_sets_headline_verbatim():
    client = make_client(json.dumps(VALID_COPY))
    result = refine_landing_copy(
        VALID_COPY, "Acme Plumbing", "Plumber", "Eugene, OR",
        'make the headline "Drain Problems? We\'re On It."', client=client,
    )
    assert result["hero_headline"] == "Drain Problems? We're On It."


def test_explicit_edit_nearest_keyword_wins():
    # "button" appears AFTER the quoted value; "headline" is closer → headline
    client = make_client(json.dumps(VALID_COPY))
    result = refine_landing_copy(
        VALID_COPY, "Acme Plumbing", "Plumber", "Eugene, OR",
        'change the headline to "We Fix Drains" and keep the button as is',
        client=client,
    )
    assert result["hero_headline"] == "We Fix Drains"
    assert result["cta_text"] == VALID_COPY["cta_text"]


def test_explicit_edit_applies_multiple_values():
    client = make_client(json.dumps(VALID_COPY))
    result = refine_landing_copy(
        VALID_COPY, "Acme Plumbing", "Plumber", "Eugene, OR",
        'set the tagline to "Fast, honest, local" and update the CTA button '
        'to "Book My Free Quote"',
        client=client,
    )
    assert result["tagline"] == "Fast, honest, local"
    assert result["cta_text"] == "Book My Free Quote"


def test_explicit_edit_ignores_mentions_without_intent():
    copy = {"cta_text": "Get Your Free Quote"}
    result = llm_engine.apply_explicit_edits(
        copy, 'the "Book Now" button should stay as is'
    )
    assert result["cta_text"] == "Get Your Free Quote"


def test_explicit_edit_ignores_unquoted_values():
    # only quoted values are enforced deterministically; unquoted text is left
    # to the LLM (avoids grabbing fragments like "be shorter")
    copy = {"cta_text": "Get Your Free Quote"}
    result = llm_engine.apply_explicit_edits(
        copy, "change the CTA button to Book Now"
    )
    assert result["cta_text"] == "Get Your Free Quote"


def test_explicit_edit_ignores_quoted_value_without_field_keyword():
    copy = {
        "hero_headline": VALID_COPY["hero_headline"],
        "cta_text": "Get Your Free Quote",
    }
    result = llm_engine.apply_explicit_edits(
        copy, 'change the about text to "We love plumbing"'
    )
    assert result == copy


def test_explicit_edit_targets_contact_heading():
    client = make_client(json.dumps(VALID_COPY))
    result = refine_landing_copy(
        VALID_COPY, "Acme Plumbing", "Plumber", "Eugene, OR",
        'change the contact heading to "Request a Free Estimate"',
        client=client,
    )
    assert result["contact_heading"] == "Request a Free Estimate"


# ── Category-aware defaults (de-canned copy) ────────────────────────────────


def test_food_category_gets_reservation_headings():
    client = make_client(json.dumps(VALID_COPY))
    copy = generate_landing_copy("Luigi's Pizzeria", "Restaurant", "Portland", client=client)
    assert copy["contact_heading"] == "Reserve a Table"
    assert copy["services_heading"] == "What We Serve"
    assert copy["cta_band_subtext"]


def test_trade_category_gets_estimate_headings():
    client = make_client(json.dumps(VALID_COPY))
    copy = generate_landing_copy("Acme Plumbing", "Plumber", "Eugene", client=client)
    assert copy["contact_heading"] == "Request a Free Estimate"
    # Quote language is legitimate for trades — the LLM CTA survives untouched.
    assert copy["cta_text"] == "Get Your Free Quote"


def test_trade_intro_mentions_city():
    client = make_client(json.dumps(VALID_COPY))
    copy = generate_landing_copy("Acme Plumbing", "Plumber", "Eugene", client=client)
    assert "Eugene" in copy["services_intro"]


def test_generic_category_gets_neutral_headings():
    client = make_client(json.dumps(VALID_COPY))
    copy = generate_landing_copy("Bright Consulting", "Consulting", "Eugene", client=client)
    assert copy["contact_heading"] == "Get in Touch"
    assert copy["services_heading"] == "What We Offer"


def test_food_guard_replaces_quote_cta_and_heading():
    payload = dict(VALID_COPY, contact_heading="Request a free quote")
    client = make_client(json.dumps(payload))
    copy = generate_landing_copy("Luigi's Pizzeria", "Restaurant", "Portland", client=client)
    assert copy["cta_text"] == "Reserve a Table"
    assert copy["contact_heading"] == "Reserve a Table"


def test_booking_guard_replaces_quote_cta():
    payload = dict(VALID_COPY, cta_text="Book Your Free Quote")
    client = make_client(json.dumps(payload))
    copy = generate_landing_copy("Glow Salon", "Salon", "Portland", client=client)
    assert copy["cta_text"] == "Book an Appointment"
    assert copy["contact_heading"] == "Book an Appointment"


def test_llm_written_headings_are_trusted():
    payload = dict(VALID_COPY, contact_heading="Book Your Private Dining Room")
    client = make_client(json.dumps(payload))
    copy = generate_landing_copy("Luigi's Pizzeria", "Restaurant", "Portland", client=client)
    assert copy["contact_heading"] == "Book Your Private Dining Room"


def test_bar_and_barber_classification():
    # word-boundary match: a bar is food, a barber shop is booking
    assert llm_engine._category_profile("Bar & Grill") == "food"
    assert llm_engine._category_profile("Barber Shop") == "booking"


def test_apply_category_defaults_only_fills_missing():
    copy = {"cta_text": "Reserve Online", "contact_heading": "Book Your Table"}
    result = llm_engine.apply_category_defaults(copy, "Cafe", "Portland")
    assert result["contact_heading"] == "Book Your Table"  # LLM value kept
    assert result["services_heading"] == "What We Serve"   # gap filled


def test_refine_fills_missing_headings_from_category():
    # old copy predates the section-heading fields
    client = make_client(json.dumps(VALID_COPY))
    refined = refine_landing_copy(
        VALID_COPY, "Luigi's Pizzeria", "Restaurant", "Portland",
        "make the tagline punchier", client=client,
    )
    assert refined["contact_heading"] == "Reserve a Table"
    assert refined["cta_text"] == "Reserve a Table"  # quote CTA guarded


def test_explicit_contact_heading_edit_beats_food_guard():
    # an operator's quoted replacement is the final word, even when it says
    # "quote" for a restaurant
    client = make_client(json.dumps(VALID_COPY))
    refined = refine_landing_copy(
        VALID_COPY, "Luigi's Pizzeria", "Restaurant", "Portland",
        'change the contact heading to "Request a free quote"', client=client,
    )
    assert refined["contact_heading"] == "Request a free quote"


# ── Section composition (AI-picked layouts) ──────────────────────────────────


def test_resolve_layout_trusts_llm_order_and_pins_hero_contact():
    copy = {
        "layout": ["contact", "hero", "menu", "services"],
        "menu_items": [{"name": "Margherita", "description": "", "price": "$14"}],
        "services": [{"title": "Toppings", "description": "x", "icon_name": "star"}],
    }
    assert llm_engine.resolve_layout(copy, "Restaurant") == [
        "hero", "menu", "services", "contact",
    ]


def test_resolve_layout_falls_back_when_too_few_known_sections():
    # two known sections is not a layout — the category default applies
    copy = {
        "layout": ["hero", "menu"],
        "menu_items": [{"name": "Margherita", "description": "", "price": "$14"}],
        "services": [{"title": "Toppings", "description": "x", "icon_name": "star"}],
        "about_text": "Family run since 1980.",
        "hours": [{"day": "Monday", "hours": "11am-9pm"}],
    }
    assert llm_engine.resolve_layout(copy, "Restaurant") == [
        "hero", "menu", "services", "about", "hours_location", "cta_band", "contact",
    ]


def test_resolve_layout_drops_sections_without_data():
    copy = {
        # LLM asked for a menu and hours, but the reference had none — the
        # page must not show empty blocks.
        "layout": ["hero", "menu", "hours_location", "services", "about", "contact"],
        "services": [{"title": "Drains", "description": "x", "icon_name": "wrench"}],
        "about_text": "Twenty years in the trade.",
    }
    assert llm_engine.resolve_layout(copy, "Plumber") == [
        "hero", "services", "about", "contact",
    ]


def test_resolve_layout_ignores_unknown_and_duplicate_sections():
    copy = {
        "layout": ["hero", "hero", "pricer", "menu", "menu", "contact"],
        "menu_items": [{"name": "Margherita", "description": "", "price": "$14"}],
    }
    # only hero/menu/contact survive → trusted, unknowns and dupes dropped
    assert llm_engine.resolve_layout(copy, "Restaurant") == [
        "hero", "menu", "contact",
    ]


def test_resolve_layout_classic_mode_uses_fixed_template_order():
    # Classic mode forces the category's fixed template order (trade profile
    # for "Plumber"), ignoring whatever layout the LLM picked — even when the
    # LLM's pick is otherwise valid.
    copy = {
        "layout": ["hero", "menu", "contact"],
        "menu_items": [{"name": "Drain check", "description": "", "price": "$49"}],
        "services": [{"title": "Drains", "description": "x", "icon_name": "wrench"}],
        "about_text": "Twenty years in the trade.",
    }
    assert llm_engine.resolve_layout(copy, "Plumber", layout_mode="classic") == [
        "hero", "services", "about", "cta_band", "contact",
    ]


def test_resolve_layout_classic_mode_still_drops_sections_without_data():
    copy = {
        "layout": ["hero", "menu", "contact"],  # irrelevant in classic mode
        "menu_items": [{"name": "Drain check", "description": "", "price": "$49"}],
    }
    # trade default minus the data-less sections (services, about)
    assert llm_engine.resolve_layout(copy, "Plumber", layout_mode="classic") == [
        "hero", "cta_band", "contact",
    ]


def test_resolve_layout_unknown_layout_mode_treated_as_ai():
    copy = {
        "layout": ["hero", "menu", "contact"],
        "menu_items": [{"name": "Margherita", "description": "", "price": "$14"}],
    }
    assert llm_engine.resolve_layout(copy, "Restaurant", layout_mode="bogus") == [
        "hero", "menu", "contact",
    ]


def test_nav_links_for_caps_labels_and_always_ends_with_contact():
    layout = ["hero", "menu", "services", "gallery", "about", "hours_location", "cta_band", "contact"]
    assert llm_engine.nav_links_for(layout) == [
        ("Menu", "#menu"),
        ("Services", "#services"),
        ("Gallery", "#gallery"),
        ("About", "#about"),
        ("Contact", "#contact"),
    ]


def test_nav_links_for_minimal_layout():
    assert llm_engine.nav_links_for(["hero", "contact"]) == [("Contact", "#contact")]


# ── menu_items / hours / layout validation ───────────────────────────────────


def test_validate_normalizes_menu_items():
    payload = dict(
        VALID_COPY,
        menu_items=[
            {"name": "Margherita", "description": "Tomato, mozzarella", "price": "$14"},
            {"name": ""},                      # no name → dropped
            "not-a-dict",                      # wrong shape → dropped
            {"name": "Calzone", "price": 12},  # non-string price → ""
        ],
    )
    client = make_client(json.dumps(payload))
    copy = generate_landing_copy("Luigi's Pizzeria", "Restaurant", "Portland", client=client)
    assert copy["menu_items"] == [
        {"name": "Margherita", "description": "Tomato, mozzarella", "price": "$14"},
        {"name": "Calzone", "description": "", "price": ""},
    ]


def test_validate_normalizes_hours_dict_and_list_forms():
    # dict form ({"monday": "9am-5pm"}) is tolerated and normalized
    payload = dict(VALID_COPY, hours={"Monday": " 9am-5pm ", "Tuesday": "", "Wednesday": "10am-6pm"})
    client = make_client(json.dumps(payload))
    copy = generate_landing_copy("Luigi's Pizzeria", "Restaurant", "Portland", client=client)
    assert copy["hours"] == [
        {"day": "Monday", "hours": "9am-5pm"},
        {"day": "Wednesday", "hours": "10am-6pm"},
    ]

    # list form dedupes by day (first entry wins)
    payload = dict(VALID_COPY, hours=[
        {"day": "Monday", "hours": "9am-5pm"},
        {"day": "Monday", "hours": "8am-4pm"},
        {"day": 12, "hours": "x"},   # non-string day → dropped
    ])
    client = make_client(json.dumps(payload))
    copy = generate_landing_copy("Luigi's Pizzeria", "Restaurant", "Portland", client=client)
    assert copy["hours"] == [{"day": "Monday", "hours": "9am-5pm"}]


def test_validate_layout_lowercases_dedupes_and_caps():
    payload = dict(VALID_COPY, layout=["Hero", "hero", "MENU", "menu", "services", "Contact"])
    client = make_client(json.dumps(payload))
    copy = generate_landing_copy("Luigi's Pizzeria", "Restaurant", "Portland", client=client)
    assert copy["layout"] == ["hero", "menu", "services", "contact"]


def test_refine_carries_over_menu_and_hours_when_llm_drops_them():
    # the model is told to keep menu/hours unchanged, but if it drops them the
    # previous values must survive fine-tuning (no data loss on refine)
    current = dict(
        VALID_COPY,
        menu_items=[{"name": "Margherita", "description": "", "price": "$14"}],
        hours=[{"day": "Monday", "hours": "9am-5pm"}],
    )
    client = make_client(json.dumps(VALID_COPY))  # LLM omits menu_items/hours
    refined = refine_landing_copy(
        current, "Luigi's Pizzeria", "Restaurant", "Portland",
        "make the tagline punchier", client=client,
    )
    assert refined["menu_items"] == current["menu_items"]
    assert refined["hours"] == current["hours"]


def test_refine_does_not_carry_over_empty_menu():
    # carry-over only fills gaps — an empty menu in the new copy stays empty
    current = dict(VALID_COPY)  # no menu_items/hours at all
    client = make_client(json.dumps(VALID_COPY))
    refined = refine_landing_copy(
        current, "Luigi's Pizzeria", "Restaurant", "Portland",
        "make the tagline punchier", client=client,
    )
    assert refined["menu_items"] == []
    assert refined["hours"] == []


# ── syntax_check_copy (pre-save QA pass) ─────────────────────────────────────


def test_syntax_check_applies_model_fixes_and_preserves_keys():
    broken = {**VALID_COPY, "hero_headline": "Best {city} plumbers in town"}
    fixed = {**broken, "hero_headline": "Best Eugene plumbers in town"}
    client = make_client(json.dumps(fixed))

    result = syntax_check_copy(
        broken, "Acme Plumbing", "Plumber", city="Eugene", client=client
    )

    assert result["hero_headline"] == "Best Eugene plumbers in town"
    # every key of the original copy survived the QA round trip
    assert set(broken.keys()) <= set(result.keys())


def test_syntax_check_keeps_original_when_model_drops_a_key():
    broken = {**VALID_COPY, "hero_headline": "Truncated headline"}
    client = make_client(json.dumps({"tagline": VALID_COPY["tagline"]}))

    result = syntax_check_copy(
        broken, "Acme Plumbing", "Plumber", city="Eugene", client=client
    )

    assert result == broken


def test_syntax_check_keeps_original_when_model_returns_garbage():
    # both the initial attempt and the JSON-repair retry fail → original kept
    client = make_client("sure, here you go!", "```json\n{broken")

    result = syntax_check_copy(
        VALID_COPY, "Acme Plumbing", "Plumber", city="Eugene", client=client
    )

    assert result == VALID_COPY


def test_syntax_check_restores_asset_keys_from_original():
    original = {
        **VALID_COPY,
        "logo_url": "assets/logo.png",
        "gallery_images": ["assets/g1.jpg"],
    }
    fixed = {**original, "hero_headline": "Fixed headline"}
    # the model "fixes" the asset references — that must be discarded
    fixed["logo_url"] = "https://evil.example/stolen.png"
    fixed["gallery_images"] = []
    client = make_client(json.dumps(fixed))

    result = syntax_check_copy(
        original, "Acme Plumbing", "Plumber", city="Eugene", client=client
    )

    assert result["hero_headline"] == "Fixed headline"
    assert result["logo_url"] == "assets/logo.png"
    assert result["gallery_images"] == ["assets/g1.jpg"]
