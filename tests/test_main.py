"""Phase 5 tests — dashboard, split-screen studio, and action endpoints.

Every endpoint is exercised through ``create_app`` with injected fakes so no
live LLM, R2, or outreach service is required.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.pipeline as pipeline
from app import models
from app.config import get_settings
from app.main import create_app

VALID_COPY = {
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


# ── fakes ────────────────────────────────────────────────────────────────────


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
        self.last_payload = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.payloads:
            # The pre-save syntax QA pass makes a second LLM call per
            # generation; tests that script a single response get it repeated
            # as a no-op "no fixes needed" round trip.
            assert self.last_payload is not None, (
                "FakeLLM called with no scripted responses"
            )
            return _Completion(self.last_payload)
        self.last_payload = self.payloads.pop(0)
        return _Completion(self.last_payload)


class FakeChat:
    def __init__(self, *payloads):
        self.completions = FakeCompletions(*payloads)


class FakeLLMClient:
    def __init__(self, *payloads):
        self.chat = FakeChat(*payloads)


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {}


# ── helpers ──────────────────────────────────────────────────────────────────


def make_test_client(tmp_path, llm_payloads=None, s3=None, sender=None):
    """Fresh temp DB + app with injected fakes. Returns (TestClient, app)."""
    models.reinit_db(f"sqlite:///{tmp_path / 'test.db'}")
    get_settings.cache_clear()
    app = create_app(
        llm_client=FakeLLMClient(*llm_payloads) if llm_payloads is not None else None,
        s3_client=s3 or FakeS3(),
        outreach_sender=sender,
    )
    return TestClient(app), app


_seed_counter = 0


def seed_business(**kw):
    """Insert a business into the current test DB and return its id."""
    global _seed_counter
    _seed_counter += 1
    defaults = dict(
        place_id=kw.pop("place_id", f"pid-{_seed_counter}"),
        name="Acme Plumbing",
        slug="acme-plumbing",
        category="Plumber",
        address="1 Main St, Eugene, OR 97401",
        phone="(541) 555-0142",
        rating=4.6,
        review_count=30,
        contact_email="owner@acme.com",
    )
    defaults.update(kw)
    factory = models.get_session_factory()
    with factory() as db:
        b = models.Business(**defaults)
        db.add(b)
        db.commit()
        db.refresh(b)
        return b.id


def get_business(business_id):
    factory = models.get_session_factory()
    with factory() as db:
        return db.get(models.Business, business_id)


# ── GET / — pipeline table ───────────────────────────────────────────────────


def test_dashboard_lists_businesses(tmp_path):
    client, _ = make_test_client(tmp_path)
    seed_business()
    seed_business(name="Beta HVAC", slug="beta-hvac")
    r = client.get("/")
    assert r.status_code == 200
    assert "Acme Plumbing" in r.text
    assert "Beta HVAC" in r.text


def test_dashboard_stage_filter(tmp_path):
    client, _ = make_test_client(tmp_path)
    seed_business(stage=models.DealStage.WON)
    seed_business(name="Beta HVAC", slug="beta-hvac", stage=models.DealStage.DISCOVERED)
    r = client.get("/", params={"stage": "won"})
    assert r.status_code == 200
    assert "Acme Plumbing" in r.text
    assert "Beta HVAC" not in r.text


def test_dashboard_invalid_stage_400(tmp_path):
    client, _ = make_test_client(tmp_path)
    assert client.get("/", params={"stage": "bogus"}).status_code == 400


def test_dashboard_flag_and_rating_filters(tmp_path):
    client, _ = make_test_client(tmp_path)
    seed_business(audit_flags=json.dumps(["No Website"]), rating=4.5)
    seed_business(
        name="Beta HVAC",
        slug="beta-hvac",
        audit_flags=json.dumps(["Stale Copyright (2019)"]),
        rating=3.0,
    )
    r = client.get("/", params={"flag": "No Website", "min_rating": 4})
    assert r.status_code == 200
    assert "Acme Plumbing" in r.text
    assert "Beta HVAC" not in r.text


# ── GET /review/{business_id} — split-screen studio ─────────────────────────


def test_review_page_with_preview(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business(preview_url="https://preview.solospark.net/acme-plumbing/index.html")
    r = client.get(f"/review/{bid}")
    assert r.status_code == 200
    assert "owner@acme.com" in r.text
    assert 'iframe src="https://preview.solospark.net/acme-plumbing/index.html"' in r.text


def test_review_page_without_preview(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business()
    r = client.get(f"/review/{bid}")
    assert r.status_code == 200
    assert "<iframe" not in r.text
    assert f"/generate/{bid}" in r.text


def test_review_unknown_business_404(tmp_path):
    client, _ = make_test_client(tmp_path)
    assert client.get("/review/999").status_code == 404


# ── POST /generate/{business_id} — LLM → Jinja2 → R2 ────────────────────────


def test_generate_endpoint_full_pipeline(tmp_path):
    s3 = FakeS3()
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3
    )
    bid = seed_business()
    r = client.post(f"/generate/{bid}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "generated"
    base = get_settings().r2_public_base_url.rstrip("/")
    assert body["preview_url"] == f"{base}/acme-plumbing/index.html"

    assert len(s3.puts) == 1
    put = s3.puts[0]
    assert put["Bucket"] == "solospark-previews"
    assert put["Key"] == "acme-plumbing/index.html"
    assert b"Proudly local" in put["Body"]

    b = get_business(bid)
    assert b.stage == models.DealStage.MOCKUP_READY
    assert b.preview_url == body["preview_url"]
    assert b.generated_copy_dict()["tagline"] == "Proudly local"


def test_generate_unknown_business_404(tmp_path):
    client, _ = make_test_client(tmp_path, llm_payloads=[json.dumps(VALID_COPY)])
    assert client.post("/generate/999").status_code == 404


def test_generate_llm_failure_502(tmp_path):
    client, _ = make_test_client(
        tmp_path, llm_payloads=["not json at all", "still not json"]
    )
    bid = seed_business()
    r = client.post(f"/generate/{bid}")
    assert r.status_code == 502


# ── POST /regenerate-prompt/{business_id} ───────────────────────────────────


def test_regenerate_prompt_sends_instructions(tmp_path):
    s3 = FakeS3()
    client, app = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3
    )
    bid = seed_business()
    r = client.post(f"/regenerate-prompt/{bid}", json={"prompt": "make it playful"})
    assert r.status_code == 200
    assert r.json()["status"] == "regenerated"

    calls = app.state.llm_client.chat.completions.calls
    user_messages = [
        m for c in calls for m in c["messages"] if m["role"] == "user"
    ]
    assert any("make it playful" in m["content"] for m in user_messages)

    b = get_business(bid)
    assert b.stage == models.DealStage.MOCKUP_READY
    assert b.generated_copy_dict()["tagline"] == "Proudly local"


def test_regenerate_prompt_enforces_named_color_when_model_repeats_old_brand(tmp_path):
    s3 = FakeS3()
    old_copy = {
        **VALID_COPY,
        "brand": {"primary": "#0f2a4a", "secondary": "#4a7fa8"},
    }
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(old_copy)], s3=s3
    )
    bid = seed_business(generated_copy=json.dumps(old_copy))

    r = client.post(
        f"/regenerate-prompt/{bid}",
        json={"prompt": "change the color scheme to red"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "regenerated"

    # the model echoed the old navy palette back — the named-color safety net
    # must replace it with the canonical red the operator asked for
    b = get_business(bid)
    copy = b.generated_copy_dict()
    assert copy["brand"] == {"primary": "#b91c1c", "secondary": "#7f1d1d"}
    # and the deployed page actually carries the new hexes
    html_body = s3.puts[0]["Body"]
    assert b"#b91c1c" in html_body
    assert b"#0f2a4a" not in html_body


def test_regenerate_prompt_enforces_named_color_when_model_drops_brand(tmp_path):
    s3 = FakeS3()
    old_copy = {
        **VALID_COPY,
        "brand": {"primary": "#0f2a4a", "secondary": "#4a7fa8"},
    }
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3  # no brand key
    )
    bid = seed_business(generated_copy=json.dumps(old_copy))

    r = client.post(
        f"/regenerate-prompt/{bid}",
        json={"prompt": "change the color scheme to red"},
    )
    assert r.status_code == 200
    b = get_business(bid)
    assert b.generated_copy_dict()["brand"] == {
        "primary": "#b91c1c",
        "secondary": "#7f1d1d",
    }


def test_regenerate_prompt_trusts_a_changed_palette(tmp_path):
    s3 = FakeS3()
    old_copy = {
        **VALID_COPY,
        "brand": {"primary": "#0f2a4a", "secondary": "#4a7fa8"},
    }
    changed = {**VALID_COPY, "brand": {"primary": "#dc2626", "secondary": "#991b1b"}}
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(changed)], s3=s3
    )
    bid = seed_business(generated_copy=json.dumps(old_copy))

    r = client.post(
        f"/regenerate-prompt/{bid}",
        json={"prompt": "change the color scheme to red"},
    )
    assert r.status_code == 200
    # the model actually changed the palette → its choice is kept as-is
    b = get_business(bid)
    assert b.generated_copy_dict()["brand"] == {
        "primary": "#dc2626",
        "secondary": "#991b1b",
    }


def test_regenerate_prompt_applies_explicit_cta_edit(tmp_path):
    s3 = FakeS3()
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3
    )  # the model returns its own copy — the quoted value must win
    bid = seed_business(generated_copy=json.dumps(VALID_COPY))

    r = client.post(
        f"/regenerate-prompt/{bid}",
        json={"prompt": 'change the CTA button to "Book My Free Quote"'},
    )
    assert r.status_code == 200
    b = get_business(bid)
    assert b.generated_copy_dict()["cta_text"] == "Book My Free Quote"
    # and the exact text is what got deployed
    html_body = s3.puts[0]["Body"]
    assert b"Book My Free Quote" in html_body


# ── POST /refine/{business_id} — fine-tune existing site, no re-scrape ───────


class FakeS3WithDelete(FakeS3):
    """Fake S3 that also supports list_objects_v2 + delete_objects."""

    def __init__(self, keys=None, fail_list=False):
        super().__init__()
        self.keys = list(keys or [])
        self.list_calls = []
        self.deleted = []
        self.fail_list = fail_list

    def list_objects_v2(self, **kwargs):
        if self.fail_list:
            raise RuntimeError("r2 down")
        self.list_calls.append(kwargs)
        prefix = kwargs.get("Prefix", "")
        contents = [{"Key": k} for k in self.keys if k.startswith(prefix)]
        return {"IsTruncated": False, "Contents": contents}

    def delete_objects(self, **kwargs):
        self.deleted.extend(o["Key"] for o in kwargs["Delete"]["Objects"])
        return {}


def test_refine_requires_existing_copy(tmp_path):
    client, app = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)]
    )
    bid = seed_business()  # no generated_copy yet
    r = client.post(f"/refine/{bid}", json={"prompt": "shorten the headline"})
    assert r.status_code == 400
    assert "generate it first" in r.json()["detail"]
    # nothing reached the LLM
    assert app.state.llm_client.chat.completions.calls == []


def test_refine_requires_prompt(tmp_path):
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)]
    )
    bid = seed_business(generated_copy=json.dumps(VALID_COPY))
    assert client.post(f"/refine/{bid}", json={"prompt": "   "}).status_code == 400
    r = client.post(f"/refine/{bid}", json={})
    assert r.status_code == 400
    assert "required" in r.json()["detail"]


def test_refine_happy_path_no_scrape_preserves_assets_and_stage(tmp_path, monkeypatch):
    import app.main as main_mod

    s3 = FakeS3WithDelete()
    client, app = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3
    )
    current = {
        **VALID_COPY,
        "brand": {"primary": "#0a3d62", "secondary": "#3c89c8"},
        "logo_url": "assets/logo.png",
        "hero_image_url": "assets/hero.jpg",
        "about_images": ["assets/about1.webp"],
    }
    bid = seed_business(
        current_website="https://www.acmeplumbing.com",  # would be scraped on /generate
        generated_copy=json.dumps(current),
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
        stage=models.DealStage.CONTACTED,
    )

    def no_scrape(url, http_session=None):
        raise AssertionError("refine must not re-scrape the customer's website")

    monkeypatch.setattr(main_mod, "scrape_site_reference", no_scrape)

    r = client.post(
        f"/refine/{bid}", json={"prompt": "mention 24/7 service in the subheadline"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "refined"
    base = get_settings().r2_public_base_url.rstrip("/")
    assert body["preview_url"] == f"{base}/acme-plumbing/index.html"

    # the LLM saw the current copy as its baseline + the operator instructions
    user_msgs = [
        m
        for c in app.state.llm_client.chat.completions.calls
        for m in c["messages"]
        if m["role"] == "user"
    ]
    assert any("mention 24/7 service" in m["content"] for m in user_msgs)
    assert any('"hero_headline"' in m["content"] for m in user_msgs)

    # only the page is re-uploaded — no asset puts (they already live in R2)
    assert [p["Key"] for p in s3.puts] == ["acme-plumbing/index.html"]
    html_body = s3.puts[0]["Body"]
    assert b"Proudly local" in html_body
    # persisted asset references survived the refine and rendered into the page
    assert b'src="assets/logo.png"' in html_body
    assert b'src="assets/hero.jpg"' in html_body

    b = get_business(bid)
    copy = b.generated_copy_dict()
    assert copy["tagline"] == "Proudly local"
    # LLM omitted brand → previous palette carried over
    assert copy["brand"] == {"primary": "#0a3d62", "secondary": "#3c89c8"}
    assert copy["logo_url"] == "assets/logo.png"
    assert copy["hero_image_url"] == "assets/hero.jpg"
    assert copy["about_images"] == ["assets/about1.webp"]
    # a real sales stage is never rolled back by fine-tuning
    assert b.stage == models.DealStage.CONTACTED


def test_refine_enforces_named_color_when_model_keeps_baseline(tmp_path):
    s3 = FakeS3WithDelete()
    current = {
        **VALID_COPY,
        "brand": {"primary": "#0f2a4a", "secondary": "#4a7fa8"},
    }
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(current)], s3=s3
    )
    bid = seed_business(
        generated_copy=json.dumps(current),
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
    )

    r = client.post(
        f"/refine/{bid}", json={"prompt": "change the color scheme to red"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "refined"

    # model echoed the old navy palette → safety net applied the requested red
    b = get_business(bid)
    copy = b.generated_copy_dict()
    assert copy["brand"] == {"primary": "#b91c1c", "secondary": "#7f1d1d"}
    html_body = s3.puts[0]["Body"]
    assert b"#b91c1c" in html_body
    assert b"#0f2a4a" not in html_body


def test_refine_applies_explicit_headline_edit(tmp_path):
    s3 = FakeS3WithDelete()
    current = {
        **VALID_COPY,
        "brand": {"primary": "#0f2a4a", "secondary": "#4a7fa8"},
    }
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(current)], s3=s3
    )  # the model returns the baseline unchanged — quoted value must win
    bid = seed_business(
        generated_copy=json.dumps(current),
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
    )

    r = client.post(
        f"/refine/{bid}",
        json={"prompt": 'make the headline "Drain Problems? We\'re On It."'},
    )
    assert r.status_code == 200
    b = get_business(bid)
    copy = b.generated_copy_dict()
    assert copy["hero_headline"] == "Drain Problems? We're On It."
    html_body = s3.puts[0]["Body"]
    assert b"Drain Problems?" in html_body


def test_refine_llm_failure_502(tmp_path):
    client, _ = make_test_client(
        tmp_path, llm_payloads=["not json at all", "still not json"]
    )
    bid = seed_business(generated_copy=json.dumps(VALID_COPY))
    r = client.post(f"/refine/{bid}", json={"prompt": "anything"})
    assert r.status_code == 502


def test_refine_unknown_business_404(tmp_path):
    client, _ = make_test_client(
        tmp_path, llm_payloads=[json.dumps(VALID_COPY)]
    )
    assert client.post("/refine/999", json={"prompt": "x"}).status_code == 404


# ── DELETE /site/{business_id} — remove deployed site + reset state ───────────


def test_delete_site_removes_objects_and_resets_state(tmp_path):
    s3 = FakeS3WithDelete(
        keys=[
            "acme-plumbing/index.html",
            "acme-plumbing/assets/logo.png",
            "other-site/index.html",  # different slug — must never be touched
        ]
    )
    client, _ = make_test_client(tmp_path, s3=s3)
    bid = seed_business(
        generated_copy=json.dumps(VALID_COPY),
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
        stage=models.DealStage.MOCKUP_READY,
    )

    r = client.delete(f"/site/{bid}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "deleted"
    assert body["removed_objects"] == 2

    assert s3.deleted == [
        "acme-plumbing/index.html",
        "acme-plumbing/assets/logo.png",
    ]
    assert len(s3.list_calls) == 1
    assert s3.list_calls[0]["Prefix"] == "acme-plumbing/"

    b = get_business(bid)
    assert b.generated_copy is None
    assert b.preview_url is None
    assert b.stage == models.DealStage.AUDITED


def test_delete_site_without_preview_makes_no_s3_calls(tmp_path):
    s3 = FakeS3WithDelete(keys=["acme-plumbing/index.html"])
    client, _ = make_test_client(tmp_path, s3=s3)
    bid = seed_business(generated_copy=json.dumps(VALID_COPY))  # no preview_url

    r = client.delete(f"/site/{bid}")
    assert r.status_code == 200
    assert r.json() == {"status": "deleted", "removed_objects": 0}
    assert s3.list_calls == []
    assert s3.deleted == []

    b = get_business(bid)
    assert b.generated_copy is None
    assert b.preview_url is None


def test_delete_site_does_not_rollback_later_stage(tmp_path):
    s3 = FakeS3WithDelete(keys=["acme-plumbing/index.html"])
    client, _ = make_test_client(tmp_path, s3=s3)
    bid = seed_business(
        generated_copy=json.dumps(VALID_COPY),
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
        stage=models.DealStage.CONTACTED,
    )

    r = client.delete(f"/site/{bid}")
    assert r.status_code == 200
    b = get_business(bid)
    assert b.preview_url is None
    assert b.stage == models.DealStage.CONTACTED  # untouched


def test_delete_site_r2_failure_502_keeps_state(tmp_path):
    s3 = FakeS3WithDelete(keys=["acme-plumbing/index.html"], fail_list=True)
    client, _ = make_test_client(tmp_path, s3=s3)
    bid = seed_business(
        generated_copy=json.dumps(VALID_COPY),
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
    )

    r = client.delete(f"/site/{bid}")
    assert r.status_code == 502
    b = get_business(bid)
    # DB state untouched when the bucket cleanup fails — retry is safe
    assert b.generated_copy is not None
    assert b.preview_url is not None


def test_delete_unknown_business_404(tmp_path):
    client, _ = make_test_client(tmp_path)
    assert client.delete("/site/999").status_code == 404


# ── POST /outreach/send/{business_id} ───────────────────────────────────────


def test_outreach_send_advances_stage(tmp_path):
    sent = []

    def fake_sender(to_email, subject, body_text, business_slug):
        sent.append(
            dict(
                to_email=to_email,
                subject=subject,
                body_text=body_text,
                business_slug=business_slug,
            )
        )
        return {"status": "sent", "id": "email_123"}

    client, _ = make_test_client(tmp_path, sender=fake_sender)
    bid = seed_business()
    r = client.post(
        f"/outreach/send/{bid}", json={"subject": "Hi there", "body": "Hello!"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "contacted"
    assert body["id"] == "email_123"
    assert sent == [
        {
            "to_email": "owner@acme.com",
            "subject": "Hi there",
            "body_text": "Hello!",
            "business_slug": "acme-plumbing",
        }
    ]
    assert get_business(bid).stage == models.DealStage.CONTACTED


def test_outreach_send_requires_email(tmp_path):
    client, _ = make_test_client(tmp_path, sender=lambda **kw: {})
    bid = seed_business(contact_email=None)
    r = client.post(f"/outreach/send/{bid}", json={"subject": "s", "body": "b"})
    assert r.status_code == 400


# ── GET /discover + POST /api/discover ───────────────────────────────────────


def _fake_places():
    return [
        {
            "id": "pid_web",
            "displayName": {"text": "Acme Plumbing"},
            "primaryTypeDisplayName": {"text": "Plumber"},
            "formattedAddress": "123 Main St, Eugene, OR 97401",
            "nationalPhoneNumber": "+1 541-555-0100",
            "websiteUri": "https://www.acmepiping.com",
        },
        {
            "id": "pid_noweb",
            "displayName": {"text": "Beta Electric"},
            "primaryTypeDisplayName": {"text": "Electrician"},
            "formattedAddress": "456 Oak Ave, Eugene, OR 97401",
            "nationalPhoneNumber": "+1 541-555-0101",
        },
    ]


def test_discover_page_renders(tmp_path):
    client, _ = make_test_client(tmp_path)
    r = client.get("/discover")
    assert r.status_code == 200
    assert "Discover New Leads" in r.text
    assert "/api/discover" in r.text


def test_api_discover_ingests_new_leads(tmp_path, monkeypatch):
    calls = []

    def fake_discover(**kw):
        calls.append(kw)
        return _fake_places()

    monkeypatch.setattr(pipeline.scanner, "discover_places", fake_discover)
    client, _ = make_test_client(tmp_path)
    r = client.post(
        "/api/discover", json={"query": "plumbers in Eugene OR", "limit": 10}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["found"] == 2
    assert [lead["slug"] for lead in body["new_leads"]] == [
        "acme-plumbing",
        "beta-electric",
    ]
    web_lead = body["new_leads"][0]
    assert web_lead["name"] == "Acme Plumbing"
    assert web_lead["website"] == "https://www.acmepiping.com"
    assert body["new_leads"][1]["website"] == ""
    assert calls == [{"query": "plumbers in Eugene OR"}]

    factory = models.get_session_factory()
    with factory() as db:
        assert db.query(models.Business).count() == 2


def test_api_discover_dedupes_on_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline.scanner, "discover_places", lambda **kw: _fake_places()
    )
    client, _ = make_test_client(tmp_path)
    first = client.post("/api/discover", json={"query": "plumbers"}).json()
    second = client.post("/api/discover", json={"query": "plumbers"}).json()
    assert first["found"] == 2 and len(first["new_leads"]) == 2
    assert second["found"] == 2 and second["new_leads"] == []


def test_api_discover_empty_query_400(tmp_path):
    client, _ = make_test_client(tmp_path)
    assert client.post("/api/discover", json={"query": "   "}).status_code == 400


def test_api_discover_missing_key_503(tmp_path, monkeypatch):
    def no_key(**kw):
        raise ValueError("PLACES_API_KEY is not set")

    monkeypatch.setattr(pipeline.scanner, "discover_places", no_key)
    client, _ = make_test_client(tmp_path)
    r = client.post("/api/discover", json={"query": "plumbers"})
    assert r.status_code == 503
    assert "PLACES_API_KEY" in r.json()["detail"]


def test_api_discover_network_failure_502(tmp_path, monkeypatch):
    import requests

    def boom(**kw):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(pipeline.scanner, "discover_places", boom)
    client, _ = make_test_client(tmp_path)
    r = client.post("/api/discover", json={"query": "plumbers"})
    assert r.status_code == 502


# ── POST /api/audit + POST /api/deploy ───────────────────────────────────────


def test_api_audit_audits_pending_leads(tmp_path, monkeypatch):
    calls = []

    def fake_audit(url, http_session=None):
        calls.append(url)
        return {
            "reachable": True,
            "flags": ["No Mobile Responsive Design"],
            "scraped_email": "found@gamma.com",
        }

    monkeypatch.setattr(pipeline.scanner, "audit_url", fake_audit)
    client, _ = make_test_client(tmp_path)
    seed_business(name="Gamma Plumber", slug="gamma-plumber",
                  current_website="https://www.gammaplumbing.com",
                  contact_email=None)
    # No website → not picked up by audit_pending
    seed_business(name="Delta Dents", slug="delta-dents",
                  current_website=None, no_website=True)

    r = client.post("/api/audit")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert calls == ["https://www.gammaplumbing.com"]
    entry = body["audited"][0]
    assert entry["slug"] == "gamma-plumber"
    assert entry["is_bad_site"] is True
    assert entry["flags"] == ["No Mobile Responsive Design"]

    b = get_business(entry["id"])
    assert b.stage == models.DealStage.AUDITED
    assert b.contact_email == "found@gamma.com"


def test_api_audit_nothing_pending(tmp_path):
    client, _ = make_test_client(tmp_path)
    seed_business(stage=models.DealStage.WON,
                  current_website="https://www.won.com")
    r = client.post("/api/audit")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0 and body["audited"] == []


def test_api_deploy_one_requires_copy(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business(generated_copy=None)
    r = client.post(f"/api/deploy/{bid}")
    assert r.status_code == 400
    assert "Generate" in r.json()["detail"]


def test_api_deploy_one_404(tmp_path):
    client, _ = make_test_client(tmp_path)
    assert client.post("/api/deploy/999").status_code == 404


def test_api_deploy_one_success(tmp_path, monkeypatch):
    uploads = []

    def fake_deploy(slug, html, s3_client=None):
        uploads.append((slug, len(html)))
        return f"https://preview.example/{slug}/"

    monkeypatch.setattr(pipeline, "deploy_to_r2", fake_deploy)
    client, _ = make_test_client(tmp_path)
    bid = seed_business(generated_copy=json.dumps(VALID_COPY))
    r = client.post(f"/api/deploy/{bid}")
    assert r.status_code == 200
    body = r.json()
    assert body["preview_url"] == "https://preview.example/acme-plumbing/"
    assert uploads and uploads[0][0] == "acme-plumbing"

    b = get_business(bid)
    assert b.preview_url == "https://preview.example/acme-plumbing/"


def test_api_deploy_bulk_skips_leads_without_copy(tmp_path, monkeypatch):
    def fake_deploy(slug, html, s3_client=None):
        return f"https://preview.example/{slug}/"

    monkeypatch.setattr(pipeline, "deploy_to_r2", fake_deploy)
    client, _ = make_test_client(tmp_path)
    seed_business(generated_copy=json.dumps(VALID_COPY))  # acme-plumbing
    seed_business(name="Beta HVAC", slug="beta-hvac")  # no copy

    r = client.post("/api/deploy", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["deployed"] == 1
    by_slug = {row["slug"]: row for row in body["results"]}
    assert by_slug["acme-plumbing"]["preview_url"] == "https://preview.example/acme-plumbing/"
    assert by_slug["beta-hvac"]["preview_url"] is None


def test_api_deploy_bulk_id_filter(tmp_path, monkeypatch):
    def fake_deploy(slug, html, s3_client=None):
        return f"https://preview.example/{slug}/"

    monkeypatch.setattr(pipeline, "deploy_to_r2", fake_deploy)
    client, _ = make_test_client(tmp_path)
    a = seed_business(generated_copy=json.dumps(VALID_COPY))  # acme-plumbing
    seed_business(name="Beta HVAC", slug="beta-hvac",
                  generated_copy=json.dumps(VALID_COPY))

    r = client.post("/api/deploy", json={"ids": [a]})
    assert r.status_code == 200
    body = r.json()
    assert body["deployed"] == 1
    assert [row["slug"] for row in body["results"]] == ["acme-plumbing"]


def test_api_deploy_r2_not_configured_503(tmp_path, monkeypatch):
    def no_r2(slug, html, s3_client=None):
        raise ValueError("R2 credentials not configured")

    monkeypatch.setattr(pipeline, "deploy_to_r2", no_r2)
    client, _ = make_test_client(tmp_path)
    bid = seed_business(generated_copy=json.dumps(VALID_COPY))
    r = client.post(f"/api/deploy/{bid}")
    assert r.status_code == 503
    assert "R2" in r.json()["detail"]


def test_api_deploy_bulk_r2_not_configured_503(tmp_path, monkeypatch):
    def no_r2(slug, html, s3_client=None):
        raise ValueError("R2 credentials not configured")

    monkeypatch.setattr(pipeline, "deploy_to_r2", no_r2)
    client, _ = make_test_client(tmp_path)
    seed_business(generated_copy=json.dumps(VALID_COPY))
    r = client.post("/api/deploy", json={})
    assert r.status_code == 503
    assert "R2" in r.json()["detail"]


# ── chunk 4 — DB-backed outreach queue + approval UI ────────────────────────


def seed_queued_email(business_id, recipient="owner@acme.com", subject=None, body=None):
    """Insert a queued OutreachEmail row and return its id."""
    factory = models.get_session_factory()
    with factory() as db:
        row = models.OutreachEmail(
            business_id=business_id,
            recipient=recipient,
            subject=subject or "Your website is costing you customers",
            body=body or "Hi — we noticed your site could use a refresh.",
            status="queued",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id


def get_outreach_row(row_id):
    factory = models.get_session_factory()
    with factory() as db:
        return db.get(models.OutreachEmail, row_id)


def test_outreach_page_lists_queued_rows(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business(is_bad_site=True, preview_url="https://preview.example/acme-plumbing/")
    seed_queued_email(bid)

    r = client.get("/outreach")
    assert r.status_code == 200
    html = r.text
    assert "Outreach Queue" in html
    assert "owner@acme.com" in html
    assert "Your website is costing you customers" in html
    assert "Acme Plumbing" in html
    # Nav link is present on every page.
    assert 'href="/outreach"' in client.get("/").text


def test_api_outreach_generate_queues_candidates(tmp_path, monkeypatch):
    def fake_pitch(name, flags, preview_url):
        return {"subject": f"Fixing {name}'s site", "body": "Pitch body text."}

    monkeypatch.setattr(pipeline, "generate_pitch_email", fake_pitch)
    client, _ = make_test_client(tmp_path)
    seed_business(
        is_bad_site=True,
        preview_url="https://preview.example/acme-plumbing/",
        stage=models.DealStage.MOCKUP_READY,
    )

    r = client.post("/api/outreach/generate")
    assert r.status_code == 200
    body = r.json()
    assert body["queued"] == 1
    assert body["skipped"] == []
    assert body["failed"] == []

    rows = models.get_session_factory()()
    with rows as db:
        stored = db.query(models.OutreachEmail).all()
    assert len(stored) == 1
    assert stored[0].status == "queued"
    assert stored[0].subject == "Fixing Acme Plumbing's site"
    assert stored[0].body == "Pitch body text."

    # Re-running is idempotent: the pending row blocks a duplicate.
    r2 = client.post("/api/outreach/generate")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["queued"] == 0
    assert body2["skipped"] == [{"slug": "acme-plumbing", "reason": "already queued"}]


def test_api_outreach_generate_no_candidates(tmp_path, monkeypatch):
    def fake_pitch(name, flags, preview_url):
        raise AssertionError("pitch generation should not run without candidates")

    monkeypatch.setattr(pipeline, "generate_pitch_email", fake_pitch)
    client, _ = make_test_client(tmp_path)
    # Healthy site → not a candidate.
    seed_business(is_bad_site=False, preview_url="https://preview.example/acme-plumbing/")
    r = client.post("/api/outreach/generate")
    assert r.status_code == 200
    assert r.json()["queued"] == 0


def test_api_outreach_send_selected_sends(tmp_path):
    calls = []

    def fake_sender(to_email, subject, body_text, business_slug, **kw):
        calls.append({"to": to_email, "subject": subject, "slug": business_slug,
                      "ledger_row_id": kw.get("ledger_row").id if kw.get("ledger_row") else None})
        return {"status": "sent", "message_id": "m-1"}

    client, _ = make_test_client(tmp_path, sender=fake_sender)
    bid = seed_business(stage=models.DealStage.MOCKUP_READY)
    row_id = seed_queued_email(bid)

    r = client.post("/api/outreach/send-selected", json={"ids": [row_id]})
    assert r.status_code == 200
    body = r.json()
    assert body["sent"] == ["acme-plumbing"]
    assert body["skipped"] == []
    assert body["rate_limited"] is False

    row = get_outreach_row(row_id)
    assert row.status == "sent"
    assert row.sent_at is not None
    b = get_business(bid)
    assert b.stage == models.DealStage.CONTACTED
    assert calls[0]["ledger_row_id"] == row_id


def test_api_outreach_send_selected_skips_non_queued_and_missing(tmp_path):
    def fake_sender(to_email, subject, body_text, business_slug, **kw):
        raise AssertionError("sender should not be called for non-queued rows")

    client, _ = make_test_client(tmp_path, sender=fake_sender)
    bid = seed_business()
    row_id = seed_queued_email(bid)
    factory = models.get_session_factory()
    with factory() as db:
        r2 = db.get(models.OutreachEmail, row_id)
        r2.status = "sent"
        db.commit()

    r = client.post("/api/outreach/send-selected", json={"ids": [row_id, 999]})
    assert r.status_code == 200
    body = r.json()
    assert body["sent"] == []
    reasons = {s["id"]: s["reason"] for s in body["skipped"]}
    assert reasons[999] == "not found"
    assert "already sent" in reasons[row_id]


def test_api_outreach_send_selected_rate_limit_stops_batch(tmp_path):
    from app.outreach import OutreachRateLimitError

    def fake_sender(to_email, subject, body_text, business_slug, **kw):
        raise OutreachRateLimitError("hourly rate limit (5/hr) exceeded")

    client, _ = make_test_client(tmp_path, sender=fake_sender)
    bid = seed_business()
    row_id = seed_queued_email(bid)

    r = client.post("/api/outreach/send-selected", json={"ids": [row_id]})
    assert r.status_code == 200
    body = r.json()
    assert body["sent"] == []
    assert body["rate_limited"] is True

    row = get_outreach_row(row_id)
    assert row.status == "queued"  # untouched by the rate-limit stop


def test_api_outreach_send_selected_dry_run_keeps_queued(tmp_path):
    def fake_sender(to_email, subject, body_text, business_slug, **kw):
        return {"status": "dry_run"}

    client, _ = make_test_client(tmp_path, sender=fake_sender)
    bid = seed_business(stage=models.DealStage.MOCKUP_READY)
    row_id = seed_queued_email(bid)

    r = client.post("/api/outreach/send-selected", json={"ids": [row_id]})
    assert r.status_code == 200
    body = r.json()
    assert body["sent"] == []
    assert body["dry_run"] == ["owner@acme.com"]

    row = get_outreach_row(row_id)
    assert row.status == "queued"
    b = get_business(bid)
    assert b.stage == models.DealStage.MOCKUP_READY  # no stage change without a real send


# ── GET /pipeline + POST /api/pipeline/run ───────────────────────────────────


def _fake_pipeline_env(monkeypatch):
    """Fake discovery, audit, LLM copy, R2 deploy, and pitch generation."""
    monkeypatch.setattr(
        pipeline.scanner, "discover_places", lambda **kw: _fake_places()
    )

    def fake_audit(url, http_session=None):
        return {
            "reachable": True,
            "flags": ["No Mobile Responsive Design"],
            "scraped_email": "owner@acmepiping.com",
        }

    monkeypatch.setattr(pipeline.scanner, "audit_url", fake_audit)
    monkeypatch.setattr(
        pipeline, "generate_landing_copy", lambda *a, **kw: dict(VALID_COPY)
    )
    monkeypatch.setattr(
        pipeline,
        "deploy_to_r2",
        lambda slug, html, s3_client=None: f"https://preview.example/{slug}/",
    )
    monkeypatch.setattr(
        pipeline,
        "generate_pitch_email",
        lambda *a, **kw: {"subject": "Hi there", "body": "Pitch body."},
    )


def test_pipeline_page_renders(tmp_path):
    client, _ = make_test_client(tmp_path)
    r = client.get("/pipeline")
    assert r.status_code == 200
    assert "Run the Full Pipeline" in r.text
    assert "/api/pipeline/run" in r.text
    # Nav link present on every page (checked via the dashboard shell)
    home = client.get("/")
    assert home.text.count('href="/pipeline"') == 1


def test_api_pipeline_run_end_to_end(tmp_path, monkeypatch):
    _fake_pipeline_env(monkeypatch)
    client, _ = make_test_client(tmp_path)
    r = client.post(
        "/api/pipeline/run",
        json={"query": "plumbers in Eugene OR", "limit": 10, "auto_deploy": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["found"] == 2
    assert [lead["slug"] for lead in body["new_leads"]] == [
        "acme-plumbing",
        "beta-electric",
    ]
    assert all(lead["id"] for lead in body["new_leads"])
    assert body["audited"] == 1  # only acme has a website to audit
    assert sorted(body["generated"]) == ["acme-plumbing", "beta-electric"]
    assert body["copy_failed"] == []
    assert {d["slug"] for d in body["deployed"]} == {
        "acme-plumbing",
        "beta-electric",
    }
    assert all(d["url"].startswith("https://preview.example/") for d in body["deployed"])
    assert body["deploy_failed"] == []
    # Only acme ends up with both a preview URL and a contact email
    assert body["outreach_queued"] == 1
    assert body["outreach_skipped"] == []

    factory = models.get_session_factory()
    with factory() as db:
        acme = db.query(models.Business).filter_by(slug="acme-plumbing").one()
        beta = db.query(models.Business).filter_by(slug="beta-electric").one()
        assert acme.stage == models.DealStage.MOCKUP_READY
        assert beta.stage == models.DealStage.MOCKUP_READY
        assert acme.preview_url == "https://preview.example/acme-plumbing/"
        assert beta.preview_url == "https://preview.example/beta-electric/"
        assert beta.contact_email is None
        rows = db.query(models.OutreachEmail).all()
        assert len(rows) == 1
        assert rows[0].status == "queued"
        assert rows[0].recipient == "owner@acmepiping.com"


def test_api_pipeline_run_without_auto_deploy(tmp_path, monkeypatch):
    _fake_pipeline_env(monkeypatch)
    client, _ = make_test_client(tmp_path)
    r = client.post("/api/pipeline/run", json={"query": "plumbers"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] == 2
    assert sorted(body["generated"]) == ["acme-plumbing", "beta-electric"]
    assert body["deployed"] == []
    # No preview URLs → nothing eligible for outreach queueing
    assert body["outreach_queued"] == 0

    factory = models.get_session_factory()
    with factory() as db:
        assert db.query(models.OutreachEmail).count() == 0


def test_api_pipeline_run_empty_query_400(tmp_path):
    client, _ = make_test_client(tmp_path)
    assert client.post("/api/pipeline/run", json={"query": "   "}).status_code == 400


def test_api_pipeline_run_copy_failure_is_per_lead(tmp_path, monkeypatch):
    from app.llm_engine import LLMGenerationError

    _fake_pipeline_env(monkeypatch)

    def flaky_copy(name, *args, **kw):
        if name == "Beta Electric":
            raise LLMGenerationError("LLM request failed: connection refused")
        return dict(VALID_COPY)

    monkeypatch.setattr(pipeline, "generate_landing_copy", flaky_copy)
    client, _ = make_test_client(tmp_path)
    r = client.post(
        "/api/pipeline/run", json={"query": "plumbers", "auto_deploy": True}
    )
    assert r.status_code == 200
    body = r.json()
    # One lead's LLM failure must not abort the run
    assert body["generated"] == ["acme-plumbing"]
    assert [f["slug"] for f in body["copy_failed"]] == ["beta-electric"]
    assert "LLM request failed" in body["copy_failed"][0]["error"]
    # Deploy only touches leads that actually have copy
    assert {d["slug"] for d in body["deployed"]} == {"acme-plumbing"}
    assert body["outreach_queued"] == 1


# ── regenerate feature: site reference + asset upload ────────────────────────

REF = {
    "source_url": "https://www.acmeplumbing.com",
    "headline": "Eugene's Trusted Plumbers Since 1998",
    "meta_description": "24/7 emergency plumbing in Eugene, OR.",
    "service_headings": ["Emergency Repairs", "Drain Cleaning"],
    "body_text": "We fix what others can't. Licensed and insured since 1998.",
    "logo_url": "https://www.acmeplumbing.com/img/acme-logo.png",
    "hero_image_url": "https://www.acmeplumbing.com/img/og-plumb.jpg",
    "about_images": ["https://www.acmeplumbing.com/img/team-work.jpg"],
}


def test_generate_with_site_reference_uploads_assets(tmp_path, monkeypatch):
    import app.main as main_mod

    s3 = FakeS3()
    client, app = make_test_client(tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3)
    bid = seed_business(current_website="https://www.acmeplumbing.com")

    monkeypatch.setattr(main_mod, "scrape_site_reference", lambda url, http_session=None: dict(REF))
    monkeypatch.setattr(
        main_mod, "fetch_image_bytes",
        lambda url, http_session=None, max_bytes=1_500_000: (b"img-bytes", "image/png"),
    )

    r = client.post(f"/generate/{bid}")
    assert r.status_code == 200
    assert r.json()["status"] == "generated"

    # LLM prompt carried the scraped reference as its factual basis
    calls = app.state.llm_client.chat.completions.calls
    user_msgs = [m for c in calls for m in c["messages"] if m["role"] == "user"]
    assert any("Eugene's Trusted Plumbers Since 1998" in m["content"] for m in user_msgs)

    # logo, hero, about + the page itself — assets before index.html
    keys = [p["Key"] for p in s3.puts]
    assert keys == [
        "acme-plumbing/assets/logo.png",
        "acme-plumbing/assets/hero.jpg",
        "acme-plumbing/assets/about.jpg",
        "acme-plumbing/index.html",
    ]

    # persisted copy stores relative asset paths; rendered HTML references them
    b = get_business(bid)
    copy = b.generated_copy_dict()
    assert copy["logo_url"] == "assets/logo.png"
    assert copy["hero_image_url"] == "assets/hero.jpg"
    assert copy["about_images"] == ["assets/about.jpg"]
    html_body = s3.puts[-1]["Body"]
    assert b'src="assets/logo.png"' in html_body
    assert b'src="assets/hero.jpg"' in html_body


def test_generate_scrape_failure_still_generates_page(tmp_path, monkeypatch):
    import app.main as main_mod

    s3 = FakeS3()
    client, _ = make_test_client(tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3)
    bid = seed_business(current_website="https://down.example.com")

    def boom(url, http_session=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod, "scrape_site_reference", boom)

    r = client.post(f"/generate/{bid}")
    assert r.status_code == 200
    assert r.json()["status"] == "generated"
    # no assets to upload — only the page itself
    assert [p["Key"] for p in s3.puts] == ["acme-plumbing/index.html"]


def test_regenerate_prompt_also_uses_site_reference(tmp_path, monkeypatch):
    import app.main as main_mod

    s3 = FakeS3()
    client, app = make_test_client(tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3)
    bid = seed_business(current_website="https://www.acmeplumbing.com")

    monkeypatch.setattr(main_mod, "scrape_site_reference", lambda url, http_session=None: dict(REF))
    monkeypatch.setattr(
        main_mod, "fetch_image_bytes",
        lambda url, http_session=None, max_bytes=1_500_000: (b"img-bytes", "image/png"),
    )

    r = client.post(f"/regenerate-prompt/{bid}", json={"prompt": "make it bolder"})
    assert r.status_code == 200
    assert r.json()["status"] == "regenerated"

    calls = app.state.llm_client.chat.completions.calls
    user_msgs = [m for c in calls for m in c["messages"] if m["role"] == "user"]
    assert any("make it bolder" in m["content"] for m in user_msgs)
    assert any("Eugene's Trusted Plumbers Since 1998" in m["content"] for m in user_msgs)


# ── review / dashboard regenerate UI ─────────────────────────────────────────


def test_review_page_with_preview_shows_regenerate_button(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business(preview_url="https://preview.solospark.net/acme-plumbing/index.html")

    r = client.get(f"/review/{bid}")
    assert r.status_code == 200
    assert 'id="regenerate-btn"' in r.text
    assert "Regenerate site" in r.text
    assert 'id="regen-status"' in r.text
    assert f'/generate/{bid}' in r.text  # fetch target in the page script


def test_review_page_without_preview_shows_generate_button(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business()

    r = client.get(f"/review/{bid}")
    assert r.status_code == 200
    assert "Generate mockup" in r.text
    assert ".generate-btn" in r.text  # fetch handler bound to the button
    assert "Regenerate site" not in r.text


def test_dashboard_labels_generate_vs_regenerate(tmp_path):
    client, _ = make_test_client(tmp_path)
    seed_business()  # no preview → "Generate"
    seed_business(name="Beta HVAC", slug="beta-hvac",
                  preview_url="https://preview.example/beta-hvac/")

    r = client.get("/")
    assert ">Generate<" in r.text
    assert ">Regenerate<" in r.text


def test_review_page_with_preview_shows_refine_checkbox_and_delete_button(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business(
        generated_copy=json.dumps(VALID_COPY),
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
    )

    r = client.get(f"/review/{bid}")
    assert r.status_code == 200
    # website-generation prompt form lives in the header, above the site buttons
    assert 'id="site-prompt-form"' in r.text
    assert r.text.index('id="site-prompt-form"') < r.text.index('id="regenerate-btn"')
    assert r.text.index('id="site-prompt-form"') < r.text.index('id="delete-site-btn"')
    assert 'id="refine-only"' in r.text
    assert "Update current site only (no re-scrape)" in r.text
    assert 'id="delete-site-btn"' in r.text
    assert "Delete site" in r.text
    # the original prompt form still exists in the outreach card, unmodified
    assert 'id="prompt-regen-form"' in r.text
    # fetch targets present in the page script
    assert f"/refine/{bid}" in r.text
    assert f"/site/{bid}" in r.text


def test_review_page_without_copy_has_no_refine_checkbox_or_delete(tmp_path):
    client, _ = make_test_client(tmp_path)
    bid = seed_business()

    r = client.get(f"/review/{bid}")
    assert r.status_code == 200
    # no site yet → no header website-prompt form, no refine checkbox, no delete button
    assert 'id="site-prompt-form"' not in r.text
    assert 'id="refine-only"' not in r.text
    assert "Delete site" not in r.text
    # the original prompt form is unchanged in the outreach card
    assert 'id="prompt-regen-form"' in r.text
    assert ">Regenerate<" in r.text


# ── Gallery image upload (content_images) ────────────────────────────────────

REF_WITH_GALLERY = {
    **REF,
    "content_images": [
        "https://www.acmeplumbing.com/img/og-plumb.jpg",   # == hero → skipped
        "https://www.acmeplumbing.com/img/team-work.jpg",  # == about[0] → skipped
        "https://www.acmeplumbing.com/img/job-site-1.jpg",
        "https://www.acmeplumbing.com/img/job-site-2.jpg",
    ],
}


def test_generate_with_gallery_content_images_uploads_them(tmp_path, monkeypatch):
    import app.main as main_mod

    s3 = FakeS3()
    client, _ = make_test_client(tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3)
    bid = seed_business(current_website="https://www.acmeplumbing.com")

    monkeypatch.setattr(
        main_mod, "scrape_site_reference",
        lambda url, http_session=None: dict(REF_WITH_GALLERY),
    )
    monkeypatch.setattr(
        main_mod, "fetch_image_bytes",
        lambda url, http_session=None, max_bytes=1_500_000: (b"img-bytes", "image/png"),
    )

    r = client.post(f"/generate/{bid}")
    assert r.status_code == 200
    assert r.json()["status"] == "generated"

    # logo/hero/about as before, plus up to 3 gallery photos (deduped), then the page
    keys = [p["Key"] for p in s3.puts]
    assert keys == [
        "acme-plumbing/assets/logo.png",
        "acme-plumbing/assets/hero.jpg",
        "acme-plumbing/assets/about.jpg",
        "acme-plumbing/assets/gallery-1.jpg",
        "acme-plumbing/assets/gallery-2.jpg",
        "acme-plumbing/index.html",
    ]

    copy = get_business(bid).generated_copy_dict()
    assert copy["gallery_images"] == ["assets/gallery-1.jpg", "assets/gallery-2.jpg"]


def test_generate_carries_over_old_gallery_when_fresh_scrape_has_none(tmp_path, monkeypatch):
    import app.main as main_mod

    s3 = FakeS3()
    client, _ = make_test_client(tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3)
    bid = seed_business(
        current_website="https://www.acmeplumbing.com",
        generated_copy=json.dumps({**VALID_COPY, "gallery_images": ["assets/old-gallery.jpg"]}),
    )

    # REF has no content_images → nothing fresh to upload
    monkeypatch.setattr(main_mod, "scrape_site_reference", lambda url, http_session=None: dict(REF))
    monkeypatch.setattr(
        main_mod, "fetch_image_bytes",
        lambda url, http_session=None, max_bytes=1_500_000: (b"img-bytes", "image/png"),
    )

    r = client.post(f"/generate/{bid}")
    assert r.status_code == 200

    # the previously stored gallery survives regeneration
    copy = get_business(bid).generated_copy_dict()
    assert copy["gallery_images"] == ["assets/old-gallery.jpg"]


# ── pre-save syntax QA gate ──────────────────────────────────────────────────


def test_generate_syntax_check_fixes_copy_before_save_and_upload(tmp_path):
    broken = {**VALID_COPY, "hero_headline": "Best {city} plumbers in town"}
    fixed = {**broken, "hero_headline": "Best Eugene plumbers in town"}
    s3 = FakeS3()
    client, app = make_test_client(
        tmp_path, llm_payloads=[json.dumps(broken), json.dumps(fixed)], s3=s3
    )
    bid = seed_business()

    r = client.post(f"/generate/{bid}")
    assert r.status_code == 200
    assert r.json()["status"] == "generated"

    # the QA pass made a second LLM call, and its fix is what got saved + rendered
    calls = app.state.llm_client.chat.completions.calls
    assert len(calls) == 2
    b = get_business(bid)
    assert b.generated_copy_dict()["hero_headline"] == "Best Eugene plumbers in town"
    html_body = s3.puts[-1]["Body"]
    assert b"Best Eugene plumbers in town" in html_body


def test_generate_syntax_check_failure_keeps_original_copy(tmp_path):
    s3 = FakeS3()
    client, _ = make_test_client(
        tmp_path,
        llm_payloads=[json.dumps(VALID_COPY), "not json", "still not json"],
        s3=s3,
    )
    bid = seed_business()

    r = client.post(f"/generate/{bid}")
    # the QA pass failed — but the generation itself must still succeed
    assert r.status_code == 200
    b = get_business(bid)
    assert b.generated_copy_dict()["hero_headline"] == VALID_COPY["hero_headline"]
    assert [p["Key"] for p in s3.puts] == ["acme-plumbing/index.html"]


def test_generate_blocks_upload_when_html_fails_syntax_check(tmp_path, monkeypatch):
    import app.main as main_mod

    s3 = FakeS3()
    client, _ = make_test_client(tmp_path, llm_payloads=[json.dumps(VALID_COPY)], s3=s3)
    bid = seed_business()
    monkeypatch.setattr(
        main_mod, "render_site_html",
        lambda business, copy: "<html><body><div>broken</body></html>",
    )

    r = client.post(f"/generate/{bid}")
    assert r.status_code == 500
    assert "syntax check" in r.json()["detail"]
    # nothing was uploaded — the broken page never reached R2
    assert s3.puts == []


def test_refine_runs_syntax_check_before_upload(tmp_path):
    import app.main as main_mod

    base = get_settings().r2_public_base_url.rstrip("/")
    current = {**VALID_COPY, "hero_headline": "Old headline with {city} left in it"}
    fixed = {**current, "hero_headline": "New clean headline"}
    s3 = FakeS3()
    client, app = make_test_client(
        tmp_path, llm_payloads=[json.dumps(current), json.dumps(fixed)], s3=s3
    )
    bid = seed_business(generated_copy=json.dumps(current))

    r = client.post(f"/refine/{bid}", json={"prompt": "clean up the headline"})
    assert r.status_code == 200
    assert r.json()["status"] == "refined"
    assert r.json()["preview_url"] == f"{base}/acme-plumbing/index.html"

    calls = app.state.llm_client.chat.completions.calls
    assert len(calls) == 2  # refine + syntax QA pass
    b = get_business(bid)
    assert b.generated_copy_dict()["hero_headline"] == "New clean headline"
