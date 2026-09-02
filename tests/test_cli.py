"""Phase 7 — tests for run.py (CLI orchestration).

All external services are faked: Google Places discovery, website auditing,
the local LLM (landing copy + pitch emails), R2 deployment, and email
dispatch. The pipeline steps are patched on ``app.pipeline`` (the shared
module behind the CLI); the CLI itself is exercised through
``run.main([...])`` against a temp SQLite ledger.
"""

from __future__ import annotations

import csv

import pytest

import run as cli
import app.models as models
import app.pipeline as pipeline
from app.llm_engine import LLMGenerationError  # noqa: F401  (imported for parity)

COPY = {
    "tagline": "Fast, honest plumbing.",
    "hero_headline": "Eugene's trusted plumbers",
    "hero_subheadline": "24/7 emergency service, upfront pricing.",
    "services": [
        {"title": "Leak Repair", "description": "Fast, permanent fixes.", "icon_name": "wrench"},
        {"title": "Drain Cleaning", "description": "Clear clogs fast.", "icon_name": "droplet"},
        {"title": "Water Heaters", "description": "Install and repair.", "icon_name": "flame"},
    ],
    "about_heading": "About Acme",
    "about_text": "Family owned since 1998. We treat every home like our own.",
    "why_choose_us": ["Licensed & insured", "Upfront pricing", "Same-day service"],
    "cta_text": "Call Now",
}

PITCH = {"subject": "Quick question about acmepiping.com", "body": "Hi — I noticed a few things..."}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in (
        "RESEND_API_KEY",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    models.reinit_db(f"sqlite:///{tmp_path}/test.db")
    yield


# ── helpers ──────────────────────────────────────────────────────────────────


def fake_places():
    return [
        {
            "id": "pid_web",
            "displayName": {"text": "Acme Plumbing"},
            "primaryTypeDisplayName": {"text": "Plumber"},
            "formattedAddress": "123 Main St, Eugene, OR 97401",
            "nationalPhoneNumber": "+1 541-555-0100",
            "websiteUri": "https://www.acmepiping.com",
            "rating": 4.6,
            "userRatingCount": 88,
        },
        {
            "id": "pid_noweb",
            "displayName": {"text": "Beta Electric"},
            "primaryTypeDisplayName": {"text": "Electrician"},
            "formattedAddress": "456 Oak Ave, Eugene, OR 97401",
            "nationalPhoneNumber": "+1 541-555-0101",
            "rating": 4.9,
            "userRatingCount": 20,
        },
        {
            "id": "pid_blocked",
            "displayName": {"text": "Gamma HVAC"},
            "primaryTypeDisplayName": {"text": "HVAC Contractor"},
            "formattedAddress": "789 Pine Rd, Eugene, OR 97401",
            "nationalPhoneNumber": "+1 541-555-0102",
            "websiteUri": "https://www.yelp.com/biz/gamma-hvac-eugene",
        },
    ]


def seed_business(**overrides) -> int:
    factory = models.get_session_factory()
    db = factory()
    defaults = dict(
        place_id="pid_1",
        name="Acme Plumbing",
        slug="acme-plumbing",
        category="Plumber",
        address="123 Main St, Eugene, OR 97401",
        phone="+1 541-555-0100",
        contact_email="owner@acmepiping.com",
    )
    defaults.update(overrides)
    b = models.Business(**defaults)
    db.add(b)
    db.commit()
    db.refresh(b)
    row_id = b.id
    db.close()
    return row_id


def store_copy(slug: str, copy: dict) -> None:
    factory = models.get_session_factory()
    db = factory()
    b = db.query(models.Business).filter_by(slug=slug).first()
    b.set_generated_copy(copy)
    db.commit()
    db.close()


def get_business(slug: str):
    factory = models.get_session_factory()
    db = factory()
    b = db.query(models.Business).filter_by(slug=slug).first()
    data = {
        "id": b.id,
        "name": b.name,
        "stage": b.stage,
        "is_bad_site": b.is_bad_site,
        "no_website": b.no_website,
        "contact_email": b.contact_email,
        "preview_url": b.preview_url,
        "audit_flags": b.audit_flags,
    }
    db.close()
    return data


def count_businesses() -> int:
    factory = models.get_session_factory()
    db = factory()
    n = db.query(models.Business).count()
    db.close()
    return n


# ── discover ─────────────────────────────────────────────────────────────────


def test_discover_ingests_and_dedups(monkeypatch):
    monkeypatch.setattr(pipeline.scanner, "discover_places", lambda **kw: fake_places())

    assert cli.main(["discover", "--query", "plumbers in Eugene OR"]) == 0
    assert count_businesses() == 3

    # Re-running the same query adds nothing (ledger dedup by place_id).
    assert cli.main(["discover", "--query", "plumbers in Eugene OR"]) == 0
    assert count_businesses() == 3


def test_discover_limit(monkeypatch):
    monkeypatch.setattr(pipeline.scanner, "discover_places", lambda **kw: fake_places())

    assert cli.main(["discover", "--query", "plumbers in Eugene OR", "--limit", "2"]) == 0
    assert count_businesses() == 2


def test_discover_marks_blocked_directory(monkeypatch):
    monkeypatch.setattr(pipeline.scanner, "discover_places", lambda **kw: fake_places())

    assert cli.main(["discover", "--query", "plumbers in Eugene OR"]) == 0
    gamma = get_business("gamma-hvac")
    assert gamma["no_website"] is True
    assert gamma["is_bad_site"] is True
    assert "blocked directory" in gamma["audit_flags"]


# ── audit ────────────────────────────────────────────────────────────────────


def test_audit_pending(monkeypatch):
    seed_business(
        current_website="https://www.acmepiping.com",
    )
    monkeypatch.setattr(
        pipeline.scanner,
        "audit_url",
        lambda url, http_session=None: {
            "flags": ["Missing Mobile Viewport", "Stale Copyright (2019)"],
            "scraped_email": "owner@acmepiping.com",
        },
    )

    assert cli.main(["audit"]) == 0
    row = get_business("acme-plumbing")
    assert row["stage"] == models.DealStage.AUDITED
    assert row["is_bad_site"] is True
    assert "Missing Mobile Viewport" in row["audit_flags"]


def test_audit_nothing_pending():
    # No-website leads are audited at ingest; nothing left to audit.
    seed_business(no_website=True, is_bad_site=True, stage=models.DealStage.AUDITED)
    assert cli.main(["audit"]) == 0


# ── generate ─────────────────────────────────────────────────────────────────


def test_generate_by_slug(monkeypatch):
    seed_business()
    calls = []

    def fake_copy(name, category, city, **kw):
        calls.append((name, category, city))
        return COPY

    monkeypatch.setattr(pipeline, "generate_landing_copy", fake_copy)

    assert cli.main(["generate", "--slug", "acme-plumbing"]) == 0
    assert calls == [("Acme Plumbing", "Plumber", "Eugene")]
    row = get_business("acme-plumbing")
    assert row["stage"] == models.DealStage.MOCKUP_READY


def test_generate_unknown_slug():
    assert cli.main(["generate", "--slug", "nope"]) == 1


# ── deploy ───────────────────────────────────────────────────────────────────


def test_deploy_all(monkeypatch):
    seed_business()
    store_copy("acme-plumbing", COPY)
    uploads = []

    def fake_deploy(slug, html, s3_client=None):
        uploads.append((slug, html))
        return f"https://preview.solospark.net/{slug}/index.html"

    monkeypatch.setattr(pipeline, "deploy_to_r2", fake_deploy)

    assert cli.main(["deploy", "--all"]) == 0
    assert len(uploads) == 1
    slug, html = uploads[0]
    assert slug == "acme-plumbing"
    assert "Eugene&#39;s trusted plumbers" in html
    assert 'href="tel:+15415550100"' in html
    row = get_business("acme-plumbing")
    assert row["preview_url"] == "https://preview.solospark.net/acme-plumbing/index.html"


def test_deploy_requires_flag():
    assert cli.main(["deploy"]) == 1


# ── outreach ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def _queue_csv(tmp_path, monkeypatch):
    path = tmp_path / "outreach_queue.csv"
    monkeypatch.setattr(cli, "OUTREACH_QUEUE_CSV", path)
    return path


def test_outreach_dry_run_writes_csv(_queue_csv, monkeypatch):
    seed_business(
        is_bad_site=True,
        stage=models.DealStage.AUDITED,
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
        audit_flags='["Missing Mobile Viewport"]',
    )
    monkeypatch.setattr(pipeline, "generate_pitch_email", lambda *a, **kw: PITCH)

    assert cli.main(["outreach"]) == 0  # dry-run is the default

    rows = list(csv.DictReader(_queue_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["business_slug"] == "acme-plumbing"
    assert row["email"] == "owner@acmepiping.com"
    assert row["subject"] == PITCH["subject"]
    assert row["preview_url"].endswith("acme-plumbing/index.html")


def test_outreach_send_dispatches(_queue_csv, monkeypatch):
    seed_business(
        is_bad_site=True,
        stage=models.DealStage.AUDITED,
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
    )
    sent = []

    def fake_send(to_email, subject, body_text, business_slug, **kw):
        sent.append(dict(to_email=to_email, subject=subject, business_slug=business_slug))
        return {"status": "sent", "provider": "smtp"}

    monkeypatch.setattr(pipeline, "generate_pitch_email", lambda *a, **kw: PITCH)
    monkeypatch.setattr(pipeline, "send_outreach_email", fake_send)

    assert cli.main(["outreach", "--send"]) == 0
    assert sent == [
        dict(to_email="owner@acmepiping.com", subject=PITCH["subject"], business_slug="acme-plumbing")
    ]
    row = get_business("acme-plumbing")
    assert row["stage"] == models.DealStage.CONTACTED


def test_outreach_skips_without_email(_queue_csv, monkeypatch):
    seed_business(
        is_bad_site=True,
        stage=models.DealStage.AUDITED,
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
        contact_email=None,
    )
    assert cli.main(["outreach"]) == 0
    assert not _queue_csv.exists()


# ── pipeline ─────────────────────────────────────────────────────────────────


def test_pipeline_end_to_end(_queue_csv, monkeypatch):
    places = fake_places()[:2]  # website lead + no-website lead
    monkeypatch.setattr(pipeline.scanner, "discover_places", lambda **kw: places)
    monkeypatch.setattr(
        pipeline.scanner,
        "audit_url",
        lambda url, http_session=None: {
            "flags": ["Missing Mobile Viewport"],
            "scraped_email": "owner@acmepiping.com",
        },
    )
    monkeypatch.setattr(pipeline, "generate_landing_copy", lambda *a, **kw: COPY)
    uploads = []

    def fake_deploy(slug, html, s3_client=None):
        uploads.append(slug)
        return f"https://preview.solospark.net/{slug}/index.html"

    monkeypatch.setattr(pipeline, "deploy_to_r2", fake_deploy)
    monkeypatch.setattr(pipeline, "generate_pitch_email", lambda *a, **kw: PITCH)

    assert cli.main(["pipeline", "--query", "plumbers in Eugene OR", "--auto-deploy"]) == 0

    acme = get_business("acme-plumbing")
    beta = get_business("beta-electric")
    # Website lead: audited (bad), copy generated, preview deployed.
    assert acme["is_bad_site"] is True
    assert acme["stage"] == models.DealStage.MOCKUP_READY
    assert acme["preview_url"].endswith("acme-plumbing/index.html")
    # No-website lead: flagged at ingest, copy generated, preview deployed.
    assert beta["no_website"] is True
    assert beta["preview_url"].endswith("beta-electric/index.html")
    assert sorted(uploads) == ["acme-plumbing", "beta-electric"]

    # Only the website lead has a contact email → exactly one queued pitch.
    rows = list(csv.DictReader(_queue_csv.open(encoding="utf-8")))
    assert [r["business_slug"] for r in rows] == ["acme-plumbing"]
