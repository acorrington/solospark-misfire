"""Tests for app.pipeline — the shared step functions behind the CLI and web UI.

These exercise the data contracts directly (no argparse, no stdout): every
external service is faked by patching the pipeline module's own imports, so
the suite runs fully offline.
"""

from __future__ import annotations

import pytest

import app.models as models
import app.pipeline as pipeline
from app.outreach import OutreachRateLimitError
from app.utils import city_from_address

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


@pytest.fixture()
def db():
    factory = models.get_session_factory()
    session = factory()
    yield session
    session.close()


# ── helpers ──────────────────────────────────────────────────────────────────


def seed(db, **overrides) -> models.Business:
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
    return b


def row_by_slug(db, slug: str) -> models.Business:
    return db.query(models.Business).filter_by(slug=slug).first()


# ── discover ─────────────────────────────────────────────────────────────────


def test_discover_returns_found_and_created(db, monkeypatch):
    places = [
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
    monkeypatch.setattr(pipeline.scanner, "discover_places", lambda **kw: places)

    found, created = pipeline.discover(db, "plumbers in Eugene OR")
    assert found == 2
    assert [b.slug for b in created] == ["acme-plumbing", "beta-electric"]

    # limit slices before ingest; re-ingesting dedups to zero new leads.
    found, created = pipeline.discover(db, "plumbers in Eugene OR", limit=1)
    assert found == 1
    assert created == []


def test_discover_limit_slices_before_ingest(db, monkeypatch):
    places = [
        {
            "id": f"pid_{i}",
            "displayName": {"text": f"Shop {i}"},
            "primaryTypeDisplayName": {"text": "Plumber"},
            "formattedAddress": f"{i} Main St, Eugene, OR 97401",
        }
        for i in range(3)
    ]
    monkeypatch.setattr(pipeline.scanner, "discover_places", lambda **kw: places)

    found, created = pipeline.discover(db, "plumbers", limit=2)
    assert found == 2
    assert len(created) == 2
    assert db.query(models.Business).count() == 2


# ── audit_pending ────────────────────────────────────────────────────────────


def test_audit_pending_returns_audited(db, monkeypatch):
    seed(db, current_website="https://www.acmepiping.com")
    monkeypatch.setattr(
        pipeline.scanner,
        "audit_url",
        lambda url, http_session=None: {
            "flags": ["Missing Mobile Viewport", "Stale Copyright (2019)"],
            "scraped_email": "owner@acmepiping.com",
        },
    )

    audited = pipeline.audit_pending(db)
    assert [b.slug for b in audited] == ["acme-plumbing"]
    row = row_by_slug(db, "acme-plumbing")
    assert row.stage == models.DealStage.AUDITED
    assert row.is_bad_site is True
    assert "Missing Mobile Viewport" in row.audit_flags_list()


def test_audit_pending_empty_when_nothing_to_fetch(db):
    seed(db, no_website=True, is_bad_site=True, stage=models.DealStage.AUDITED)
    assert pipeline.audit_pending(db) == []


# ── generate_for ─────────────────────────────────────────────────────────────


def test_generate_for_persists_copy_and_stage(db, monkeypatch):
    seed(db)
    calls = []

    def fake_copy(name, category, city, **kw):
        calls.append((name, category, city))
        return COPY

    monkeypatch.setattr(pipeline, "generate_landing_copy", fake_copy)

    copy = pipeline.generate_for(db, row_by_slug(db, "acme-plumbing"))
    assert copy == COPY
    assert calls == [("Acme Plumbing", "Plumber", "Eugene")]
    row = row_by_slug(db, "acme-plumbing")
    assert row.stage == models.DealStage.MOCKUP_READY
    assert row.generated_copy_dict() == COPY


def test_generate_for_preserves_advanced_stage(db, monkeypatch):
    seed(db, stage=models.DealStage.CONTACTED)
    monkeypatch.setattr(pipeline, "generate_landing_copy", lambda *a, **kw: COPY)

    pipeline.generate_for(db, row_by_slug(db, "acme-plumbing"))
    assert row_by_slug(db, "acme-plumbing").stage == models.DealStage.CONTACTED


def test_deploy_business_blocks_upload_when_html_fails_syntax_check(db, monkeypatch):
    seed(db)
    uploads = []

    def fake_deploy(slug, html, s3_client=None):
        uploads.append((slug, html))
        return f"https://preview.solospark.dev/{slug}/index.html"

    monkeypatch.setattr(pipeline, "deploy_to_r2", fake_deploy)
    # a broken render must be caught by the gate and never uploaded
    monkeypatch.setattr(
        pipeline, "render_site_html", lambda b, c: "<html><body><div>oops</body></html>"
    )

    with pytest.raises(ValueError, match="syntax check"):
        pipeline.deploy_business(db, row_by_slug(db, "acme-plumbing"))

    assert uploads == []


# ── deploy_many ──────────────────────────────────────────────────────────────


def test_deploy_many_skips_leads_without_copy(db, monkeypatch):
    uploads = []
    monkeypatch.setattr(
        pipeline,
        "deploy_to_r2",
        lambda slug, html, s3_client=None: uploads.append(slug)
        or f"https://preview.solospark.net/{slug}/index.html",
    )

    with_copy = seed(db, place_id="p1", slug="acme-plumbing")
    with_copy.set_generated_copy(COPY)
    db.commit()
    no_copy = seed(db, place_id="p2", slug="beta-electric", name="Beta Electric")

    results = pipeline.deploy_many(db, [with_copy, no_copy])
    assert uploads == ["acme-plumbing"]
    urls = {b.slug: url for b, url in results}
    assert urls["acme-plumbing"].endswith("acme-plumbing/index.html")
    assert urls["beta-electric"] is None


# ── outreach_batch ───────────────────────────────────────────────────────────


def test_outreach_batch_dry_run_returns_rows(db, monkeypatch):
    monkeypatch.setattr(pipeline, "generate_pitch_email", lambda *a, **kw: PITCH)
    seed(
        db,
        is_bad_site=True,
        stage=models.DealStage.AUDITED,
        preview_url="https://preview.solospark.net/acme-plumbing/index.html",
        audit_flags='["Missing Mobile Viewport"]',
    )

    result = pipeline.outreach_batch(db, [row_by_slug(db, "acme-plumbing")], dry_run=True)
    assert result.counts == (1, 0, 0, 0)
    row = result.queued[0]
    assert row["business_slug"] == "acme-plumbing"
    assert row["name"] == "Acme Plumbing"
    assert row["email"] == "owner@acmepiping.com"
    assert row["subject"] == PITCH["subject"]
    assert row["body"] == PITCH["body"]
    assert row["preview_url"].endswith("acme-plumbing/index.html")
    assert row["generated_at"]  # non-empty ISO timestamp

    # Dry-run must not touch the outreach ledger.
    assert db.query(models.OutreachEmail).count() == 0


def test_outreach_batch_send_dispatches_and_marks_contacted(db, monkeypatch):
    sent = []
    monkeypatch.setattr(pipeline, "generate_pitch_email", lambda *a, **kw: PITCH)

    def fake_send(to_email, subject, body_text, business_slug, **kw):
        sent.append(business_slug)
        return {"status": "sent", "provider": "smtp"}

    monkeypatch.setattr(pipeline, "send_outreach_email", fake_send)
    seed(db, is_bad_site=True, stage=models.DealStage.AUDITED,
         preview_url="https://preview.solospark.net/acme-plumbing/index.html")

    result = pipeline.outreach_batch(db, [row_by_slug(db, "acme-plumbing")], dry_run=False)
    assert result.counts == (0, 1, 0, 0)
    assert sent == ["acme-plumbing"]
    assert row_by_slug(db, "acme-plumbing").stage == models.DealStage.CONTACTED


def test_outreach_batch_stops_on_rate_limit(db, monkeypatch):
    calls = {"n": 0}

    def flaky_send(to_email, subject, body_text, business_slug, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OutreachRateLimitError("limit")
        return {"status": "sent", "provider": "smtp"}

    monkeypatch.setattr(pipeline, "generate_pitch_email", lambda *a, **kw: PITCH)
    monkeypatch.setattr(pipeline, "send_outreach_email", flaky_send)
    seed(db, place_id="p1", slug="acme-plumbing", is_bad_site=True,
         preview_url="https://preview.solospark.net/acme-plumbing/index.html")
    seed(db, place_id="p2", slug="beta-electric", name="Beta Electric",
         is_bad_site=True, preview_url="https://preview.solospark.net/beta-electric/index.html",
         contact_email="owner@beta.com")

    result = pipeline.outreach_batch(
        db, [row_by_slug(db, "acme-plumbing"), row_by_slug(db, "beta-electric")], dry_run=False
    )
    assert result.counts == (0, 1, 0, 0)
    # The second lead must be untouched: batch stops on the rate limit.
    assert row_by_slug(db, "beta-electric").stage != models.DealStage.CONTACTED


def test_outreach_batch_skips_value_errors(db, monkeypatch):
    def refusing_send(to_email, subject, body_text, business_slug, **kw):
        raise ValueError("already sent")

    monkeypatch.setattr(pipeline, "generate_pitch_email", lambda *a, **kw: PITCH)
    monkeypatch.setattr(pipeline, "send_outreach_email", refusing_send)
    seed(db, is_bad_site=True, stage=models.DealStage.AUDITED,
         preview_url="https://preview.solospark.net/acme-plumbing/index.html")

    result = pipeline.outreach_batch(db, [row_by_slug(db, "acme-plumbing")], dry_run=False)
    assert result.counts == (0, 0, 1, 0)
    assert result.skipped[0] == ("acme-plumbing", "already sent")


# ── outreach_candidates ──────────────────────────────────────────────────────


def test_outreach_candidates_filters(db):
    seed(db, place_id="pa", slug="alpha", name="Alpha", is_bad_site=True,
         preview_url="https://preview.solospark.net/alpha/index.html", contact_email="a@x.com")
    beta = seed(db, place_id="pb", slug="beta", name="Beta", is_bad_site=True,
                preview_url="https://preview.solospark.net/beta/index.html", contact_email="b@x.com")
    db.add(models.OutreachEmail(business_id=beta.id, recipient="b@x.com", status="sent"))
    seed(db, place_id="pc", slug="gamma", name="Gamma", is_bad_site=True,
         preview_url="https://preview.solospark.net/gamma/index.html", contact_email="c@x.com",
         opted_out=True)
    seed(db, place_id="pd", slug="delta", name="Delta", is_bad_site=True,
         contact_email="d@x.com")  # no preview URL
    db.commit()

    assert [b.slug for b in pipeline.outreach_candidates(db)] == ["alpha"]


# ── city_from_address ────────────────────────────────────────────────────────


def test_city_from_address():
    assert city_from_address("123 Main St, Eugene, OR 97401") == "Eugene"
    assert city_from_address(None) == ""
    assert city_from_address("") == ""
    assert city_from_address("OnlyOnePart") == ""
