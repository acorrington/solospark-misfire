"""Phase 8 tests — Stripe webhooks, delinquency killswitch & offboarding.

Covers webhook interpretation (activation + suspension), the R2 killswitch
(maintenance page up / landing page restore), and zip export for offboarding.
All S3 access goes through an injected fake; no live services required.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import billing, models
from app.config import get_settings
from app.main import create_app


# ── fixtures / helpers ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in (
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "R2_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    models.reinit_db(f"sqlite:///{tmp_path}/test.db")
    yield
    get_settings.cache_clear()


def seed_business(**overrides):
    db = models.get_session_factory()()
    try:
        fields = {
            "place_id": "pid_1",
            "name": "Acme Plumbing",
            "slug": "acme-plumbing",
            "category": "plumber",
            "address": "42 River St, Eugene, OR",
            "phone": "(541) 555-0142",
        }
        fields.update(overrides)
        business = models.Business(**fields)
        db.add(business)
        db.commit()
        db.refresh(business)
        return business
    finally:
        db.close()


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


class FakeS3:
    """Records puts; serves list/get from an in-memory object store."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        body = kwargs["Body"]
        self.objects[kwargs["Key"]] = body if isinstance(body, (bytes, bytearray)) else bytes(body)
        return {}

    def list_objects_v2(self, Bucket=None, Prefix=None, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix or ""))
        return {
            "Contents": [{"Key": k, "Size": len(self.objects[k])} for k in keys],
            "IsTruncated": False,
        }

    def get_object(self, Bucket=None, Key=None):
        return {"Body": io.BytesIO(self.objects[Key])}


def put_body(kwargs) -> str:
    body = kwargs["Body"]
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8")
    return body.read().decode("utf-8")


def store_copy(business, copy):
    """Persist generated copy through a live session (set_generated_copy
    only mutates the attribute — no implicit commit)."""
    db = models.get_session_factory()()
    try:
        row = db.query(models.Business).filter_by(id=business.id).one()
        row.set_generated_copy(copy)
        db.commit()
    finally:
        db.close()


# ── webhook: checkout.session.completed ──────────────────────────────────────


def test_checkout_completed_activates_and_wins():
    seed_business(stripe_customer_id="cus_123")
    payload = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "metadata": {"slug": "acme-plumbing"},
                "customer": "cus_123",
                "subscription": "sub_abc",
            }
        },
    }
    result = billing.handle_webhook_event("checkout.session.completed", payload)

    assert result == {"status": "activated", "slug": "acme-plumbing"}
    db = models.get_session_factory()()
    try:
        business = db.query(models.Business).filter_by(slug="acme-plumbing").one()
        assert business.subscription_status is models.SubscriptionStatus.ACTIVE
        assert business.stage is models.DealStage.WON
        assert business.stripe_subscription_id == "sub_abc"
    finally:
        db.close()


def test_checkout_completed_lookup_via_customer_id():
    seed_business(stripe_customer_id="cus_only")
    payload = {
        "type": "checkout.session.completed",
        "data": {"object": {"customer": "cus_only"}},
    }
    result = billing.handle_webhook_event("checkout.session.completed", payload)
    assert result["status"] == "activated"


# ── webhook: suspension events ───────────────────────────────────────────────


def test_payment_failed_suspends_site():
    seed_business()
    s3 = FakeS3()
    payload = {
        "type": "invoice.payment_failed",
        "data": {"object": {"metadata": {"slug": "acme-plumbing"}}},
    }
    result = billing.handle_webhook_event("invoice.payment_failed", payload, s3_client=s3)

    assert result == {"status": "suspended", "slug": "acme-plumbing"}
    db = models.get_session_factory()()
    try:
        business = db.query(models.Business).filter_by(slug="acme-plumbing").one()
        assert business.subscription_status is models.SubscriptionStatus.PAST_DUE
    finally:
        db.close()
    assert len(s3.puts) == 1
    put = s3.puts[0]
    assert put["Key"] == "acme-plumbing/index.html"
    html = put_body(put)
    assert "Under Scheduled Maintenance" in html
    assert "Acme Plumbing is temporarily offline" in html
    assert 'href="tel:+5415550142"' in html


def test_subscription_deleted_lookup_via_customer_id():
    seed_business(stripe_customer_id="cus_999")
    s3 = FakeS3()
    payload = {
        "type": "customer.subscription.deleted",
        "data": {"object": {"customer": "cus_999"}},
    }
    result = billing.handle_webhook_event(
        "customer.subscription.deleted", payload, s3_client=s3
    )
    assert result["status"] == "suspended"
    db = models.get_session_factory()()
    try:
        business = db.query(models.Business).filter_by(slug="acme-plumbing").one()
        assert business.subscription_status is models.SubscriptionStatus.PAST_DUE
    finally:
        db.close()


def test_unknown_event_type_raises():
    seed_business()
    with pytest.raises(ValueError, match="Unsupported webhook event type"):
        billing.handle_webhook_event("charge.refunded", {"data": {"object": {}}})


def test_unknown_business_raises_lookup_error():
    s3 = FakeS3()
    payload = {
        "type": "invoice.payment_failed",
        "data": {"object": {"metadata": {"slug": "ghost-business"}}},
    }
    with pytest.raises(LookupError):
        billing.handle_webhook_event("invoice.payment_failed", payload, s3_client=s3)
    assert s3.puts == []


# ── killswitch: set_site_suspension ──────────────────────────────────────────


def test_suspend_true_writes_maintenance_page():
    seed_business()
    s3 = FakeS3()
    url = billing.set_site_suspension("acme-plumbing", True, s3_client=s3)

    assert url == "https://preview.solospark.net/acme-plumbing/index.html"
    put = s3.puts[0]
    html = put_body(put)
    assert "<!DOCTYPE html>" in html
    assert "Under Scheduled Maintenance" in html
    assert "Acme Plumbing" in html


def test_suspend_true_without_business_row_still_works():
    s3 = FakeS3()
    url = billing.set_site_suspension("ghost-business", True, s3_client=s3)
    assert url == "https://preview.solospark.net/ghost-business/index.html"
    html = put_body(s3.puts[0])
    assert "Under Scheduled Maintenance" in html
    assert "tel:" not in html  # no phone known → no call link


def test_suspend_false_restores_landing_page():
    business = seed_business()
    store_copy(business, COPY)
    s3 = FakeS3()

    billing.set_site_suspension("acme-plumbing", True, s3_client=s3)
    url = billing.set_site_suspension("acme-plumbing", False, s3_client=s3)

    assert url == "https://preview.solospark.net/acme-plumbing/index.html"
    assert len(s3.puts) == 2
    restored = put_body(s3.puts[1])
    assert "Reliable plumbing for Eugene homes" in restored
    assert "Under Scheduled Maintenance" not in restored


def test_suspend_false_without_copy_returns_none():
    seed_business()  # no generated copy stored
    s3 = FakeS3()
    assert billing.set_site_suspension("acme-plumbing", False, s3_client=s3) is None
    assert s3.puts == []


# ── offboarding: export_site_zip ─────────────────────────────────────────────


def test_export_zip_bundles_r2_and_local_assets(tmp_path):
    seed_business()
    s3 = FakeS3(
        objects={
            "acme-plumbing/index.html": b"<html>landing</html>",
            "acme-plumbing/about.html": b"<html>about</html>",
        }
    )
    assets_dir = billing.ASSETS_DIR / "acme-plumbing"
    assets_dir.mkdir(parents=True, exist_ok=True)
    logo = assets_dir / "logo.png"
    logo.write_bytes(b"\x89PNG fake")
    try:
        out = tmp_path / "offboard" / "acme.zip"
        result = billing.export_site_zip("acme-plumbing", str(out), s3_client=s3)

        assert Path(result).is_file()
        with zipfile.ZipFile(result) as zf:
            names = zf.namelist()
            assert sorted(names) == ["about.html", "index.html", "logo.png"]
            assert zf.read("index.html") == b"<html>landing</html>"
            assert zf.read("logo.png") == b"\x89PNG fake"
    finally:
        logo.unlink(missing_ok=True)
        assets_dir.rmdir()


def test_export_zip_falls_back_to_rendered_copy(tmp_path):
    business = seed_business()
    store_copy(business, COPY)
    s3 = FakeS3()  # R2 empty

    out = tmp_path / "fallback.zip"
    result = billing.export_site_zip("acme-plumbing", str(out), s3_client=s3)

    with zipfile.ZipFile(result) as zf:
        assert zf.namelist() == ["index.html"]
        html = zf.read("index.html").decode("utf-8")
        assert "Reliable plumbing for Eugene homes" in html


def test_export_zip_raises_when_nothing_found(tmp_path):
    seed_business()  # row exists but no copy and R2 empty
    s3 = FakeS3()
    with pytest.raises(ValueError, match="No site assets found"):
        billing.export_site_zip("acme-plumbing", str(tmp_path / "x.zip"), s3_client=s3)


def test_export_zip_raises_for_unknown_slug(tmp_path):
    s3 = FakeS3()
    with pytest.raises(ValueError):
        billing.export_site_zip("ghost-business", str(tmp_path / "y.zip"), s3_client=s3)


# ── HTTP endpoint: POST /api/billing/webhook ────────────────────────────────


def test_endpoint_activates_via_testclient():
    seed_business()
    s3 = FakeS3()
    client = TestClient(create_app(s3_client=s3))

    resp = client.post(
        "/api/billing/webhook",
        json={
            "type": "checkout.session.completed",
            "data": {"object": {"metadata": {"slug": "acme-plumbing"}}},
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "activated", "slug": "acme-plumbing"}


def test_endpoint_suspends_via_testclient():
    seed_business()
    s3 = FakeS3()
    client = TestClient(create_app(s3_client=s3))

    resp = client.post(
        "/api/billing/webhook",
        json={
            "type": "customer.subscription.deleted",
            "data": {"object": {"metadata": {"slug": "acme-plumbing"}}},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "suspended"
    assert len(s3.puts) == 1


def test_endpoint_unknown_type_returns_400():
    client = TestClient(create_app(s3_client=FakeS3()))
    resp = client.post(
        "/api/billing/webhook", json={"type": "charge.refunded", "data": {}}
    )
    assert resp.status_code == 400


def test_endpoint_unknown_business_returns_404():
    client = TestClient(create_app(s3_client=FakeS3()))
    resp = client.post(
        "/api/billing/webhook",
        json={
            "type": "invoice.payment_failed",
            "data": {"object": {"metadata": {"slug": "ghost"}}},
        },
    )
    assert resp.status_code == 404


def test_endpoint_malformed_json_returns_400():
    client = TestClient(create_app(s3_client=FakeS3()))
    resp = client.post(
        "/api/billing/webhook",
        content=b"not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_endpoint_missing_type_returns_400():
    client = TestClient(create_app(s3_client=FakeS3()))
    resp = client.post("/api/billing/webhook", json={"data": {}})
    assert resp.status_code == 400
