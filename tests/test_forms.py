"""Phase 9 tests — contact form lead routing relay.

Covers owner-email composition, provider selection (Resend / SMTP / dry-run),
routing errors, and the POST /api/forms/submit endpoint with both form-encoded
and JSON bodies. No live mail services required: transports are injected fakes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import forms, models
from app.config import get_settings
from app.main import create_app


# ── fixtures / helpers ───────────────────────────────────────────────────────


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
            "contact_email": "owner@acmepiping.com",
        }
        fields.update(overrides)
        business = models.Business(**fields)
        db.add(business)
        db.commit()
        db.refresh(business)
        return business
    finally:
        db.close()


def submit(slug="acme-plumbing"):
    return {
        "business_slug": slug,
        "name": "Dana Prospect",
        "phone": "(541) 555-9876",
        "message": "My water heater is leaking — can you come out today?",
    }


# ── core relay (app/forms.py) ────────────────────────────────────────────────


def test_build_inquiry_email_composes_owner_notification():
    business = seed_business()
    subject, body = forms.build_inquiry_email(
        business, "Dana Prospect", "(541) 555-9876", "Leak in the basement."
    )
    assert "Dana Prospect" in subject
    assert "Acme Plumbing" in subject
    assert "Name:  Dana Prospect" in body
    assert "Phone: (541) 555-9876" in body
    assert "Leak in the basement." in body
    assert get_settings().business_physical_address in body


def test_send_dry_run_when_no_provider_configured():
    seed_business()
    result = forms.send_inquiry_email(
        "acme-plumbing", "Dana Prospect", "(541) 555-9876", "Hello there"
    )
    assert result == {"status": "dry_run", "provider": "none"}


def test_send_via_smtp_when_host_configured(monkeypatch):
    seed_business()
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    get_settings.cache_clear()

    captured = {}

    class FakeServer:
        def sendmail(self, from_addr, to_addrs, raw):
            captured["from"] = from_addr
            captured["to"] = to_addrs
            captured["raw"] = raw

        def quit(self):
            captured["quit"] = True

    def smtp_factory(host, port):
        captured["host"] = host
        captured["port"] = port
        return FakeServer()

    result = forms.send_inquiry_email(
        "acme-plumbing", "Dana Prospect", "(541) 555-9876", "Hello there",
        smtp_factory=smtp_factory,
    )
    assert result == {"status": "sent", "provider": "smtp"}
    assert (captured["host"], captured["port"]) == ("smtp.example.com", 587)
    assert captured["to"] == ["owner@acmepiping.com"]
    assert "Dana Prospect" in captured["raw"]
    assert "Hello there" in captured["raw"]
    assert captured.get("quit") is True


def test_send_via_resend_when_key_configured(monkeypatch):
    seed_business()
    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    get_settings.cache_clear()

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"id": "em_123"}

    class FakeSession:
        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResp()

    result = forms.send_inquiry_email(
        "acme-plumbing", "Dana Prospect", "(541) 555-9876", "Hello there",
        http_session=FakeSession(),
    )
    assert result == {"status": "sent", "provider": "resend", "id": "em_123"}
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["to"] == ["owner@acmepiping.com"]
    assert captured["json"]["subject"].startswith("New website inquiry from Dana Prospect")
    assert "Hello there" in captured["json"]["text"]


def test_unknown_slug_raises_lookup_error():
    with pytest.raises(LookupError, match="Unknown business slug"):
        forms.send_inquiry_email("nope", "Dana", "123", "Hi")


def test_missing_owner_email_raises_value_error():
    seed_business(contact_email=None)
    with pytest.raises(ValueError, match="No owner email on file"):
        forms.send_inquiry_email("acme-plumbing", "Dana", "123", "Hi")


# ── endpoint (POST /api/forms/submit) ────────────────────────────────────────


def test_endpoint_form_encoded_returns_spec_success():
    seed_business()
    client = TestClient(create_app())
    resp = client.post("/api/forms/submit", data=submit())
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "success",
        "message": "Inquiry sent successfully",
    }


def test_endpoint_accepts_json_body():
    seed_business()
    client = TestClient(create_app())
    resp = client.post("/api/forms/submit", json=submit())
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_endpoint_missing_required_fields_returns_400():
    seed_business()
    client = TestClient(create_app())
    payload = submit()
    del payload["name"]
    resp = client.post("/api/forms/submit", data=payload)
    assert resp.status_code == 400


def test_endpoint_unknown_slug_returns_404():
    seed_business()
    client = TestClient(create_app())
    resp = client.post("/api/forms/submit", data=submit(slug="ghost-slug"))
    assert resp.status_code == 404


def test_endpoint_no_owner_email_returns_400():
    seed_business(contact_email=None)
    client = TestClient(create_app())
    resp = client.post("/api/forms/submit", data=submit())
    assert resp.status_code == 400
