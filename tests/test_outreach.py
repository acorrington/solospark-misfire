"""Phase 6 tests — outreach dispatch, CAN-SPAM footer, rate limit, opt-out."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import models
from app.config import get_settings
from app.main import create_app
from app.outreach import (
    RESEND_API_URL,
    OutreachRateLimitError,
    send_outreach_email,
)

# Env vars the tests toggle — cleared before every test so providers are off.
_TOGGLED_ENV = (
    "RESEND_API_KEY",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "BUSINESS_PHYSICAL_ADDRESS",
    "OUTREACH_RATE_LIMIT_PER_HOUR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _TOGGLED_ENV:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield


def _setup_db(tmp_path) -> None:
    models.reinit_db(f"sqlite:///{tmp_path / 'test.db'}")


_seed_counter = 0


def seed_business(**kw) -> int:
    """Insert a business into the current test DB and return its id."""
    global _seed_counter
    _seed_counter += 1
    defaults = dict(
        place_id=f"outreach-pid-{_seed_counter}",
        name="Acme Plumbing",
        slug="acme-plumbing",
        category="Plumber",
        address="10 Main St, Eugene, OR 97401",
        phone="(541) 555-0100",
        current_website=None,
        contact_email="owner@acme.com",
        rating=4.6,
        review_count=38,
        stage=models.DealStage.AUDITED,
    )
    defaults.update(kw)
    db = models.get_session_factory()()
    try:
        business = models.Business(**defaults)
        db.add(business)
        db.commit()
        db.refresh(business)
        return business.id
    finally:
        db.close()


def _ledger_rows(business_id: int | None = None) -> list:
    db = models.get_session_factory()()
    try:
        query = select(models.OutreachEmail)
        if business_id is not None:
            query = query.where(models.OutreachEmail.business_id == business_id)
        return list(db.scalars(query).all())
    finally:
        db.close()


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"id": "email_123"}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeResendSession:
    """Stands in for requests.Session against the Resend API."""

    def __init__(self, response=None):
        self.calls = []
        self.response = response or _FakeResponse()

    def post(self, url, headers=None, json=None, **kwargs):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response


class FakeSMTP:
    def __init__(self):
        self.logged_in = None
        self.sent = []
        self.quit_called = False

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        self.logged_in = (user, password)

    def sendmail(self, from_addr, to_addrs, msg):
        self.sent.append((from_addr, to_addrs, msg))

    def quit(self):
        self.quit_called = True


# ── Provider dispatch + footer + ledger ─────────────────────────────────────


def test_send_via_resend_appends_footer_and_records_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv(
        "BUSINESS_PHYSICAL_ADDRESS", "123 Spark Way, Roseburg, OR 97470"
    )
    get_settings.cache_clear()
    _setup_db(tmp_path)
    seed_business()

    fake = FakeResendSession()
    result = send_outreach_email(
        "owner@acme.com", "Your site is costing you jobs", "Hi there.",
        "acme-plumbing", http_session=fake,
    )

    assert result == {"status": "sent", "provider": "resend", "id": "email_123"}

    call = fake.calls[0]
    assert call["url"] == RESEND_API_URL
    assert call["headers"]["Authorization"] == "Bearer re_test_key"
    payload = call["json"]
    assert payload["from"] == "aaron@solosparkmail.com"
    assert payload["to"] == ["owner@acme.com"]
    assert payload["subject"] == "Your site is costing you jobs"

    text = payload["text"]
    assert text.startswith("Hi there.")
    # CAN-SPAM footer: physical address + STOP reply + unsubscribe link.
    assert "123 Spark Way, Roseburg, OR 97470" in text
    assert "Reply STOP or click here to unsubscribe:" in text
    assert "/api/outreach/unsubscribe?email=owner%40acme.com" in text

    rows = _ledger_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "sent"
    assert row.recipient == "owner@acme.com"
    assert row.sent_at is not None
    assert row.body == text


def test_send_via_smtp_when_no_resend_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "smtp-user")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-pass")
    get_settings.cache_clear()
    _setup_db(tmp_path)
    seed_business()

    fake_smtp = FakeSMTP()
    captured = {}

    def factory(host, port):
        captured["host"] = host
        captured["port"] = port
        return fake_smtp

    result = send_outreach_email(
        "owner@acme.com", "Hi", "Body text.", "acme-plumbing",
        smtp_factory=factory,
    )

    assert result == {"status": "sent", "provider": "smtp"}
    assert captured == {"host": "smtp.example.com", "port": 587}
    assert fake_smtp.logged_in == ("smtp-user", "smtp-pass")
    assert fake_smtp.quit_called is True

    from_addr, to_addrs, msg = fake_smtp.sent[0]
    assert from_addr == "aaron@solosparkmail.com"
    assert to_addrs == ["owner@acme.com"]
    assert "Subject: Hi" in msg
    assert "Reply STOP or click here to unsubscribe:" in msg

    rows = _ledger_rows()
    assert len(rows) == 1 and rows[0].status == "sent"


def test_dry_run_when_no_provider_configured(tmp_path):
    _setup_db(tmp_path)
    bid = seed_business()

    result = send_outreach_email(
        "owner@acme.com", "Hi", "Body text.", "acme-plumbing"
    )

    assert result == {"status": "dry_run", "provider": "none"}
    rows = _ledger_rows(bid)
    assert len(rows) == 1
    assert rows[0].status == "queued"
    assert rows[0].sent_at is None


def test_failed_send_records_ledger_and_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    get_settings.cache_clear()
    _setup_db(tmp_path)
    bid = seed_business()

    fake = FakeResendSession(response=_FakeResponse(status_code=500, payload={}))
    with pytest.raises(RuntimeError, match="Resend API error 500"):
        send_outreach_email(
            "owner@acme.com", "Hi", "Body.", "acme-plumbing", http_session=fake
        )

    rows = _ledger_rows(bid)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "500" in (rows[0].error or "")


# ── Rate limiting ────────────────────────────────────────────────────────────


def _seed_recent_sends(count: int, sent_at: datetime) -> None:
    """Insert `count` distinct businesses each with one 'sent' ledger row."""
    db = models.get_session_factory()()
    try:
        for i in range(count):
            business = models.Business(
                place_id=f"rl-pid-{i}-{count}",
                name=f"Lead {i}",
                slug=f"lead-{i}-{count}",
                contact_email=f"lead{i}@example.com",
                stage=models.DealStage.AUDITED,
            )
            db.add(business)
            db.flush()
            db.add(
                models.OutreachEmail(
                    business_id=business.id,
                    recipient=f"lead{i}@example.com",
                    subject="s",
                    body="b",
                    status="sent",
                    sent_at=sent_at,
                )
            )
        db.commit()
    finally:
        db.close()


def test_rate_limit_blocks_sixth_send_in_hour(tmp_path):
    _setup_db(tmp_path)  # dry-run mode: no providers configured
    _seed_recent_sends(5, datetime.now(timezone.utc) - timedelta(minutes=10))

    with pytest.raises(OutreachRateLimitError, match="rate limit"):
        _send_fresh_lead()


def _send_fresh_lead() -> dict:
    """Helper: seed a fresh business and attempt its first email."""
    seed_business(slug="fresh-lead", contact_email="fresh@example.com")
    return send_outreach_email("fresh@example.com", "Hi", "Body.", "fresh-lead")


def test_rate_limit_window_expires(tmp_path):
    _setup_db(tmp_path)  # dry-run mode
    _seed_recent_sends(5, datetime.now(timezone.utc) - timedelta(hours=2))

    result = _send_fresh_lead()
    assert result == {"status": "dry_run", "provider": "none"}
    rows = _ledger_rows()
    assert len(rows) == 6  # 5 stale + 1 new queued


# ── Guard rails (dedup / opt-out / unknown slug) ─────────────────────────────


def test_dedup_refuses_second_send_to_same_business(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    get_settings.cache_clear()
    _setup_db(tmp_path)
    seed_business()

    first = send_outreach_email(
        "owner@acme.com", "Hi", "Body.", "acme-plumbing",
        http_session=FakeResendSession(),
    )
    assert first["status"] == "sent"

    with pytest.raises(ValueError, match="already received"):
        send_outreach_email(
            "owner@acme.com", "Again", "Body.", "acme-plumbing",
            http_session=FakeResendSession(),
        )


def test_opted_out_business_is_refused(tmp_path):
    _setup_db(tmp_path)
    seed_business(opted_out=True)

    with pytest.raises(ValueError, match="opted out"):
        send_outreach_email("owner@acme.com", "Hi", "Body.", "acme-plumbing")


def test_unknown_slug_raises_value_error(tmp_path):
    _setup_db(tmp_path)

    with pytest.raises(ValueError, match="Unknown business slug"):
        send_outreach_email("nobody@example.com", "Hi", "Body.", "ghost-slug")


# ── Unsubscribe endpoint + 429 mapping ───────────────────────────────────────


def test_unsubscribe_marks_leads_lost_and_opted_out(tmp_path):
    _setup_db(tmp_path)
    bid = seed_business(
        contact_email="Owner@Acme.com", stage=models.DealStage.CONTACTED
    )

    client = TestClient(create_app())
    resp = client.get(
        "/api/outreach/unsubscribe", params={"email": "owner@acme.com"}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unsubscribed"
    assert data["businesses_updated"] == 1

    db = models.get_session_factory()()
    try:
        business = db.get(models.Business, bid)
        assert business.opted_out is True
        assert business.stage == models.DealStage.LOST
    finally:
        db.close()


def test_send_endpoint_maps_rate_limit_to_429(tmp_path):
    _setup_db(tmp_path)
    bid = seed_business()

    def _boom(**kwargs):
        raise OutreachRateLimitError("slow down")

    client = TestClient(create_app(outreach_sender=_boom))
    resp = client.post(
        f"/outreach/send/{bid}", json={"subject": "Hi", "body": "Hello"}
    )

    assert resp.status_code == 429
    assert resp.json()["detail"] == "slow down"
