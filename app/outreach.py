"""Phase 6 — Outreach email dispatch (CAN-SPAM compliant).

Sends personalized pitch emails to discovered local businesses via the
Resend HTTP API, falling back to plain SMTP when no Resend key is set, and
falling back to a no-op dry-run when neither is configured (so the pipeline
and tests run without live mail services).

Every real send:
  * appends the CAN-SPAM footer (physical address + STOP reply + unsubscribe
    link),
  * is rate-limited to ``OUTREACH_RATE_LIMIT_PER_HOUR`` sends per rolling
    hour (enforced from the OutreachEmail ledger, so state survives restarts),
  * refuses businesses that have opted out or already received a sent email
    (ledger dedup), and
  * records a ledger row (sent / queued / failed) for audit and CLI review.
"""

from __future__ import annotations

import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any
from urllib.parse import quote

import requests
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import Business, DealStage, OutreachEmail, get_db, get_session_factory

RESEND_API_URL = "https://api.resend.com/emails"


class OutreachRateLimitError(Exception):
    """Raised when sending would exceed the hourly outreach rate limit.

    ``app.main`` maps this exception to an HTTP 429 response (it matches on
    the class name, so keep the name stable).
    """


# ── CAN-SPAM footer ──────────────────────────────────────────────────────────


def unsubscribe_url_for(email: str) -> str:
    """Absolute URL of the opt-out endpoint for a given recipient address."""
    settings = get_settings()
    return (
        f"http://localhost:{settings.port}/api/outreach/unsubscribe"
        f"?email={quote(email, safe='')}"
    )


def build_footer(to_email: str) -> str:
    """CAN-SPAM footer: physical address + STOP reply + unsubscribe link."""
    settings = get_settings()
    return "\n".join(
        [
            "",
            "----",
            "SoloSpark LLC",
            settings.business_physical_address,
            "",
            f"Reply STOP or click here to unsubscribe: {unsubscribe_url_for(to_email)}",
        ]
    )


# ── Rate limiting (ledger-backed so it survives restarts) ───────────────────


def count_recent_sends(db: Session, window: timedelta | None = None) -> int:
    """Count ledger rows marked 'sent' whose sent_at falls inside the window."""
    if window is None:
        window = timedelta(hours=1)
    cutoff = datetime.now(timezone.utc) - window
    count = db.scalar(
        select(func.count())
        .select_from(OutreachEmail)
        .where(
            OutreachEmail.status == "sent",
            OutreachEmail.sent_at >= cutoff,
        )
    )
    return int(count or 0)


# ── Providers ────────────────────────────────────────────────────────────────


def _send_via_resend(
    to_email: str, subject: str, body_text: str, http_session=None
) -> dict[str, Any]:
    """POST the email to the Resend API. Returns provider result metadata."""
    settings = get_settings()
    session = http_session if http_session is not None else requests.Session()
    resp = session.post(
        RESEND_API_URL,
        headers={"Authorization": f"Bearer {settings.resend_api_key}"},
        json={
            "from": settings.outreach_from_email,
            "to": [to_email],
            "subject": subject,
            "text": body_text,
        },
    )
    if getattr(resp, "status_code", 200) >= 400:
        detail = getattr(resp, "text", "") or ""
        raise RuntimeError(f"Resend API error {resp.status_code}: {detail[:300]}")
    data = resp.json() if hasattr(resp, "json") else {}
    return {"status": "sent", "provider": "resend", "id": (data or {}).get("id")}


def _send_via_smtp(
    to_email: str, subject: str, body_text: str, smtp_factory=None
) -> dict[str, Any]:
    """Send through a plain SMTP server (STARTTLS + optional login)."""
    settings = get_settings()

    msg = EmailMessage()
    msg["From"] = settings.outreach_from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body_text)

    if smtp_factory is None:

        def smtp_factory(host: str, port: int):  # real network path
            server = smtplib.SMTP(host, port, timeout=30)
            server.starttls(context=ssl.create_default_context())
            return server

    server = smtp_factory(settings.smtp_host, settings.smtp_port)
    try:
        if settings.smtp_user and settings.smtp_password:
            server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(settings.outreach_from_email, [to_email], msg.as_string())
    finally:
        quit_fn = getattr(server, "quit", None)
        if callable(quit_fn):
            try:
                quit_fn()
            except Exception:  # pragma: no cover — best effort cleanup
                pass
    return {"status": "sent", "provider": "smtp"}


# ── Public entry point ───────────────────────────────────────────────────────


def send_outreach_email(
    to_email: str,
    subject: str,
    body_text: str,
    business_slug: str,
    *,
    db: Session | None = None,
    http_session=None,
    smtp_factory=None,
    ledger_row: OutreachEmail | None = None,
) -> dict[str, Any]:
    """Send one CAN-SPAM-compliant outreach email for a business.

    Provider selection: Resend API when ``RESEND_API_KEY`` is set, else SMTP
    when ``SMTP_HOST`` is set, else a dry-run (no mail sent, ledger row
    recorded as "queued" for review).

    When ``ledger_row`` is given (the GUI approval flow), that existing
    "queued" row is updated in place instead of inserting a new one, so the
    queue row and the audit trail stay a single record per send attempt.

    Raises:
        ValueError: unknown slug, opted-out business, or duplicate send.
        OutreachRateLimitError: hourly rate limit would be exceeded.
        RuntimeError: the mail provider rejected the send (a "failed" ledger
            row is recorded first).
    """
    settings = get_settings()
    own_session = db is None
    if own_session:
        db = get_session_factory()()
    try:
        business = db.scalar(select(Business).where(Business.slug == business_slug))
        if business is None:
            raise ValueError(f"Unknown business slug: {business_slug!r}")
        if business.opted_out:
            raise ValueError(
                f"Business {business_slug!r} has opted out of outreach"
            )

        already_sent = db.scalar(
            select(OutreachEmail.id)
            .where(
                OutreachEmail.business_id == business.id,
                OutreachEmail.status == "sent",
            )
            .limit(1)
        )
        if already_sent is not None:
            raise ValueError(
                f"Business {business_slug!r} has already received an outreach email"
            )

        limit = settings.outreach_rate_limit_per_hour
        recent = count_recent_sends(db, timedelta(hours=1))
        if recent >= limit:
            raise OutreachRateLimitError(
                f"Outreach rate limit exceeded: {recent} emails sent in the last "
                f"hour (max {limit}/hour). Try again later."
            )

        full_body = body_text.rstrip() + "\n" + build_footer(to_email)

        try:
            if settings.resend_api_key:
                result = _send_via_resend(
                    to_email, subject, full_body, http_session=http_session
                )
            elif settings.smtp_host:
                result = _send_via_smtp(
                    to_email, subject, full_body, smtp_factory=smtp_factory
                )
            else:
                result = {"status": "dry_run", "provider": "none"}
        except Exception as exc:  # record the failure, then surface it
            if ledger_row is not None:
                ledger_row.status = "failed"
                ledger_row.error = str(exc)
            else:
                db.add(
                    OutreachEmail(
                        business_id=business.id,
                        recipient=to_email,
                        subject=subject,
                        body=full_body,
                        status="failed",
                        error=str(exc),
                    )
                )
            db.commit()
            raise

        is_real_send = result["status"] == "sent"
        if ledger_row is not None:
            ledger_row.recipient = to_email
            ledger_row.subject = subject
            ledger_row.body = full_body
            ledger_row.status = "sent" if is_real_send else "queued"
            ledger_row.sent_at = (
                datetime.now(timezone.utc) if is_real_send else None
            )
        else:
            db.add(
                OutreachEmail(
                    business_id=business.id,
                    recipient=to_email,
                    subject=subject,
                    body=full_body,
                    status="sent" if is_real_send else "queued",
                    sent_at=datetime.now(timezone.utc) if is_real_send else None,
                )
            )
        db.commit()
        return result
    finally:
        if own_session:
            db.close()


# ── CAN-SPAM opt-out endpoint ────────────────────────────────────────────────

outreach_router = APIRouter(prefix="/api/outreach", tags=["outreach"])


@outreach_router.get("/unsubscribe")
def unsubscribe(email: str, db: Session = Depends(get_db)):
    """Mark every lead matching the address as opted out and lost."""
    rows = db.scalars(
        select(Business).where(
            func.lower(Business.contact_email) == email.strip().lower()
        )
    ).all()
    for business in rows:
        business.opted_out = True
        business.stage = DealStage.LOST
    db.commit()
    return {
        "status": "unsubscribed",
        "email": email,
        "businesses_updated": len(rows),
    }
