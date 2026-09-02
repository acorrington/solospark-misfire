"""Phase 9 — Contact form lead routing relay.

Receives contact-form submissions from the deployed client sites and emails
the business owner immediately with the prospect's contact info plus the
inquiry details.

This is a transactional notification to the site owner, not outbound
marketing — so it deliberately bypasses the Phase 6 outreach ledger: no
CAN-SPAM footer, no rate-limit slot consumption, no per-business dedup.
Every submission is routed exactly once.

Provider selection mirrors Phase 6: Resend API when ``RESEND_API_KEY`` is
set, plain SMTP when ``SMTP_HOST`` is set, otherwise a dry-run (no mail
sent) so the pipeline and tests run without live mail services.
"""

from __future__ import annotations

from typing import Any

from .config import get_settings
from .models import Business, get_session_factory
from .outreach import _send_via_resend, _send_via_smtp


def build_inquiry_email(
    business: Business, name: str, phone: str, message: str
) -> tuple[str, str]:
    """Compose the (subject, body) pair for the owner notification email."""
    who = name.strip() or "a visitor"
    subject = f"New website inquiry from {who} — {business.name}"
    lines = [
        f"You have a new inquiry from your website ({business.name}).",
        "",
        f"Name:  {name.strip()}",
        f"Phone: {phone.strip()}",
        "",
        "Message:",
        message.strip(),
        "",
        "----",
        "SoloSpark LLC",
        get_settings().business_physical_address,
    ]
    return subject, "\n".join(lines)


def send_inquiry_email(
    business_slug: str,
    name: str,
    phone: str,
    message: str,
    *,
    http_session=None,
    smtp_factory=None,
) -> dict[str, Any]:
    """Route one contact-form submission to the business owner.

    Returns provider metadata (``{"status": "sent", "provider": ...}``) or a
    dry-run marker when no mail provider is configured.

    Raises:
        LookupError: unknown business slug (the site no longer exists).
        ValueError: the business has no ``contact_email`` on file, so there
            is nowhere to route the inquiry.
        RuntimeError: the mail provider rejected the send.
    """
    db = get_session_factory()()
    try:
        business = db.query(Business).filter_by(slug=business_slug).first()
        if business is None:
            raise LookupError(f"Unknown business slug: {business_slug!r}")
        to_email = (business.contact_email or "").strip()
        if not to_email:
            raise ValueError(
                f"No owner email on file for {business.name!r} — "
                "set Business.contact_email to receive inquiries"
            )
    finally:
        db.close()

    subject, body = build_inquiry_email(business, name, phone, message)
    settings = get_settings()
    if settings.resend_api_key:
        return _send_via_resend(to_email, subject, body, http_session=http_session)
    if settings.smtp_host:
        return _send_via_smtp(to_email, subject, body, smtp_factory=smtp_factory)
    return {"status": "dry_run", "provider": "none"}
