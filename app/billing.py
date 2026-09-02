"""Phase 8 — Stripe webhooks, delinquency killswitch & client offboarding.

Three capabilities:

* ``handle_webhook_event`` — interprets Stripe events:
    - ``checkout.session.completed`` → subscription active, deal won
    - ``customer.subscription.deleted`` / ``invoice.payment_failed`` →
      subscription past due and the live site is suspended (killswitch)
* ``set_site_suspension`` — overwrites ``{slug}/index.html`` on R2 with a
  clean "Under Scheduled Maintenance" page (or restores the stored landing
  page when unsuspended)
* ``export_site_zip`` — bundles every generated asset for a client into a
  standalone ``.zip`` for offboarding

Every external dependency (S3/R2, database) is injectable or backed by the
module-level session factory, so all paths are testable without live services.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

from .config import get_settings
from .deployer import (
    _tel_href,
    build_s3_client,
    preview_url_for,
    render_site_html,
)
from .models import Business, DealStage, SubscriptionStatus, get_session_factory

# Project-level asset directory (logos etc. downloaded during scanning).
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


# ── Maintenance page (killswitch content) ────────────────────────────────────


def maintenance_page_html(name: str | None = None, phone: str | None = None) -> str:
    """Render the clean 'Under Scheduled Maintenance' landing page.

    Self-contained (inline CSS, no CDN dependencies) so it renders even when
    the real site's assets are unavailable.
    """
    if name:
        intro = f"{name} is temporarily offline while we finish some scheduled work."
    else:
        intro = "This site is temporarily offline while we finish some scheduled work."

    contact = ""
    href = _tel_href(phone or "")
    if href:
        contact = (
            '<p>Need help right now? Call '
            f'<a href="tel:{href}">{phone}</a> — we answer fast.</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Under Scheduled Maintenance</title>
<style>
  body {{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:#f8fafc; color:#0f172a;
         font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
  main {{ text-align:center; padding:2rem; max-width:34rem; }}
  .badge {{ display:inline-block; background:#e2e8f0; color:#334155; border-radius:9999px;
            padding:.375rem .875rem; font-size:.875rem; font-weight:600; margin-bottom:1.25rem; }}
  h1 {{ font-size:2rem; margin:0 0 .75rem; letter-spacing:-.02em; }}
  p {{ color:#475569; line-height:1.6; margin:.5rem 0; }}
  a {{ color:#1d4ed8; font-weight:600; text-decoration:none; }}
</style>
</head>
<body>
<main>
  <span class="badge">Scheduled maintenance</span>
  <h1>Under Scheduled Maintenance</h1>
  <p>{intro} Please check back soon.</p>
  {contact}
</main>
</body>
</html>"""


# ── Killswitch ───────────────────────────────────────────────────────────────


def _find_by_slug(db, slug: str) -> Business | None:
    return db.query(Business).filter(Business.slug == slug).first()


def set_site_suspension(
    slug: str, suspend: bool, *, s3_client=None
) -> str | None:
    """Suspend or restore a deployed site on R2.

    * ``suspend=True`` — overwrites ``{slug}/index.html`` with the maintenance
      page (works even if the business row is missing; the page just omits
      name/phone). Returns the public preview URL.
    * ``suspend=False`` — re-renders the landing page from the stored LLM copy
      and uploads it again. Returns the preview URL, or ``None`` when there is
      no stored copy to restore from.
    """
    client = s3_client if s3_client is not None else build_s3_client()
    settings = get_settings()

    db = get_session_factory()()
    try:
        business = _find_by_slug(db, slug)
        name = business.name if business else None
        phone = business.phone if business else None
        copy = business.generated_copy_dict() if business else None
    finally:
        db.close()

    key = f"{slug}/index.html"
    if suspend:
        html = maintenance_page_html(name, phone)
        client.put_object(
            Bucket=settings.r2_bucket_name,
            Key=key,
            Body=html.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
        )
        return preview_url_for(slug)

    if not copy:
        return None
    html = render_site_html(
        {
            "name": business.name,
            "slug": business.slug,
            "category": business.category or "",
            "address": business.address or "",
            "phone": business.phone or "",
        },
        copy,
    )
    client.put_object(
        Bucket=settings.r2_bucket_name,
        Key=key,
        Body=html.encode("utf-8"),
        ContentType="text/html; charset=utf-8",
    )
    return preview_url_for(slug)


# ── Offboarding export ───────────────────────────────────────────────────────


def _business_data(business: Business) -> dict[str, str]:
    return {
        "name": business.name,
        "slug": business.slug,
        "category": business.category or "",
        "address": business.address or "",
        "phone": business.phone or "",
    }


def export_site_zip(slug: str, output_path: str, *, s3_client=None) -> str:
    """Bundle every generated asset for ``slug`` into a standalone zip.

    Collects all objects under the ``{slug}/`` R2 prefix (multi-page sites
    included), adds any locally stored asset files (e.g. ``assets/{slug}/logo.png``),
    and falls back to rendering the landing page from the stored copy when R2
    holds nothing. Returns the absolute path of the written zip.

    Raises ``ValueError`` when there is no site content at all.
    """
    client = s3_client if s3_client is not None else build_s3_client()
    settings = get_settings()
    prefix = f"{slug}/"

    files: dict[str, bytes] = {}
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": settings.r2_bucket_name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs) or {}
        for obj in resp.get("Contents", []):
            key = str(obj["Key"])
            body = client.get_object(Bucket=settings.r2_bucket_name, Key=key)["Body"]
            data = body.read() if hasattr(body, "read") else bytes(body)
            rel = key[len(prefix):]
            if rel:
                files[rel] = data
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    if not files:
        db = get_session_factory()()
        try:
            business = _find_by_slug(db, slug)
            copy = business.generated_copy_dict() if business else None
        finally:
            db.close()
        if not copy:
            raise ValueError(
                f"No site assets found for slug {slug!r} — R2 is empty and no "
                "stored copy exists to render from"
            )
        files["index.html"] = render_site_html(_business_data(business), copy).encode("utf-8")

    local_dir = ASSETS_DIR / slug
    if local_dir.is_dir():
        for path in sorted(local_dir.iterdir()):
            if path.is_file():
                files.setdefault(path.name, path.read_bytes())

    out = Path(output_path)
    if str(out.parent) not in ("", "."):
        out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname in sorted(files):
            zf.writestr(arcname, files[arcname])
    return str(out.resolve())


# ── Stripe webhook handling ──────────────────────────────────────────────────


def _find_business_in(db, payload: dict[str, Any]) -> Business | None:
    """Locate the business a Stripe event refers to.

    Lookup order: ``data.object.metadata.slug`` → ``client_reference_id`` →
    ``stripe_customer_id == data.object.customer``.
    """
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    obj = data.get("object") if isinstance(data.get("object"), dict) else {}
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    slug = metadata.get("slug") or obj.get("client_reference_id")
    if slug:
        business = _find_by_slug(db, str(slug))
        if business is not None:
            return business

    customer = obj.get("customer")
    if customer:
        return (
            db.query(Business)
            .filter(Business.stripe_customer_id == str(customer))
            .first()
        )
    return None


def handle_webhook_event(
    event_type: str, payload: dict[str, Any], *, s3_client=None
) -> dict[str, Any]:
    """Apply one Stripe webhook event to the pipeline.

    Returns ``{"status": "activated"|"suspended", "slug": ...}``.
    Raises ``ValueError`` for unsupported event types and ``LookupError`` when
    no business matches the payload.
    """
    if not isinstance(payload, dict):
        raise ValueError("Webhook payload must be a JSON object")

    if event_type == "checkout.session.completed":
        action = "activate"
    elif event_type in ("customer.subscription.deleted", "invoice.payment_failed"):
        action = "suspend"
    else:
        raise ValueError(f"Unsupported webhook event type: {event_type!r}")

    db = get_session_factory()()
    try:
        business = _find_business_in(db, payload)
        if business is None:
            raise LookupError("No business matches this webhook payload")

        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        obj = data.get("object") if isinstance(data.get("object"), dict) else {}

        if action == "activate":
            business.subscription_status = SubscriptionStatus.ACTIVE
            business.stage = DealStage.WON
            sub_id = obj.get("subscription") or obj.get("id")
            if sub_id:
                business.stripe_subscription_id = str(sub_id)
            db.commit()
            return {"status": "activated", "slug": business.slug}

        business.subscription_status = SubscriptionStatus.PAST_DUE
        set_site_suspension(business.slug, True, s3_client=s3_client)
        db.commit()
        return {"status": "suspended", "slug": business.slug}
    finally:
        db.close()
