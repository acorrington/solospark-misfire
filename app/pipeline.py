"""Shared pipeline steps used by both the CLI (run.py) and the web dashboard.

These functions are deliberately UI-agnostic: they take an explicit SQLAlchemy
session, return structured data, and never print on their own. The CLI wraps
them with argument parsing and console output; FastAPI routes call them
directly. An optional ``log`` callback (the CLI passes ``print``) receives the
same per-item progress lines the old CLI printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy import select

from app import scanner
from app.deployer import check_html_syntax, deploy_to_r2, render_site_html
from app.llm_engine import LLMGenerationError, generate_landing_copy, generate_pitch_email
from app.models import Business, DealStage, OutreachEmail
from app.outreach import OutreachRateLimitError, send_outreach_email
from app.utils import city_from_address

# Optional progress callback: the CLI passes ``print``; the web layer omits it.
Log = Callable[[str], None]


def discover(db, query: str, limit: int | None = None) -> tuple[int, list[Business]]:
    """Search Google Places and ingest new leads (ledger-deduped).

    Returns ``(total_places_found, newly_created_businesses)``.
    """
    places = scanner.discover_places(query=query)
    if limit:
        places = places[:limit]
    created = scanner.ingest_places(db, places)
    return len(places), created


def audit_pending(db, *, log: Optional[Log] = None) -> list[Business]:
    """Audit every DISCOVERED lead that has a website to fetch.

    Returns the audited businesses (flags/stage already updated + committed).
    """
    pending = db.query(Business).filter(
        Business.stage == DealStage.DISCOVERED,
        Business.current_website.is_not(None),
    ).all()
    for b in pending:
        scanner.audit_business(db, b)
        if log:
            flags = b.audit_flags_list()
            verdict = "bad site" if b.is_bad_site else "looks fine"
            detail = ", ".join(flags) if flags else "no issues found"
            log(f"  • {b.name} [{b.slug}] — {verdict}: {detail}")
    return pending


def generate_for(db, business: Business) -> dict:
    """Generate landing copy for one lead and persist it (stage → MOCKUP_READY)."""
    copy = generate_landing_copy(
        business.name, business.category or "", city_from_address(business.address)
    )
    business.set_generated_copy(copy)
    if business.stage not in (
        DealStage.PITCH_APPROVED,
        DealStage.CONTACTED,
        DealStage.REPLIED,
        DealStage.WON,
    ):
        business.stage = DealStage.MOCKUP_READY
    db.commit()
    return copy


def deploy_business(db, business: Business) -> str:
    """Render the stored copy and publish it to R2; store the preview URL."""
    copy = business.generated_copy_dict()
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
    problems = check_html_syntax(html)
    if problems:
        # Never ship a structurally broken page — the web layer maps this to 503.
        raise ValueError(
            "site failed the pre-upload syntax check and was NOT uploaded: "
            + "; ".join(problems)
        )
    url = deploy_to_r2(business.slug, html)
    business.preview_url = url
    db.commit()
    return url


def deploy_many(
    db, businesses: list[Business], *, log: Optional[Log] = None
) -> list[tuple[Business, str | None]]:
    """Deploy every lead that has stored generated copy.

    Returns a list of ``(business, preview_url)`` pairs; the URL is ``None``
    for leads skipped because they have no generated copy.
    """
    results: list[tuple[Business, str | None]] = []
    for b in businesses:
        if not b.generated_copy_dict():
            if log:
                log(f"  - {b.name} [{b.slug}] — skipped (no generated copy)")
            results.append((b, None))
            continue
        url = deploy_business(db, b)
        if log:
            log(f"  ✓ {b.name} [{b.slug}] → {url}")
        results.append((b, url))
    return results


@dataclass
class OutreachResult:
    """Structured outcome of :func:`outreach_batch`.

    ``queued`` rows carry the CSV column names (business_slug, name, email,
    subject, body, preview_url, generated_at) so the CLI can write them to the
    dry-run review queue verbatim.
    """

    queued: list[dict] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)  # slugs successfully dispatched
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (slug, reason)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (slug, error)

    @property
    def counts(self) -> tuple[int, int, int, int]:
        """Return ``(queued, sent, skipped, failed)``."""
        return len(self.queued), len(self.sent), len(self.skipped), len(self.failed)


def outreach_batch(
    db,
    businesses: list[Business],
    dry_run: bool,
    *,
    log: Optional[Log] = None,
    persist_to_ledger: bool = False,
) -> OutreachResult:
    """Generate pitch emails and either queue them (dry-run) or dispatch.

    Dry-run mode only builds the queued rows — it never touches the outreach
    ledger, so re-running is safe. With ``persist_to_ledger=True`` (the GUI
    approval flow), each dry-run pitch is also stored as an
    ``OutreachEmail(status="queued")`` row; leads that already have a pending
    queue row are skipped, which keeps re-runs idempotent. Dispatch mode stops
    the whole batch on a rate-limit error; per-business ``ValueError``s
    (already sent / opted out) skip that lead and continue.
    """
    result = OutreachResult()
    for b in businesses:
        if not b.contact_email:
            if log:
                log(f"  - {b.name} [{b.slug}] — skipped (no contact email)")
            result.skipped.append((b.slug, "no contact email"))
            continue
        try:
            pitch = generate_pitch_email(
                b.name, b.audit_flags_list() or ["No Website"], b.preview_url or ""
            )
        except LLMGenerationError as exc:
            if log:
                log(f"  ! {b.name} [{b.slug}] — pitch generation failed: {exc}")
            result.failed.append((b.slug, str(exc)))
            continue

        if dry_run:
            if persist_to_ledger and db.scalar(
                select(OutreachEmail.id).where(
                    OutreachEmail.business_id == b.id,
                    OutreachEmail.status == "queued",
                )
            ) is not None:
                if log:
                    log(f"  - {b.name} [{b.slug}] — skipped (already queued)")
                result.skipped.append((b.slug, "already queued"))
                continue
            if persist_to_ledger:
                db.add(
                    OutreachEmail(
                        business_id=b.id,
                        recipient=b.contact_email,
                        subject=pitch["subject"],
                        body=pitch["body"],
                        status="queued",
                    )
                )
                db.commit()
            result.queued.append(
                {
                    "business_slug": b.slug,
                    "name": b.name,
                    "email": b.contact_email,
                    "subject": pitch["subject"],
                    "body": pitch["body"],
                    "preview_url": b.preview_url or "",
                    "generated_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }
            )
            if log:
                log(f"  ~ {b.name} [{b.slug}] — queued for review")
        else:
            try:
                send_outreach_email(
                    to_email=b.contact_email,
                    subject=pitch["subject"],
                    body_text=pitch["body"],
                    business_slug=b.slug,
                )
            except OutreachRateLimitError as exc:
                if log:
                    log(f"  ! rate limit hit — stopping dispatch: {exc}")
                return result  # stop the whole batch (spec 6.2)
            except ValueError as exc:
                if log:
                    log(f"  - {b.name} [{b.slug}] — skipped ({exc})")
                result.skipped.append((b.slug, str(exc)))
                continue
            b.stage = DealStage.CONTACTED
            db.commit()
            result.sent.append(b.slug)
            if log:
                log(f"  ✓ {b.name} [{b.slug}] — dispatched to {b.contact_email}")

    return result


def outreach_candidates(db) -> list[Business]:
    """Bad-site leads with a preview URL, an email, and no prior send."""
    sent_ids = {
        row[0]
        for row in db.query(OutreachEmail.business_id).filter_by(status="sent").all()
    }
    targets = db.query(Business).filter(
        Business.is_bad_site.is_(True),
        Business.preview_url.is_not(None),
        Business.contact_email.is_not(None),
        Business.opted_out == False,  # noqa: E712 — SQL boolean comparison
    ).all()
    return [b for b in targets if b.id not in sent_ids]


@dataclass
class PipelineRun:
    """Structured result of a full end-to-end pipeline run (GUI trigger)."""

    found: int = 0
    new_leads: list[dict] = field(default_factory=list)  # name/slug/site
    audited: int = 0
    generated: list[str] = field(default_factory=list)  # slugs with fresh copy
    copy_failed: list[tuple[str, str]] = field(default_factory=list)  # (slug, error)
    deployed: list[dict] = field(default_factory=list)  # {slug, url}
    deploy_failed: list[tuple[str, str]] = field(default_factory=list)  # (slug, error)
    outreach_queued: int = 0
    outreach_skipped: list[tuple[str, str]] = field(default_factory=list)  # (slug, reason)


def run_pipeline(
    db,
    query: str,
    limit: int | None = None,
    auto_deploy: bool = False,
    *,
    log: Optional[Log] = None,
) -> PipelineRun:
    """Discover → audit → generate copy → (deploy) → queue outreach.

    Mirrors the ``run.py pipeline`` command for the dashboard, with one
    deliberate difference: a single lead's LLM or R2 failure is recorded in
    ``copy_failed`` / ``deploy_failed`` instead of aborting the whole run.
    Outreach pitches are queued (never sent) into the approval ledger for this
    run's leads that end up with both a preview URL and a contact email.
    """
    run = PipelineRun()

    # 1. Discover
    found, created = discover(db, query, limit)
    run.found = found
    for b in created:
        site = (
            "no website"
            if b.no_website
            else (b.current_website or "directory listing only")
        )
        run.new_leads.append(
            {"id": b.id, "name": b.name, "slug": b.slug, "site": site}
        )
        if log:
            log(f"  + {b.name} [{b.slug}] — {site}")

    # 2. Audit pending leads with websites
    audited = audit_pending(db, log=log)
    run.audited = len(audited)

    # 3. Generate landing copy for every lead that needs one (no site or bad site)
    targets = db.query(Business).filter(
        Business.generated_copy.is_(None),
        (Business.no_website.is_(True)) | (Business.is_bad_site.is_(True)),
    ).all()
    for b in targets:
        try:
            generate_for(db, b)
            run.generated.append(b.slug)
            if log:
                log(f"  ✓ {b.name} [{b.slug}] — copy generated")
        except LLMGenerationError as exc:
            run.copy_failed.append((b.slug, str(exc)))
            if log:
                log(f"  ✗ {b.name} [{b.slug}] — copy failed: {exc}")

    # 4. Optional deploy of every lead that now has stored copy
    if auto_deploy:
        ready = db.query(Business).filter(
            Business.generated_copy.is_not(None)
        ).all()
        for b in ready:
            try:
                url = deploy_business(db, b)
                run.deployed.append({"slug": b.slug, "url": url})
                if log:
                    log(f"  ✓ {b.name} [{b.slug}] → {url}")
            except Exception as exc:  # noqa: BLE001 — bulk run must survive any per-lead failure
                run.deploy_failed.append((b.slug, str(exc)))
                if log:
                    log(f"  ✗ {b.name} [{b.slug}] — deploy failed: {exc}")

    # 5. Queue outreach for this run's leads that have a preview + email
    outreach_targets = [b for b in targets if b.preview_url and b.contact_email]
    if outreach_targets:
        result = outreach_batch(
            db, outreach_targets, dry_run=True, persist_to_ledger=True
        )
        run.outreach_queued = len(result.queued)
        run.outreach_skipped = list(result.skipped)

    return run
