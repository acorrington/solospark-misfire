"""Phase 7 — SoloSpark CLI: orchestration, pipeline & safety controls.

Commands:
    python run.py discover --query "plumbers in Eugene OR" --limit 20
    python run.py audit
    python run.py generate --slug "acme-plumbing"
    python run.py deploy --all
    python run.py outreach [--dry-run | --send]
    python run.py pipeline --query "electricians in Roseburg OR" --auto-deploy

The pipeline logic itself lives in app.pipeline (shared with the web
dashboard); this module is a thin CLI wrapper that adds argument parsing,
console output, and the dry-run CSV review queue.

Safety: the SQLite ledger (Business.place_id + slug uniqueness) prevents
duplicate discovery, and the OutreachEmail ledger plus per-business "already
sent" check prevent repeated email outreach. Blocked directories (.gov, .edu,
Yelp, YellowPages, ...) are filtered at ingest time in app.scanner.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

from app.llm_engine import LLMGenerationError
from app.models import Business, get_session_factory
from app.pipeline import (
    audit_pending,
    deploy_many,
    discover,
    generate_for,
    outreach_batch,
    outreach_candidates,
)

OUTREACH_QUEUE_CSV = Path("outreach_queue.csv")

QUEUE_COLUMNS = [
    "business_slug",
    "name",
    "email",
    "subject",
    "body",
    "preview_url",
    "generated_at",
]


def _append_queue_csv(rows: list[dict]) -> None:
    """Append dry-run pitch rows to the local review queue (CSV)."""
    new = not OUTREACH_QUEUE_CSV.exists()
    with OUTREACH_QUEUE_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUEUE_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _print_new_leads(created: list[Business]) -> None:
    for b in created:
        if b.no_website:
            site = "no website"
        elif b.current_website:
            site = b.current_website
        else:
            site = "blocked directory listing only"
        print(f"  + {b.name} [{b.slug}] — {site}")


# ── command handlers ─────────────────────────────────────────────────────────


def cmd_discover(args) -> None:
    db = get_session_factory()()
    try:
        found, created = discover(db, args.query, args.limit)
    finally:
        db.close()
    print(f"Discovered {found} place(s); {len(created)} new lead(s) added to the ledger.")
    _print_new_leads(created)


def cmd_audit(args) -> None:
    db = get_session_factory()()
    try:
        audited = audit_pending(db, log=print)
    finally:
        db.close()
    if not audited:
        print("Nothing to audit — no DISCOVERED leads with a website.")
    else:
        print(f"Audited {len(audited)} lead(s).")


def cmd_generate(args) -> None:
    db = get_session_factory()()
    try:
        b = db.query(Business).filter_by(slug=args.slug).first()
        if b is None:
            raise LookupError(f"Unknown business slug: {args.slug!r}")
        generate_for(db, b)
        print(
            f"Generated copy for {b.name} [{b.slug}]. "
            f"Publish with: python run.py deploy --slug {b.slug}"
        )
    finally:
        db.close()


def cmd_deploy(args) -> None:
    if not args.all and not args.slug:
        raise ValueError("deploy requires --all or --slug")
    db = get_session_factory()()
    try:
        if args.slug:
            b = db.query(Business).filter_by(slug=args.slug).first()
            if b is None:
                raise LookupError(f"Unknown business slug: {args.slug!r}")
            targets = [b]
        else:
            targets = (
                db.query(Business)
                .filter(Business.generated_copy.is_not(None))
                .all()
            )
        results = deploy_many(db, targets, log=print)
    finally:
        db.close()
    deployed = sum(1 for _, url in results if url is not None)
    print(f"Deployed {deployed} site(s).")


def cmd_outreach(args) -> None:
    dry_run = not args.send  # dry-run is the default mode (spec 6.2)
    db = get_session_factory()()
    try:
        targets = outreach_candidates(db)
        if not targets:
            print("No outreach candidates (bad-site leads with a preview URL and email).")
            return
        mode = "dry-run queue" if dry_run else "auto-dispatch"
        print(f"{len(targets)} candidate(s) — {mode}")
        result = outreach_batch(db, targets, dry_run=dry_run, log=print)
    finally:
        db.close()
    queued, sent, skipped, failed = result.counts
    summary = f"Outreach complete: {queued} queued, {sent} sent, {skipped} skipped, {failed} failed."
    if dry_run:
        if result.queued:
            _append_queue_csv(result.queued)
        summary += f" Review the queue in {OUTREACH_QUEUE_CSV}, then run with --send to dispatch."
    print(summary)


def cmd_pipeline(args) -> None:
    db = get_session_factory()()
    try:
        found, created = discover(db, args.query, args.limit)
        print(f"Discovered {found} place(s); {len(created)} new lead(s) added to the ledger.")
        _print_new_leads(created)

        audited_n = len(audit_pending(db, log=print))

        targets = db.query(Business).filter(
            Business.generated_copy.is_(None),
            (Business.no_website.is_(True)) | (Business.is_bad_site.is_(True)),
        ).all()
        generated = 0
        for b in targets:
            generate_for(db, b)
            generated += 1
            print(f"  ✓ copy: {b.name} [{b.slug}]")

        deployed = 0
        if args.auto_deploy:
            with_copy = (
                db.query(Business).filter(Business.generated_copy.is_not(None)).all()
            )
            results = deploy_many(db, with_copy, log=print)
            deployed = sum(1 for _, url in results if url is not None)

        queued = sent = 0
        candidates = [b for b in targets if b.preview_url and b.contact_email]
        if candidates:
            result = outreach_batch(db, candidates, dry_run=True, log=print)
            queued, sent, _, _ = result.counts
            if result.queued:
                _append_queue_csv(result.queued)
        else:
            print("No new outreach queued (leads need a deployed preview + contact email).")

        print(
            f"Pipeline complete: {len(created)} new lead(s), {audited_n} audited, "
            f"{generated} mockup(s) generated, {deployed} deployed, "
            f"{queued} outreach queued."
        )
    finally:
        db.close()


# ── argument parsing ─────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="SoloSpark AI Agency Platform — CLI orchestration & pipeline.",
        epilog=(
            "Safety: discovery is ledger-deduped (Business.place_id), outreach is "
            "ledger-deduped (OutreachEmail) and rate-limited. Dry-run outreach is "
            "the default mode; pass --send to actually dispatch emails."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_disc = sub.add_parser("discover", help="Search Google Places and ingest new leads")
    p_disc.add_argument("--query", required=True, help='e.g. "plumbers in Eugene OR"')
    p_disc.add_argument(
        "--limit", type=int, default=None, help="max results (default: API limit)"
    )
    p_disc.set_defaults(func=cmd_discover)

    p_audit = sub.add_parser("audit", help="Audit all pending DISCOVERED leads with websites")
    p_audit.set_defaults(func=cmd_audit)

    p_gen = sub.add_parser("generate", help="Generate landing copy for one lead")
    p_gen.add_argument("--slug", required=True, help="business slug")
    p_gen.set_defaults(func=cmd_generate)

    p_dep = sub.add_parser("deploy", help="Render + publish preview sites to R2")
    grp = p_dep.add_mutually_exclusive_group()
    grp.add_argument("--all", action="store_true", help="deploy every lead with generated copy")
    grp.add_argument("--slug", default=None, help="deploy a single lead by slug")
    p_dep.set_defaults(func=cmd_deploy)

    p_out = sub.add_parser(
        "outreach", help="Generate pitch emails (dry-run queue by default)"
    )
    grp2 = p_out.add_mutually_exclusive_group()
    grp2.add_argument(
        "--dry-run", action="store_true", help="build the CSV review queue only (default)"
    )
    grp2.add_argument("--send", action="store_true", help="actually dispatch emails")
    p_out.set_defaults(func=cmd_outreach)

    p_pipe = sub.add_parser("pipeline", help="End-to-end: discover → audit → mockups → outreach")
    p_pipe.add_argument("--query", required=True, help='e.g. "electricians in Roseburg OR"')
    p_pipe.add_argument("--limit", type=int, default=None, help="max places to ingest")
    p_pipe.add_argument(
        "--auto-deploy", action="store_true", help="also publish mockups to R2"
    )
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so Unicode output (✓, →, •)
    # never crashes the CLI.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — best effort only
                pass

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, LookupError, LLMGenerationError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"error: network failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
