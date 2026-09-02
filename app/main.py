"""Phase 5 — FastAPI web management dashboard & split-screen studio.

Endpoints:
    GET  /                            lead pipeline table (stage/flag/rating filters)
    GET  /discover                    discover new leads from Google Places
    POST /api/discover                run a Places search and ingest new leads
    POST /api/audit                   audit all pending DISCOVERED leads with sites
    POST /api/deploy                  bulk render + publish previews to R2
    POST /api/deploy/{business_id}    render + publish one lead's preview to R2
    GET  /review/{business_id}        split-screen studio (info + pitch form | iframe)
    POST /generate/{business_id}      LLM copy → Jinja2 compile → R2 deploy
    POST /regenerate-prompt/{id}      re-generate with a custom prompt adjustment
    POST /refine/{business_id}        fine-tune the existing site from instructions only
    DELETE /site/{business_id}        remove deployed objects + reset preview state
    POST /outreach/send/{business_id} dispatch the reviewed email → stage=contacted
    GET  /pipeline                    one-click full pipeline trigger page
    POST /api/pipeline/run            discover → audit → copy → (deploy) → queue outreach
    GET  /outreach                    DB-backed pitch queue with approval actions
    POST /api/outreach/generate       draft pitches into the review queue (ledger)
    POST /api/outreach/send-selected  approve + dispatch selected queued pitches
    GET  /api/outreach/unsubscribe    CAN-SPAM opt-out (marks leads lost)
    POST /api/billing/webhook         Stripe events (activation + killswitch)
    POST /api/forms/submit            lead-form relay from deployed sites

The app factory accepts injectable clients (``llm_client``, ``s3_client``,
``outreach_sender``) so every endpoint is testable without live services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from datetime import datetime, timezone

import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import billing, forms, pipeline
from .config import get_settings
from .deployer import (
    check_html_syntax,
    delete_site_objects,
    deploy_site_assets,
    deploy_to_r2,
    render_site_html,
)
from .llm_engine import (
    LLMGenerationError,
    apply_explicit_edits,
    enforce_named_color,
    generate_landing_copy,
    refine_landing_copy,
)
from .site_ref import asset_filename, fetch_image_bytes, scrape_site_reference
from .models import Business, DealStage, OutreachEmail, get_db
from .outreach import outreach_router

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

# Monthly site plan price in cents (Stripe Checkout).
SITE_PLAN_PRICE_CENTS = 9900


# Pydantic request bodies live at module level: with PEP 563 string
# annotations, classes defined inside ``create_app`` are unresolvable from
# module globals and FastAPI would treat them as query parameters.
class RegenerateRequest(BaseModel):
    prompt: str = ""


class RefineRequest(BaseModel):
    """Instructions for fine-tuning an already-generated site (no re-scrape)."""

    prompt: str = ""


class OutreachPayload(BaseModel):
    subject: str
    body: str


class DiscoverRequest(BaseModel):
    query: str
    limit: int | None = Field(default=None, ge=1, le=100)


class DeployRequest(BaseModel):
    ids: list[int] | None = None


class SendSelectedRequest(BaseModel):
    ids: list[int]


class PipelineRunRequest(BaseModel):
    query: str
    limit: int | None = Field(default=None, ge=1, le=100)
    auto_deploy: bool = False


def _city_from_address(address: str | None) -> str:
    """Best-effort city extraction: '123 Main St, Eugene, OR 97401' → 'Eugene'."""
    if not address:
        return ""
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 2:
        return parts[-2]
    return ""


def stripe_payment_link(business: Business) -> str | None:
    """Build a Stripe Checkout subscription link, or ``None`` when unconfigured.

    Never raises — the dashboard degrades gracefully to a "not configured" hint.
    """
    settings = get_settings()
    if not settings.stripe_secret_key:
        return None
    try:
        import stripe

        stripe.api_key = settings.stripe_secret_key
        base = f"http://localhost:{settings.port}"
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=business.contact_email,
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": SITE_PLAN_PRICE_CENTS,
                        "recurring": {"interval": "month"},
                        "product_data": {
                            "name": f"SoloSpark Site — {business.name}"
                        },
                    },
                    "quantity": 1,
                }
            ],
            success_url=f"{base}/review/{business.id}",
            cancel_url=f"{base}/review/{business.id}",
        )
        return session.url
    except Exception:  # noqa: BLE001 — network/API failures are non-fatal here
        return None


def create_app(
    llm_client=None,
    s3_client=None,
    outreach_sender: Callable[..., dict] | None = None,
) -> FastAPI:
    """Build the dashboard app with optional injected service clients."""
    app = FastAPI(title="SoloSpark Lead Pipeline")
    app.state.llm_client = llm_client
    app.state.s3_client = s3_client
    app.state.outreach_sender = outreach_sender
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(outreach_router)

    def _sender() -> Callable[..., dict]:
        if app.state.outreach_sender is not None:
            return app.state.outreach_sender
        # Lazy import: Phase 6 module; P5 stays runnable before it exists.
        from .outreach import send_outreach_email

        return send_outreach_email

    def _get_business_or_404(db: Session, business_id: int) -> Business:
        business = db.get(Business, business_id)
        if business is None:
            raise HTTPException(status_code=404, detail="Business not found")
        return business

    # ── GET / — pipeline table with filters ──────────────────────────────────

    @app.get("/")
    def dashboard(
        request: Request,
        stage: str | None = Query(default=None),
        flag: str | None = Query(default=None),
        min_rating: float | None = Query(default=None, ge=0, le=5),
        db: Session = Depends(get_db),
    ):
        query = select(Business)
        if stage:
            try:
                query = query.where(Business.stage == DealStage(stage))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=f"Unknown stage: {stage}"
                ) from exc
        if min_rating is not None:
            query = query.where(Business.rating >= min_rating)
        businesses = list(db.scalars(query).all())
        if flag:
            businesses = [b for b in businesses if flag in b.audit_flags_list()]

        all_businesses = list(db.scalars(select(Business)).all())
        flags = sorted({f for b in all_businesses for f in b.audit_flags_list()})

        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "businesses": businesses,
                "stages": [s.value for s in DealStage],
                "flags": flags,
                "stage_filter": stage or "",
                "flag_filter": flag or "",
                "min_rating_filter": min_rating if min_rating is not None else "",
            },
        )

    # ── GET /discover — new-lead discovery page ──────────────────────────────

    @app.get("/discover")
    def discover_page(request: Request):
        return templates.TemplateResponse(request, "discover.html", {})

    # ── POST /api/discover — Places search + ledger ingest ───────────────────

    @app.post("/api/discover")
    def discover(payload: DiscoverRequest, db: Session = Depends(get_db)):
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        try:
            found, created = pipeline.discover(db, query, limit=payload.limit)
        except ValueError as exc:  # e.g. PLACES_API_KEY not configured
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except requests.RequestException as exc:  # upstream network failure
            raise HTTPException(
                status_code=502, detail=f"Places lookup failed: {exc}"
            ) from exc
        return {
            "status": "ok",
            "found": found,
            "new_leads": [
                {
                    "id": b.id,
                    "name": b.name,
                    "slug": b.slug,
                    "category": b.category or "",
                    "address": b.address or "",
                    "phone": b.phone or "",
                    "website": b.current_website or "",
                }
                for b in created
            ],
        }

    # ── GET /pipeline — one-click full pipeline trigger page ──────────────────

    @app.get("/pipeline")
    def pipeline_page(request: Request):
        return templates.TemplateResponse(request, "pipeline.html", {})

    # ── POST /api/pipeline/run — discover → audit → copy → (deploy) → queue ──

    @app.post("/api/pipeline/run")
    def run_pipeline_api(payload: PipelineRunRequest, db: Session = Depends(get_db)):
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        try:
            run = pipeline.run_pipeline(
                db, query, limit=payload.limit, auto_deploy=payload.auto_deploy
            )
        except ValueError as exc:  # e.g. PLACES_API_KEY not configured
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except requests.RequestException as exc:  # upstream network failure
            raise HTTPException(
                status_code=502, detail=f"Places lookup failed: {exc}"
            ) from exc
        return {
            "status": "ok",
            "found": run.found,
            "new_leads": run.new_leads,
            "audited": run.audited,
            "generated": run.generated,
            "copy_failed": [
                {"slug": slug, "error": error} for slug, error in run.copy_failed
            ],
            "deployed": run.deployed,
            "deploy_failed": [
                {"slug": slug, "error": error} for slug, error in run.deploy_failed
            ],
            "outreach_queued": run.outreach_queued,
            "outreach_skipped": [
                {"slug": slug, "reason": reason}
                for slug, reason in run.outreach_skipped
            ],
        }

    # ── POST /api/audit — audit all pending DISCOVERED leads with websites ───

    @app.post("/api/audit")
    def run_audit(db: Session = Depends(get_db)):
        try:
            audited = pipeline.audit_pending(db)
        except requests.RequestException as exc:  # upstream network failure
            raise HTTPException(
                status_code=502, detail=f"Audit failed: {exc}"
            ) from exc
        return {
            "status": "ok",
            "count": len(audited),
            "audited": [
                {
                    "id": b.id,
                    "name": b.name,
                    "slug": b.slug,
                    "is_bad_site": bool(b.is_bad_site),
                    "flags": list(b.audit_flags_list()),
                }
                for b in audited
            ],
        }

    # ── POST /api/deploy — bulk render + publish previews to R2 ──────────────

    @app.post("/api/deploy")
    def deploy_bulk(payload: DeployRequest, db: Session = Depends(get_db)):
        query = db.query(Business)
        if payload.ids:
            query = query.filter(Business.id.in_(payload.ids))
        try:
            results = pipeline.deploy_many(db, query.all())
        except ValueError as exc:  # e.g. R2 not configured
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ok",
            "deployed": sum(1 for _, url in results if url),
            "results": [
                {"id": b.id, "slug": b.slug, "preview_url": url}
                for b, url in results
            ],
        }

    # ── POST /api/deploy/{business_id} — deploy one lead's preview ───────────

    @app.post("/api/deploy/{business_id}")
    def deploy_one(business_id: int, db: Session = Depends(get_db)):
        business = _get_business_or_404(db, business_id)
        if not business.generated_copy:
            raise HTTPException(
                status_code=400, detail="No generated copy — run Generate first"
            )
        try:
            preview_url = pipeline.deploy_business(db, business)
        except ValueError as exc:  # e.g. R2 credentials not configured
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ok",
            "id": business.id,
            "slug": business.slug,
            "preview_url": preview_url,
        }

    # ── GET /review/{business_id} — split-screen studio ──────────────────────

    @app.get("/review/{business_id}")
    def review(business_id: int, request: Request, db: Session = Depends(get_db)):
        business = _get_business_or_404(db, business_id)
        copy = business.generated_copy_dict() or {}
        return templates.TemplateResponse(
            request,
            "review.html",
            {
                "business": business,
                "flags": business.audit_flags_list(),
                "pitch_subject": copy.get("pitch_subject", ""),
                "pitch_body": copy.get("pitch_body", ""),
                "payment_link": stripe_payment_link(business),
            },
        )

    # ── POST /generate/{business_id} — LLM → Jinja2 → R2 ─────────────────────

    def _scrape_reference(business: Business) -> dict[str, Any]:
        """Best-effort scrape of the business's existing website.

        Returns ``{}`` when there is no URL or anything goes wrong — a failed
        scrape must never block generation.
        """
        url = (business.current_website or "").strip()
        if not url:
            return {}
        try:
            return scrape_site_reference(url) or {}
        except Exception:  # noqa: BLE001 — reference is strictly best-effort
            return {}

    def _attach_site_assets(
        business: Business, ref: dict[str, Any]
    ) -> dict[str, Any]:
        """Download logo/hero/about/gallery images from the reference site → R2.

        Returns a partial copy-dict of relative asset paths (e.g.
        ``{"logo_url": "assets/logo.png"}``). Each download/upload failure is
        skipped silently; an empty dict means "no fresh assets".
        """
        if not ref:
            return {}
        plan: list[tuple[str, str, str]] = []  # (prefix, url, copy field)
        if ref.get("logo_url"):
            plan.append(("logo", ref["logo_url"], "logo_url"))
        if ref.get("hero_image_url"):
            plan.append(("hero", ref["hero_image_url"], "hero_image_url"))
        about = (ref.get("about_images") or [])[:1]
        if about:
            plan.append(("about", about[0], "about_images"))
        # Gallery: up to three extra photos for the dedicated gallery section,
        # skipping anything already planned (hero/about) so no image uploads
        # twice. Gated on the key existing — old reference fixtures without it
        # add zero puts.
        used_urls = {url for _prefix, url, _field in plan}
        gallery_candidates = [
            u
            for u in (ref.get("content_images") or [])
            if isinstance(u, str) and u.strip() and u not in used_urls
        ]
        for i, url in enumerate(gallery_candidates[:3], start=1):
            plan.append((f"gallery-{i}", url, "gallery_images"))

        assets: list[dict[str, Any]] = []
        fields: list[str] = []
        for prefix, url, field in plan:
            try:
                fetched = fetch_image_bytes(url)
            except Exception:  # noqa: BLE001 — one bad image must not block the rest
                continue
            if not fetched:
                continue
            data, ctype = fetched
            assets.append(
                {
                    "key": asset_filename(url, ctype, prefix=prefix),
                    "data": data,
                    "content_type": ctype,
                }
            )
            fields.append(field)

        if not assets:
            return {}
        try:
            relative = deploy_site_assets(
                business.slug, assets, s3_client=app.state.s3_client
            )
        except Exception:  # noqa: BLE001 — asset upload failure is non-fatal
            return {}
        if len(relative) != len(fields):
            return {}

        merged: dict[str, Any] = {}
        for field, path in zip(fields, relative):
            if field in ("about_images", "gallery_images"):
                merged.setdefault(field, []).append(path)
            else:
                merged[field] = path
        return merged

    def _merge_assets(
        copy: dict[str, Any], business: Business, ref: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach freshly scraped assets to *copy*.

        Falls back to previously persisted asset references (the objects are
        still in the bucket) when a fresh scrape yields nothing for a field.
        """
        old = business.generated_copy_dict() or {}
        merged = dict(copy)
        fresh = _attach_site_assets(business, ref)
        for field in ("logo_url", "hero_image_url"):
            value = fresh.get(field) or old.get(field) or ""
            if value:
                merged[field] = value
        about = fresh.get("about_images") or old.get("about_images") or []
        if about:
            merged["about_images"] = list(about)[:2]
        gallery = fresh.get("gallery_images") or old.get("gallery_images") or []
        if gallery:
            merged["gallery_images"] = list(gallery)[:4]
        return merged

    def _render_and_deploy(
        business: Business, copy: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Render the (already syntax-QA'd) copy, gate the HTML, publish to R2.

        The model-based syntax QA pass runs inside ``generate_landing_copy`` /
        ``refine_landing_copy`` — before the deterministic safety nets — so it
        cannot revert an operator's explicit edit here. The rendered page must
        still clear the deterministic ``check_html_syntax`` gate before anything
        is uploaded. Returns ``(preview_url, copy_used)`` so callers persist
        exactly the copy that went live.
        """
        html = render_site_html(
            {
                "name": business.name,
                "slug": business.slug,
                "category": business.category or "",
                "address": business.address or "",
                "phone": business.phone or "",
                "city": _city_from_address(business.address),
                "rating": business.rating or None,
                "review_count": business.review_count or None,
            },
            copy,
        )
        problems = check_html_syntax(html)
        if problems:
            raise HTTPException(
                status_code=500,
                detail="Rendered site failed the syntax check and was NOT "
                "uploaded: " + "; ".join(problems),
            )
        try:
            preview_url = deploy_to_r2(business.slug, html, s3_client=app.state.s3_client)
        except ValueError as exc:  # R2 credentials not configured
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return preview_url, copy

    @app.post("/generate/{business_id}")
    def generate(business_id: int, db: Session = Depends(get_db)):
        business = _get_business_or_404(db, business_id)
        reference = _scrape_reference(business)
        try:
            copy = generate_landing_copy(
                business.name,
                business.category or "",
                _city_from_address(business.address),
                client=app.state.llm_client,
                site_reference=reference or None,
            )
        except LLMGenerationError as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM generation failed: {exc}"
            ) from exc
        copy = _merge_assets(copy, business, reference)
        preview_url, copy = _render_and_deploy(business, copy)
        business.set_generated_copy(copy)
        business.preview_url = preview_url
        business.stage = DealStage.MOCKUP_READY
        db.commit()
        return {"status": "generated", "preview_url": preview_url}

    # ── POST /regenerate-prompt/{business_id} — custom prompt re-generation ──

    @app.post("/regenerate-prompt/{business_id}")
    def regenerate_prompt(
        business_id: int, payload: RegenerateRequest, db: Session = Depends(get_db)
    ):
        business = _get_business_or_404(db, business_id)
        reference = _scrape_reference(business)
        try:
            copy = generate_landing_copy(
                business.name,
                business.category or "",
                _city_from_address(business.address),
                extra_instructions=payload.prompt,
                client=app.state.llm_client,
                site_reference=reference or None,
            )
        except LLMGenerationError as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM generation failed: {exc}"
            ) from exc
        copy = _merge_assets(copy, business, reference)
        # Safety net: if the operator asked for a named color but the model
        # re-emitted the old palette (or dropped it), apply the requested one.
        previous_brand = (business.generated_copy_dict() or {}).get("brand")
        enforce_named_color(copy, payload.prompt, previous_brand)
        # Second safety net: quoted replacement values are applied verbatim.
        apply_explicit_edits(copy, payload.prompt)
        preview_url, copy = _render_and_deploy(business, copy)
        business.set_generated_copy(copy)
        business.preview_url = preview_url
        if business.stage not in (
            DealStage.PITCH_APPROVED,
            DealStage.CONTACTED,
            DealStage.REPLIED,
            DealStage.WON,
        ):
            business.stage = DealStage.MOCKUP_READY
        db.commit()
        return {"status": "regenerated", "preview_url": preview_url}

    # ── POST /refine/{business_id} — fine-tune existing site from instructions ─
    #
    # Unlike /generate and /regenerate-prompt this never scrapes the customer's
    # website: the LLM edits the persisted copy in place, so logos/hero images
    # and everything else already deployed are reused as-is. This is the
    # "update the current site from my prompt text" path — fine-tuning before
    # the customer pays.

    @app.post("/refine/{business_id}")
    def refine_site(
        business_id: int, payload: RefineRequest, db: Session = Depends(get_db)
    ):
        business = _get_business_or_404(db, business_id)
        current = business.generated_copy_dict() or {}
        if not current:
            raise HTTPException(
                status_code=400,
                detail="No generated site to refine — generate it first",
            )
        prompt = (payload.prompt or "").strip()
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Prompt is required for fine-tuning"
            )
        try:
            refined = refine_landing_copy(
                current,
                business.name,
                business.category or "",
                _city_from_address(business.address),
                prompt,
                client=app.state.llm_client,
            )
        except LLMGenerationError as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM refinement failed: {exc}"
            ) from exc

        # Persisted asset references stay in the bucket — carry them over so
        # the re-render keeps pointing at the same logo/hero/about objects.
        for field in ("logo_url", "hero_image_url", "about_images"):
            if current.get(field):
                refined[field] = current[field]

        preview_url, refined = _render_and_deploy(business, refined)
        business.set_generated_copy(refined)
        business.preview_url = preview_url
        if business.stage not in (
            DealStage.PITCH_APPROVED,
            DealStage.CONTACTED,
            DealStage.REPLIED,
            DealStage.WON,
        ):
            business.stage = DealStage.MOCKUP_READY
        db.commit()
        return {"status": "refined", "preview_url": preview_url}

    # ── DELETE /site/{business_id} — remove deployed site + reset state ───────

    @app.delete("/site/{business_id}")
    def delete_site(business_id: int, db: Session = Depends(get_db)):
        business = _get_business_or_404(db, business_id)
        removed = 0
        if business.preview_url:
            try:
                removed = delete_site_objects(
                    business.slug, s3_client=app.state.s3_client
                )
            except Exception as exc:  # noqa: BLE001 — surface R2 failures clearly
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not remove objects from R2: {exc}",
                ) from exc

        business.generated_copy = None
        business.preview_url = None
        # A removed mockup means the lead is back to its audited state — but
        # never roll a real sales stage (contacted, replied, won…) backwards.
        if business.stage == DealStage.MOCKUP_READY:
            business.stage = DealStage.AUDITED
        db.commit()
        return {"status": "deleted", "removed_objects": removed}

    # ── POST /outreach/send/{business_id} — dispatch reviewed email ──────────

    @app.post("/outreach/send/{business_id}")
    def send_outreach(
        business_id: int, payload: OutreachPayload, db: Session = Depends(get_db)
    ):
        business = _get_business_or_404(db, business_id)
        if not business.contact_email:
            raise HTTPException(
                status_code=400, detail="No contact email on file for this business"
            )
        try:
            result = _sender()(
                to_email=business.contact_email,
                subject=payload.subject,
                body_text=payload.body,
                business_slug=business.slug,
            )
        except Exception as exc:  # noqa: BLE001 — map outreach failures to HTTP
            if type(exc).__name__ == "OutreachRateLimitError":
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            raise HTTPException(
                status_code=502, detail=f"Outreach dispatch failed: {exc}"
            ) from exc
        business.stage = DealStage.CONTACTED
        db.commit()
        response: dict[str, Any] = {"status": "contacted"}
        if isinstance(result, dict):
            # Endpoint status wins; sender extras (e.g. message id) merge in.
            for key, value in result.items():
                response.setdefault(key, value)
        return response

    # ── GET /outreach — DB-backed pitch queue with approval actions ───────────

    @app.get("/outreach")
    def outreach_queue_page(request: Request, db: Session = Depends(get_db)):
        rows = (
            db.query(OutreachEmail)
            .filter(OutreachEmail.status == "queued")
            .order_by(OutreachEmail.id.desc())
            .all()
        )
        businesses: dict[int, Business] = {}
        if rows:
            businesses = {
                b.id: b
                for b in db.query(Business).filter(
                    Business.id.in_([r.business_id for r in rows])
                ).all()
            }
        return templates.TemplateResponse(
            request,
            "outreach.html",
            {
                "request": request,
                "rows": [
                    {"email": r, "business": businesses.get(r.business_id)}
                    for r in rows
                ],
                "sent_count": db.query(OutreachEmail).filter_by(status="sent").count(),
                "failed_count": db.query(OutreachEmail).filter_by(status="failed").count(),
            },
        )

    # ── POST /api/outreach/generate — build pitches into the review queue ─────

    @app.post("/api/outreach/generate")
    def generate_outreach_queue(db: Session = Depends(get_db)):
        candidates = pipeline.outreach_candidates(db)
        result = pipeline.outreach_batch(
            db, candidates, dry_run=True, persist_to_ledger=True
        )
        return {
            "status": "ok",
            "queued": len(result.queued),
            "skipped": [
                {"slug": slug, "reason": reason} for slug, reason in result.skipped
            ],
            "failed": [
                {"slug": slug, "error": error} for slug, error in result.failed
            ],
        }

    # ── POST /api/outreach/send-selected — approve + dispatch queued pitches ──

    @app.post("/api/outreach/send-selected")
    def send_selected(payload: SendSelectedRequest, db: Session = Depends(get_db)):
        sent: list[str] = []
        dry_run: list[str] = []
        skipped: list[dict[str, Any]] = []
        rate_limited = False
        for row_id in payload.ids:
            row = db.get(OutreachEmail, row_id)
            if row is None:
                skipped.append({"id": row_id, "reason": "not found"})
                continue
            if row.status != "queued":
                skipped.append({"id": row_id, "reason": f"already {row.status}"})
                continue
            business = db.get(Business, row.business_id)
            try:
                result = _sender()(
                    to_email=row.recipient,
                    subject=row.subject or "",
                    body_text=row.body or "",
                    business_slug=business.slug if business else "",
                    ledger_row=row,
                )
            except Exception as exc:  # noqa: BLE001 — map outreach failures
                if type(exc).__name__ == "OutreachRateLimitError":
                    rate_limited = True
                    break  # spec 6.2: a rate-limit stop halts the whole batch
                skipped.append({"id": row_id, "reason": str(exc)})
                continue
            status = result.get("status") if isinstance(result, dict) else None
            if status == "sent" and business is not None:
                # Belt-and-braces: the real sender already updated the ledger
                # row via ``ledger_row``; injected senders may not have.
                row.status = "sent"
                if row.sent_at is None:
                    row.sent_at = datetime.now(timezone.utc)
                business.stage = DealStage.CONTACTED
                db.commit()
                sent.append(business.slug)
            elif status == "dry_run":
                dry_run.append(row.recipient)
        return {
            "status": "ok",
            "sent": sent,
            "dry_run": dry_run,
            "skipped": skipped,
            "rate_limited": rate_limited,
        }

    # ── POST /api/billing/webhook — Stripe event relay ───────────────────────
    # Stripe payloads vary per event type, so the body is parsed as raw JSON
    # instead of a fixed Pydantic model.

    @app.post("/api/billing/webhook")
    async def billing_webhook(request: Request):
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001 — malformed body → 400
            raise HTTPException(
                status_code=400, detail="Webhook body must be a JSON object"
            ) from exc
        event_type = payload.get("type") if isinstance(payload, dict) else None
        if not event_type:
            raise HTTPException(status_code=400, detail="Missing 'type' field")
        try:
            return billing.handle_webhook_event(
                event_type, payload, s3_client=app.state.s3_client
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── POST /api/forms/submit — contact form lead routing relay (Phase 9) ────
    # The deployed sites post a classic HTML form (application/x-www-form-
    # urlencoded); a JSON body is accepted too for API-style clients.

    @app.post("/api/forms/submit")
    async def form_submit(request: Request):
        content_type = request.headers.get("content-type", "")
        try:
            if "application/json" in content_type:
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise ValueError("not a JSON object")
                data = {k: ("" if v is None else str(v)) for k, v in payload.items()}
            else:
                form_data = await request.form()
                data = {key: str(value) for key, value in form_data.items()}
        except Exception as exc:  # noqa: BLE001 — unreadable body → 400
            raise HTTPException(
                status_code=400, detail="Submission must be form or JSON data"
            ) from exc

        slug = (data.get("business_slug") or "").strip()
        name = (data.get("name") or "").strip()
        phone = (data.get("phone") or "").strip()
        message = (data.get("message") or "").strip()
        if not slug or not name or not message:
            raise HTTPException(
                status_code=400,
                detail="business_slug, name and message are required",
            )

        try:
            forms.send_inquiry_email(slug, name, phone, message)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — provider failure → 502
            raise HTTPException(
                status_code=502, detail=f"Inquiry dispatch failed: {exc}"
            ) from exc

        return {"status": "success", "message": "Inquiry sent successfully"}

    return app


# Module-level default app for `uvicorn app.main:app`.
app = create_app()
