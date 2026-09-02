# SoloSpark AI Agency Platform

Local-business lead generation + website pipeline for a one-person agency.
The system discovers local businesses (Google Places), audits their websites,
generates landing-page copy with a local LLM, renders and deploys preview
sites to Cloudflare R2, and runs CAN-SPAM-compliant outreach with Stripe
billing — all orchestrated from a CLI or the review dashboard.

## Setup

Requires Python 3.14 (any recent 3.10+ works).

```powershell
pip install -r requirements.txt
copy .env.example .env   # then fill in your keys
```

Only `PLACES_API_KEY` is needed for discovery; everything else has a
local/offline path:

| Service | Needed for | Offline behavior when unset |
|---|---|---|
| Local LLM (`LOCAL_LLM_URL`, LM Studio / Unsloth) | copy + pitch generation | requests fail with `LLMGenerationError` |
| Google Places (`PLACES_API_KEY`) | discovery | `discover` exits with an error |
| Cloudflare R2 (`R2_*`) | preview deployment | deploy fails with a clear config error |
| Resend / SMTP (`RESEND_API_KEY` or `SMTP_HOST`) | outreach + inquiry relay | falls back to **dry-run** (records only) |
| Stripe (`STRIPE_SECRET_KEY`, webhook secret) | billing webhooks | webhook endpoint returns 400/502 on unknown events |

The ledger is a local SQLite file (`solospark.db`, override with
`DATABASE_URL`) — no server setup needed.

## CLI (the whole pipeline in one command)

```powershell
# One-shot: discover → audit → generate mockups → deploy → queue outreach
python run.py pipeline --query "plumbers in Eugene OR" --limit 20 --auto-deploy
```

Step by step instead:

```powershell
python run.py discover --query "plumbers in Eugene OR" --limit 20
python run.py audit                          # audit all DISCOVERED leads with websites
python run.py generate --slug acme-plumbing  # LLM landing copy for one lead
python run.py deploy --all                   # render + publish previews to R2
# (or: python run.py deploy --slug acme-plumbing)

python run.py outreach                       # DRY-RUN by default → outreach_queue.csv
python run.py outreach --send                # actually dispatch (rate-limited, 5/hr)
```

Notes:

- `discover` dedupes by Google `place_id`; businesses whose only "website" is a
  Yelp/Facebook directory listing are flagged as bad-site leads automatically.
- Outreach **dry-run is the default** (spec 6.2): it writes generated pitches to
  `outreach_queue.csv` for manual review and touches no email provider or rate
  ledger. `--send` dispatches via Resend (or SMTP fallback), enforces the
  rolling 5-emails/hour limit, skips leads already contacted or opted out, and
  stops the batch immediately if the rate limit is hit.
- Every email carries a CAN-SPAM physical address + unsubscribe link;
  unsubscribes set the lead to LOST and are honored forever.

## Dashboard (review + manual actions)

```powershell
uvicorn app.main:app --port 8000
```

- `GET /` — lead list with stages, audit flags, preview links, plus bulk
  "Audit pending" / "Deploy all ready" actions and per-row Deploy buttons
- `GET /discover` — discovery page (search query + max-results form)
- `POST /api/discover` — run a Google Places search and ingest new leads
  (ledger-deduped by `place_id`; returns found count + new-lead rows)
- `POST /api/audit` — audit every DISCOVERED lead that has a website
  (flags + scraped email persisted; stage → AUDITED)
- `POST /api/deploy` — render + publish previews for all leads with stored
  copy (optionally `{"ids": [...]}` to scope); returns per-lead URLs
- `POST /api/deploy/{id}` — deploy one lead's preview to R2
- `GET /pipeline` — one-click pipeline trigger page (query + max-results form,
  auto-deploy toggle) with per-stage result cards
- `POST /api/pipeline/run` — run the full pipeline in one request: discover →
  audit pending sites → generate copy for every no-website/bad-site lead →
  optional deploy to R2 → queue outreach pitches (dry-run, persisted to the
  ledger for the `/outreach` approval flow). Per-lead LLM/R2 failures are
  collected in `copy_failed` / `deploy_failed` and never abort the run;
  missing `PLACES_API_KEY` → 503, Places network error → 502
- `GET /review/{id}` — per-lead review page: generated copy, live preview,
  one-click actions
- `POST /generate/{id}` — (re)generate landing copy via the local LLM
- `POST /regenerate-prompt/{id}` — regenerate with an extra instruction
  (preserves advanced deal stages)
- `POST /outreach/send/{id}` — dispatch a single pitch email (ledger-checked)
- `GET /outreach` — DB-backed outreach queue: every queued/sent/failed pitch
  with the owning lead, recipient, subject and body preview
- `POST /api/outreach/generate` — draft pitches for all eligible leads (bad
  site + live preview + email, not opted out, never sent) into the review
  queue; idempotent (already-queued leads are skipped); per-lead LLM failures
  are reported, not fatal
- `POST /api/outreach/send-selected` — approve + dispatch a chosen set of
  queued rows (`{"ids": [...]}`); each is ledger-checked and CAN-SPAM footed;
  on success the lead moves to CONTACTED; a rate-limit hit stops the batch
  (returns `rate_limited: true`, HTTP 200)
- `GET /api/outreach/unsubscribe?email=...` — CAN-SPAM opt-out link
- `POST /api/billing/webhook` — Stripe events: activate on
  `checkout.session.completed`, suspend site + serve maintenance page on
  `customer.subscription.deleted` / `invoice.payment_failed`
- `POST /api/forms/submit` — lead-form relay from deployed sites (transactional;
  intentionally bypasses the outreach rate ledger)

## Layout

```
run.py                  CLI orchestrator (Phase 7) — thin wrapper over app/pipeline.py
app/config.py           settings (.env → frozen dataclass)
app/models.py           SQLAlchemy models + deal-stage enums, SQLite ledger
app/scanner.py          Places discovery, website audit, ingest/dedup
app/pipeline.py         shared pipeline steps (discover/audit/generate/deploy/outreach) used by CLI + web
app/llm_engine.py       local-LLM JSON generation: landing copy, pitch emails
app/deployer.py         Jinja site rendering + R2 (S3) deployment
app/builder.py          multi-page site builder (validated schema, themes)
app/outreach.py         CAN-SPAM footer, rate limit, Resend/SMTP/dry-run send
app/forms.py            inquiry relay from deployed sites
app/billing.py          Stripe webhook handling, suspend/restore/export
app/main.py             FastAPI dashboard + API routes
site_templates/         Jinja2 templates for generated sites
templates/              dashboard templates
tests/                  pytest suite — all external services faked
```

## Tests

```powershell
python -m pytest tests/ -q
```

The full suite (140+ tests) runs without any live service: LLM, R2, Resend,
SMTP, Stripe and Google Places are all injected as fakes or exercised through
their dry-run paths.
