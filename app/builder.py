"""Phase 7 — Post-payment multi-page prompt builder.

After a customer pays, the operator (or the client) can ask for extra pages.
``expand_site_with_prompt`` sends the existing landing-copy JSON plus the
client's free-form prompt to the local LLM and receives an *extended site
schema*: a unified color theme plus content for four standard subpages
(``about.html``, ``services.html``, ``gallery.html``, ``contact.html``).

The schema is then compiled to full HTML by ``render_site_pages`` (shared
navigation + theme on every page) and uploaded with
``deploy_expanded_site`` — either the preview bucket or a production client
bucket, one object per page under ``{slug}/``.

Everything is injectable (LLM client, S3 client, template dir) so the whole
pipeline runs in tests without live services.
"""

from __future__ import annotations

import json
import re
from datetime import date

from .config import get_settings
from .deployer import (
    DEFAULT_TEMPLATE_DIR,
    _get_env,
    _paragraphs,
    build_s3_client,
    icon_svg,
    preview_url_for,
)
from .llm_engine import LLMGenerationError, _chat_json, build_client


class BuilderError(RuntimeError):
    """Raised when the LLM output cannot be turned into a valid site schema."""


ALLOWED_SUBPAGES = ("about.html", "services.html", "gallery.html", "contact.html")

SUBPAGE_LABELS = {
    "about.html": "About",
    "services.html": "Services",
    "gallery.html": "Gallery",
    "contact.html": "Contact",
}

DEFAULT_THEME = {"primary": "#1d4ed8", "secondary": "#0f172a"}

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")

MAX_PARAGRAPHS = 6
MAX_ITEMS = 8
MAX_CAPTIONS = 12

_BUSINESS_KEYS = ("name", "slug", "category", "city", "phone", "address")


# ── normalization helpers ─────────────────────────────────────────────────────


def _normalize_business(raw: object) -> dict:
    """Coerce a raw business mapping to the exact keys templates expect."""
    src = raw if isinstance(raw, dict) else {}
    return {key: str(src.get(key) or "").strip() for key in _BUSINESS_KEYS}


def _normalize_copy(raw: object) -> dict:
    """Fill every key the site templates reference with safe defaults."""
    copy = raw if isinstance(raw, dict) else {}

    services = []
    for svc in copy.get("services") or []:
        if not isinstance(svc, dict):
            continue
        title = str(svc.get("title") or "").strip()
        if not title:
            continue
        services.append(
            {
                "title": title,
                "description": str(svc.get("description") or "").strip(),
                "icon_name": str(svc.get("icon_name") or "star"),
            }
        )

    def s(key: str) -> str:
        return str(copy.get(key) or "").strip()

    badges = [str(b).strip() for b in (copy.get("why_choose_us") or []) if str(b).strip()]

    return {
        "tagline": s("tagline"),
        "hero_headline": s("hero_headline"),
        "hero_subheadline": s("hero_subheadline"),
        "services": services,
        "about_heading": s("about_heading"),
        "about_text": s("about_text"),
        "why_choose_us": badges[:3],
        "cta_text": s("cta_text") or "Request a Free Quote",
    }


def _clean_theme(raw: object) -> dict:
    """Keep only valid hex colors; fall back to defaults per key."""
    src = raw if isinstance(raw, dict) else {}
    out = {}
    for key in ("primary", "secondary"):
        value = str(src.get(key) or "").strip()
        out[key] = value if _HEX_RE.match(value) else DEFAULT_THEME[key]
    return out


def _default_page(filename: str, copy: dict, business: dict) -> dict:
    """Sensible fallback content so the site is always complete (5 pages)."""
    name = business.get("name") or "the business"
    category = business.get("category") or "local service"
    city = business.get("city") or ""
    where = f" in {city}" if city else ""

    if filename == "about.html":
        paras = _paragraphs(copy.get("about_text") or "")
        if not paras:
            paras = [
                f"{name} is a locally owned and operated {category} serving{where}. "
                "Our team shows up on time, does the work right, and stands behind every job."
            ]
        return {"heading": "About Us", "paragraphs": paras[:MAX_PARAGRAPHS], "items": [], "captions": []}

    if filename == "services.html":
        items = [
            {"title": svc["title"], "text": svc["description"]}
            for svc in copy.get("services") or []
            if svc.get("title")
        ][:MAX_ITEMS]
        return {
            "heading": "Our Services",
            "paragraphs": [],
            "items": items,
            "captions": [],
        }

    if filename == "gallery.html":
        captions = [f"{svc['title']} project" for svc in (copy.get("services") or [])[:MAX_CAPTIONS] if svc.get("title")]
        if not captions:
            captions = ["Job site visit", "Completed project", "Before & after", "Final walkthrough"]
        return {"heading": "Recent Work", "paragraphs": [], "items": [], "captions": captions}

    # contact.html
    phone = business.get("phone") or ""
    paras = [f"Call us at {phone} and we'll get right back to you." if phone else "Give us a call and we'll get right back to you."]
    paras.append("Or send us a message using the form below — we respond fast.")
    return {"heading": "Contact Us", "paragraphs": paras, "items": [], "captions": []}


def _normalize_page(filename: str, raw: object, copy: dict, business: dict) -> dict:
    """Clamp one LLM-provided page; fall back to defaults field by field."""
    d = _default_page(filename, copy, business)
    if not isinstance(raw, dict):
        return d

    heading = str(raw.get("heading") or "").strip() or d["heading"]

    paragraphs = [str(p).strip() for p in (raw.get("paragraphs") or []) if str(p).strip()]
    paragraphs = paragraphs[:MAX_PARAGRAPHS] or d["paragraphs"]

    items = []
    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        items.append({"title": title, "text": str(item.get("text") or "").strip()})
    items = items[:MAX_ITEMS] or d["items"]

    captions = [str(c).strip() for c in (raw.get("captions") or []) if str(c).strip()]
    captions = captions[:MAX_CAPTIONS] or d["captions"]

    return {"heading": heading, "paragraphs": paragraphs, "items": items, "captions": captions}


def validate_site_schema(data: object, copy: dict, business: dict) -> dict:
    """Validate raw LLM output into ``{"theme": ..., "pages": ...}``.

    Unknown page keys are dropped; missing standard pages are filled with
    defaults so the result always renders a complete 5-page site.
    """
    if not isinstance(data, dict):
        raise BuilderError("LLM site schema must be a JSON object")
    theme = _clean_theme(data.get("theme"))
    pages_raw = data.get("pages") if isinstance(data.get("pages"), dict) else {}
    pages = {
        filename: _normalize_page(filename, pages_raw.get(filename), copy, business)
        for filename in ALLOWED_SUBPAGES
    }
    return {"theme": theme, "pages": pages}


# ── expansion (LLM) ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a website architect who converts one-page business sites into "
    "polished multi-page sites for local service businesses. You always "
    "respond with a single valid JSON object — never markdown, never commentary."
)


def expand_site_with_prompt(
    current_json: dict,
    client_prompt: str,
    *,
    client=None,
) -> dict:
    """Extend an existing one-page site schema into a multi-page site.

    ``current_json`` may be either the bare landing-copy dict or a wrapper of
    the form ``{"business": {...}, "copy": {...}}`` (the shape stored on the
    Business row). Returns a self-contained site schema::

        {
          "business": {...}, "copy": {...}, "theme": {...},
          "nav": [{"label", "href"}, ...],
          "pages": {"about.html": {...}, "services.html": {...},
                    "gallery.html": {...}, "contact.html": {...}}
        }
    """
    current = current_json if isinstance(current_json, dict) else {}
    if isinstance(current.get("copy"), dict):
        copy_raw = current["copy"]
        business_raw = current.get("business") or {}
    else:
        copy_raw = current
        business_raw = {k: current[k] for k in _BUSINESS_KEYS if k in current}

    business = _normalize_business(business_raw)
    copy = _normalize_copy(copy_raw)

    llm = client if client is not None else build_client()

    user_prompt = (
        "Extend this existing one-page website into a multi-page site.\n\n"
        f"Business name: {business['name'] or 'Unknown'}\n"
        f"Category: {business['category'] or 'local service business'}\n"
        f"City: {business['city'] or 'Unknown'}\n\n"
        "Existing landing page copy (keep the new content consistent with this voice):\n"
        + json.dumps(
            {k: copy[k] for k in ("tagline", "hero_headline", "about_heading", "cta_text")},
            ensure_ascii=False,
        )
        + "\n\nWrite new content ONLY for these subpages:\n"
        '  "about.html": {"heading": "...", "paragraphs": ["1-3 short paragraphs"]},\n'
        '  "services.html": {"heading": "...", "items": [{"title": "...", "text": "one sentence"}]},\n'
        '  "gallery.html": {"heading": "...", "captions": ["4-8 short captions for typical jobs/projects"]},\n'
        '  "contact.html": {"heading": "...", "paragraphs": ["1-2 sentences directing customers to call or message"]}\n\n'
        "Also choose a unified color theme that fits the business:\n"
        '  "theme": {"primary": "#rrggbb", "secondary": "#rrggbb"}\n\n'
        'Return JSON with exactly these top-level keys: {"theme": {...}, "pages": {...}}'
    )
    prompt = str(client_prompt or "").strip()
    if prompt:
        user_prompt += f"\n\nAdditional instructions from the client:\n{prompt}"

    try:
        data = _chat_json(llm, _SYSTEM_PROMPT, user_prompt, max_tokens=1600)
    except LLMGenerationError as exc:
        raise BuilderError(f"LLM did not return a valid site schema: {exc}") from exc

    result = validate_site_schema(data, copy, business)
    nav = [{"label": "Home", "href": "index.html"}]
    nav += [{"label": SUBPAGE_LABELS[f], "href": f} for f in ALLOWED_SUBPAGES]
    return {**result, "business": business, "copy": copy, "nav": nav}


# ── compilation ───────────────────────────────────────────────────────────────


def render_site_pages(site_schema: dict, template_dir=None) -> dict[str, str]:
    """Compile a site schema into ``{filename: html}`` for all 5 pages.

    Every page shares the navigation and theme from the schema; missing or
    malformed fields fall back to defaults so rendering never raises on
    partial LLM output.
    """
    schema = site_schema if isinstance(site_schema, dict) else {}

    business = _normalize_business(schema.get("business"))
    copy = _normalize_copy(schema.get("copy"))
    theme = _clean_theme(schema.get("theme"))

    nav_raw = schema.get("nav")
    nav = [
        {"label": str(i.get("label") or "").strip(), "href": str(i.get("href") or "").strip()}
        for i in (nav_raw if isinstance(nav_raw, list) else [])
        if isinstance(i, dict) and str(i.get("href") or "").strip()
    ]
    if not nav:
        nav = [{"label": "Home", "href": "index.html"}]
        nav += [{"label": SUBPAGE_LABELS[f], "href": f} for f in ALLOWED_SUBPAGES]

    services = [
        {**svc, "icon_svg": icon_svg(svc.get("icon_name") or "star")}
        for svc in copy["services"]
    ]

    env = _get_env(str(template_dir or DEFAULT_TEMPLATE_DIR))
    form_action = str(
        (schema.get("business") or {}).get("form_action") or "/api/forms/submit"
    )
    ctx = {
        "business": business,
        "copy": {**copy, "services": services},
        "theme": theme,
        "nav_items": nav,
        "form_action": form_action,
        "current_year": date.today().year,
    }

    pages: dict[str, str] = {}
    pages["index.html"] = env.get_template("site_index.html.j2").render(
        **ctx, active_href="index.html"
    )

    page_tpl = env.get_template("site_page.html.j2")
    for filename in ALLOWED_SUBPAGES:
        raw_page = (schema.get("pages") or {}).get(filename)
        page = _normalize_page(filename, raw_page, copy, business)
        pages[filename] = page_tpl.render(
            **ctx,
            page_name=filename,
            heading=page["heading"],
            paragraphs=page["paragraphs"],
            items=page["items"],
            captions=page["captions"],
            show_form=(filename == "contact.html"),
            active_href=filename,
        )
    # templates may carry leading blank lines; keep uploaded HTML tidy
    return {name: html.lstrip() for name, html in pages.items()}


# ── deployment ────────────────────────────────────────────────────────────────


def deploy_expanded_site(slug: str, pages: dict[str, str], *, s3_client=None) -> str:
    """Upload every page of an expanded site to R2 under ``{slug}/``.

    Returns the public preview URL for the site root.
    """
    client = s3_client if s3_client is not None else build_s3_client()
    settings = get_settings()
    for filename in sorted(pages):
        client.put_object(
            Bucket=settings.r2_bucket_name,
            Key=f"{slug}/{filename}",
            Body=str(pages[filename]).encode("utf-8"),
            ContentType="text/html; charset=utf-8",
        )
    return preview_url_for(slug)
