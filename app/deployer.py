"""Phase 4 — Static site template engine & Cloudflare R2 deployer.

* ``render_site_html`` compiles validated LLM copy + business data into a
  standalone, mobile-ready HTML page (Tailwind via CDN).
* ``deploy_to_r2`` pushes the page to Cloudflare R2 / any S3-compatible
  bucket and returns the absolute live preview URL.

Testability: the S3 client is injectable — tests pass a fake with a
``put_object`` method, so no network or real bucket is ever touched.
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .llm_engine import apply_category_defaults, nav_links_for, resolve_layout

# ── SVG icon set (heroicons outline, stroke=currentColor) ───────────────────

_ICONS: dict[str, str] = {
    "wrench": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
              'stroke-width="1.5" stroke="currentColor" class="w-6 h-6"><path stroke-linecap="round" '
              'stroke-linejoin="round" d="M11.42 15.17 17.25 21A2.652 2.652 0 0 0 21 17.25l-5.877-5.877M11.42 '
              '15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 1 '
              '1-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.713-.067l4.209 1.026a1.75 1.75 0 0 '
              '0 1.176-2.604l-3.806-4.192a4.5 4.5 0 0 0-6.335 0L3.077 9.877a4.5 4.5 0 0 0 0 6.336l4.192 '
              '3.806c.835.761 2.057 1.068 3.202.802a4.493 4.493 0 0 0 1.713-.067Z"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
              'stroke-width="1.5" stroke="currentColor" class="w-6 h-6"><path stroke-linecap="round" '
              'stroke-linejoin="round" d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 '
              '6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31'
              '-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z"/></svg>',
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
             'stroke-width="1.5" stroke="currentColor" class="w-6 h-6"><path stroke-linecap="round" '
             'stroke-linejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/></svg>',
    "phone": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
             'stroke-width="1.5" stroke="currentColor" class="w-6 h-6"><path stroke-linecap="round" '
             'stroke-linejoin="round" d="M2.25 6.75c0 8.284 6.716 15 15 15h2.25a2.25 2.25 0 0 0 2.25-2.25v'
             '-1.372c0-.516-.351-.966-.852-1.091l-4.423-1.106c-.44-.11-.902.055-1.173.417l-.97 1.293c-.282'
             '.376-.769.542-1.21.38a12.035 12.035 0 0 1-7.143-7.143c-.162-.441.004-.928.38-1.21l1.293-.97'
             'c.363-.271.527-.734.417-1.173L6.963 3.102a1.125 1.125 0 0 0-1.091-.852H4.5A2.25 2.25 0 0 0 '
             '2.25 4.5v2.25Z"/></svg>',
    "star": '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" '
            'stroke-width="1.5" stroke="currentColor" class="w-6 h-6"><path stroke-linecap="round" '
            'stroke-linejoin="round" d="M11.48 3.499a.562.562 0 0 1 1.04 0l2.125 5.111a.563.563 0 0 0 '
            '.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 0 0-.182.557l1.285 '
            '5.385a.562.562 0 0 1-.84.61l-4.725-2.885a.562.562 0 0 0-.586 0L6.982 20.54a.562.562 0 0 '
            '1-.84-.61l1.285-5.386a.562.562 0 0 0-.182-.557l-4.204-3.602a.562.562 0 0 1 .321-.988l5.518'
            '-.442a.563.563 0 0 0 .475-.345L11.48 3.5Z"/></svg>',
}


def icon_svg(name: str) -> str:
    """Resolve an icon name to its inline SVG (unknown names fall back to star)."""
    return _ICONS.get(name, _ICONS["star"])


# ── Jinja2 environment ───────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE_DIR = _PROJECT_ROOT / "site_templates"
SITE_TEMPLATE_NAME = "landing.html.j2"


def _tel_href(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    return f"+{digits}" if digits else ""


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


# ── Brand palette ────────────────────────────────────────────────────────────

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

DEFAULT_BRAND_PRIMARY = "#1d4ed8"  # blue-600 — fallback when the LLM gives none
DEFAULT_BRAND_SECONDARY = "#0f172a"  # slate-900


def _shade(hex_color: str, factor: float) -> str:
    """Blend *hex_color* toward black (factor < 0) or white (factor > 0)."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return hex_color
    channels = [int(value[i : i + 2], 16) for i in (0, 2, 4)]
    target = 0 if factor < 0 else 255
    strength = min(abs(factor), 1.0)
    mixed = [round(c + (target - c) * strength) for c in channels]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def build_brand_palette(brand: dict | None) -> dict:
    """Normalize an optional validated brand dict into a render-ready palette.

    Always returns ``{"primary", "primary_dark", "primary_light", "secondary"}``
    with safe hex values — invalid/missing input falls back to defaults so the
    template never sees an undefined color (StrictUndefined).
    """
    brand = brand if isinstance(brand, dict) else {}
    primary = brand.get("primary")
    if not (isinstance(primary, str) and _HEX_COLOR_RE.match(primary.strip())):
        primary = DEFAULT_BRAND_PRIMARY
    primary = primary.strip().lower()

    secondary = brand.get("secondary")
    if not (isinstance(secondary, str) and _HEX_COLOR_RE.match(secondary.strip())):
        secondary = DEFAULT_BRAND_SECONDARY
    secondary = secondary.strip().lower()

    return {
        "primary": primary,
        "primary_dark": _shade(primary, -0.28),
        "primary_light": _shade(primary, 0.88),
        "secondary": secondary,
    }


@lru_cache(maxsize=4)
def _get_env(template_dir: str):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True,
        undefined=StrictUndefined,
    )
    env.filters["tel_href"] = _tel_href
    env.filters["paragraphs"] = _paragraphs
    return env


# ── Site rendering ───────────────────────────────────────────────────────────


def render_site_html(
    business_data: dict,
    copy_data: dict,
    template_dir: str | None = None,
    layout_mode: str | None = None,
) -> str:
    """Compile a standalone landing page for one business.

    ``business_data`` keys: name, slug, category, address, phone (optional);
    city, rating, review_count (optional — trust badges).
    ``copy_data`` is the validated schema from ``llm_engine.validate_landing_copy``,
    optionally extended with asset references persisted at generation time:
    ``brand`` (palette), ``logo_url``, ``hero_image_url``, ``about_images``.
    Optional override: ``business_data["form_action"]`` for the lead form target.
    ``layout_mode`` ("ai" | "classic") forces the section order; when omitted it
    falls back to ``copy_data["layout_mode"]`` persisted at generation time,
    then to "ai".
    """
    env = _get_env(str(template_dir or DEFAULT_TEMPLATE_DIR))
    template = env.get_template(SITE_TEMPLATE_NAME)

    # Normalize optional fields so the template can stay under StrictUndefined.
    business = {
        **{
            "name": "",
            "slug": "",
            "category": "",
            "phone": "",
            "address": "",
            "city": "",
            "rating": None,
            "review_count": None,
        },
        **business_data,
    }

    # Copies generated before the section-heading fields existed lack them —
    # fill from the category profile so StrictUndefined never sees a gap and
    # old sites re-render with headings that fit their business.
    apply_category_defaults(copy_data, business["category"], business["city"])

    services = [
        {**svc, "icon_svg": icon_svg(svc.get("icon_name", "star"))}
        for svc in copy_data.get("services", [])
        if isinstance(svc, dict) and svc.get("title")
    ]

    def _image_list(key: str, cap: int) -> list[str]:
        raw = copy_data.get(key)
        if not isinstance(raw, list):
            return []
        urls = [u for u in raw if isinstance(u, str) and u]
        return urls[:cap]

    about_images = _image_list("about_images", 2)
    gallery_images = _image_list("gallery_images", 4)

    copy = {
        **copy_data,
        "services": services,
        "brand": build_brand_palette(copy_data.get("brand")),
        "logo_url": copy_data.get("logo_url") or "",
        "hero_image_url": copy_data.get("hero_image_url") or "",
        "about_images": about_images,
        "gallery_images": gallery_images,
    }

    # Section composition: in "ai" mode the LLM picks which sections appear and
    # their order; in "classic" mode the category's fixed template order is used.
    # resolve_layout normalizes either (pins hero/contact, drops data-less
    # sections) so a restaurant leads with its menu while a plumber keeps the
    # classic services-first flow. nav_links mirrors the resolved order.
    mode = layout_mode or copy_data.get("layout_mode") or "ai"
    if mode not in ("ai", "classic"):
        mode = "ai"
    layout = resolve_layout(copy, business["category"], mode)

    return template.render(
        business=business,
        copy=copy,
        layout=layout,
        nav_links=nav_links_for(layout),
        form_action=business_data.get("form_action") or "/api/forms/submit",
        current_year=date.today().year,
    )


# ── Cloudflare R2 deployment ─────────────────────────────────────────────────


def build_s3_client():
    """Create a boto3 S3 client pointed at the configured R2 endpoint."""
    from .config import get_settings

    settings = get_settings()
    if not (settings.r2_endpoint_url and settings.r2_access_key_id and settings.r2_secret_access_key):
        raise ValueError(
            "R2 credentials are not configured — set R2_ENDPOINT_URL, "
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY"
        )
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
    )


def preview_url_for(slug: str) -> str:
    """Absolute public URL for a deployed site.

    Uses the direct object path (``{slug}/index.html``): R2 custom domains
    serve explicit object paths, not folder-index URLs.
    """
    from .config import get_settings

    base = get_settings().r2_public_base_url.rstrip("/")
    return f"{base}/{slug}/index.html"


def deploy_to_r2(slug: str, html_content: str, s3_client=None) -> str:
    """Upload ``{slug}/index.html`` to R2 and return the live preview URL.

    Pass a fake S3 client in tests; when omitted, one is built from settings.
    """
    client = s3_client if s3_client is not None else build_s3_client()
    from .config import get_settings

    settings = get_settings()
    client.put_object(
        Bucket=settings.r2_bucket_name,
        Key=f"{slug}/index.html",
        Body=html_content.encode("utf-8"),
        ContentType="text/html; charset=utf-8",
    )
    return preview_url_for(slug)


def delete_site_objects(slug: str, s3_client=None) -> int:
    """Delete every object under ``{slug}/`` (page + assets) from the bucket.

    Lists with pagination and issues one bulk delete per page. Returns the
    number of objects removed; a blank slug is a no-op returning 0. Used by
    the site-delete endpoint so a removed preview leaves no orphan objects
    behind in R2.
    """
    slug = (slug or "").strip().lstrip("/")
    if not slug:
        return 0
    client = s3_client if s3_client is not None else build_s3_client()
    from .config import get_settings

    settings = get_settings()
    prefix = f"{slug}/"
    removed = 0
    token: str | None = None
    while True:
        kwargs: dict = {"Bucket": settings.r2_bucket_name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs) or {}
        keys = [obj["Key"] for obj in (page.get("Contents") or []) if obj.get("Key")]
        if keys:
            client.delete_objects(
                Bucket=settings.r2_bucket_name,
                Delete={"Objects": [{"Key": k} for k in keys], "Quiet": True},
            )
            removed += len(keys)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return removed


def deploy_site_assets(
    slug: str,
    assets: list[dict],
    s3_client=None,
) -> list[str]:
    """Upload site assets under ``{slug}/assets/`` and return relative paths.

    Each asset is ``{"key": "logo.png", "data": bytes, "content_type": "image/png"}``.
    Returns the page-relative URLs (``"assets/logo.png"``) in input order —
    these are what get persisted into the copy dict so later re-renders keep
    referencing objects that already exist in the bucket.
    """
    client = s3_client if s3_client is not None else build_s3_client()
    from .config import get_settings

    settings = get_settings()
    relative: list[str] = []
    for asset in assets or []:
        key = (asset.get("key") or "").strip().lstrip("/")
        data = asset.get("data")
        if not key or not isinstance(data, (bytes, bytearray)):
            continue
        client.put_object(
            Bucket=settings.r2_bucket_name,
            Key=f"{slug}/assets/{key}",
            Body=bytes(data),
            ContentType=asset.get("content_type") or "application/octet-stream",
        )
        relative.append(f"assets/{key}")
    return relative


# ── Pre-upload syntax gate ───────────────────────────────────────────────────

_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "source", "track", "wbr",
}


def _scan_tags(html: str) -> list[tuple[str, bool, bool]] | None:
    """Tokenize tags the way a browser starts to: '<' + name, then scan to
    the first '>' that is not inside a quoted attribute value.

    Returns a list of (name, is_closing, is_self_closed) tuples, or ``None``
    when the document contains an unterminated tag (a '<' with no matching
    '>' — or a quote that never closes — before end of input).
    """
    tags: list[tuple[str, bool, bool]] = []
    i, n = 0, len(html)
    while i < n:
        lt = html.find("<", i)
        if lt == -1:
            break
        j = lt + 1
        closing = False
        if j < n and html[j] == "/":
            closing = True
            j += 1
        if j >= n or not html[j].isalpha():
            # Not a tag opener ('<' in plain text, doctype, etc.)
            i = lt + 1
            continue
        name_start = j
        while j < n and (html[j].isalpha() or html[j].isdigit()):
            j += 1
        name = html[name_start:j].lower()
        k = j
        while k < n:
            c = html[k]
            if c == '"' or c == "'":
                end = html.find(c, k + 1)
                if end == -1:
                    return None  # quote never closes → malformed tag
                k = end + 1
            elif c == ">":
                break
            else:
                k += 1
        else:
            return None  # tag never closed before end of input
        tags.append((name, closing, html[lt : k + 1].endswith("/>")))
        i = k + 1
    return tags


def check_html_syntax(html: str) -> list[str]:
    """Deterministic structural checks for a rendered page before upload.

    Returns a list of problems (empty list = safe to publish): unbalanced or
    mismatched tags, unterminated tags, unrendered Jinja markers, and missing
    shell elements. This gate catches template bugs that would ship a broken
    page to the customer; copy-level issues are handled upstream by the LLM
    syntax check (``llm_engine.syntax_check_copy``).
    """
    problems: list[str] = []

    if re.search(r"\{\{.*?\}\}|\{%.*?%\}", html, re.DOTALL):
        problems.append("unrendered Jinja template markers")

    for probe in ("<!DOCTYPE html>", "<title>", "</html>"):
        if probe not in html:
            problems.append(f"missing {probe}")

    # Raw-text elements: browsers never tag-parse their contents, so drop the
    # bodies (inline JS/CSS full of quotes would otherwise confuse the scan).
    scanable = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1\s*>", "", html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Strip comments so tags inside them are never counted.
    scanable = re.sub(r"<!--.*?-->", "", scanable, flags=re.DOTALL)

    tags = _scan_tags(scanable)
    if tags is None:
        problems.append("unterminated tag (no closing '>')")
        return problems

    stack: list[str] = []
    for name, closing, self_closed in tags:
        if closing:
            if not stack:
                problems.append(f"stray closing tag </{name}>")
            elif stack[-1] != name:
                problems.append(
                    f"mismatched closing tag </{name}> (expected </{stack[-1]}>)"
                )
                # Resync: pop until the matching open tag, if any.
                if name in stack:
                    while stack and stack.pop() != name:
                        pass
            else:
                stack.pop()
        elif not self_closed and name not in _VOID_TAGS:
            stack.append(name)
    if stack:
        problems.append(f"unclosed tags: {', '.join(stack[:5])}")
    return problems
