"""Phase 3 — Local LLM content generation (OpenAI-compatible endpoint).

Talks to LM Studio / Unsloth Desktop via the ``openai`` SDK and produces:

* **Landing page copy** — strict-JSON schema consumed by the site template
  (tagline, hero, services grid, about, trust badges, CTA).
* **Outreach pitch email** — ``{"subject", "body"}`` referencing the specific
  audit flaws and the live preview URL.

Testability: every public function accepts an injectable ``client`` (anything
with ``chat.completions.create(**kwargs)``). When omitted, a real client is
built from ``LOCAL_LLM_URL`` / ``LOCAL_LLM_MODEL`` settings — so tests run with
a fake client and no live model.
"""

from __future__ import annotations

import json
import re

from .config import get_settings

# ── Errors ───────────────────────────────────────────────────────────────────


class LLMGenerationError(RuntimeError):
    """Raised when the LLM response cannot be parsed/validated into schema."""


# ── Client construction ──────────────────────────────────────────────────────


def build_client():
    """Create an OpenAI-compatible client pointed at the local LLM server."""
    from openai import OpenAI  # imported lazily so tests don't need it

    settings = get_settings()
    return OpenAI(base_url=settings.local_llm_url, api_key=settings.local_llm_api_key)


# ── JSON extraction / validation ─────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

ALLOWED_ICONS = ("wrench", "shield", "clock", "phone", "star")

DEFAULT_TRUST_BADGES = ["Licensed & Insured", "Fast Response", "Local Expertise"]


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response (fences/prose tolerated)."""
    if not text:
        raise LLMGenerationError("empty LLM response")
    candidate = text.strip()

    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(candidate)
        if not match:
            raise LLMGenerationError(f"no JSON object in LLM response: {text[:200]!r}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMGenerationError(f"invalid JSON from LLM: {exc}") from exc

    if not isinstance(data, dict):
        raise LLMGenerationError("LLM JSON is not an object")
    return data


def _require_str(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LLMGenerationError(f"missing/invalid string field: {key!r}")
    return value.strip()


def validate_landing_copy(data: dict) -> dict:
    """Normalize an LLM payload into the exact schema the template expects."""
    tagline = _require_str(data, "tagline")
    hero_headline = _require_str(data, "hero_headline")
    hero_subheadline = _require_str(data, "hero_subheadline")

    raw_services = data.get("services")
    if not isinstance(raw_services, list) or not raw_services:
        raise LLMGenerationError("'services' must be a non-empty array")
    services = []
    for item in raw_services[:6]:  # schema caps at 6 cards
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        description = item.get("description")
        if not isinstance(title, str) or not title.strip():
            continue
        icon = item.get("icon_name")
        if icon not in ALLOWED_ICONS:
            icon = "star"
        services.append({
            "title": title.strip(),
            "description": (description or "").strip() if isinstance(description, str) else "",
            "icon_name": icon,
        })
    if not services:
        raise LLMGenerationError("no valid service items in LLM response")

    about_heading = _require_str(data, "about_heading")
    about_text = _require_str(data, "about_text")

    badges = data.get("why_choose_us")
    if isinstance(badges, list):
        badges = [b.strip() for b in badges if isinstance(b, str) and b.strip()][:3]
    else:
        badges = []
    while len(badges) < 3:
        badge = DEFAULT_TRUST_BADGES[len(badges)]
        if badge not in badges:
            badges.append(badge)

    cta_text = _require_str(data, "cta_text")

    def _opt_str(key: str) -> str:
        value = data.get(key)
        return value.strip() if isinstance(value, str) else ""

    def _opt_menu_items() -> list[dict]:
        raw = data.get("menu_items")
        items = []
        if isinstance(raw, list):
            for item in raw[:12]:  # schema caps at 12 dishes
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                description = item.get("description")
                price = item.get("price")
                items.append({
                    "name": name.strip(),
                    "description": description.strip() if isinstance(description, str) else "",
                    "price": price.strip() if isinstance(price, str) else "",
                })
        return items

    def _opt_hours() -> list[dict]:
        raw = data.get("hours")
        entries: list[tuple[str, str]] = []
        if isinstance(raw, list):
            for item in raw[:10]:
                if not isinstance(item, dict):
                    continue
                day = item.get("day")
                hours = item.get("hours")
                if (
                    isinstance(day, str) and isinstance(hours, str)
                    and day.strip() and hours.strip()
                ):
                    entries.append((day.strip(), hours.strip()))
        elif isinstance(raw, dict):  # tolerate {"monday": "9am-5pm"} style
            for day, hours in raw.items():
                if isinstance(hours, str) and str(day).strip() and hours.strip():
                    entries.append((str(day).strip(), hours.strip()))
        out = []
        seen: set[str] = set()
        for day, hours in entries[:10]:
            if day not in seen:
                seen.add(day)
                out.append({"day": day, "hours": hours})
        return out

    def _opt_layout() -> list[str]:
        raw = data.get("layout")
        if not isinstance(raw, list):
            return []
        names = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                name = item.strip().lower()
                if name not in names:
                    names.append(name)
        return names[:10]

    return {
        "tagline": tagline,
        "hero_headline": hero_headline,
        "hero_subheadline": hero_subheadline,
        "services": services,
        "about_heading": about_heading,
        "about_text": about_text,
        "why_choose_us": badges,
        "cta_text": cta_text,
        # Section headings — written by the LLM to match the business; empty
        # ones are filled with category-aware defaults (see
        # apply_category_defaults) so the template never shows canned text.
        "services_heading": _opt_str("services_heading"),
        "services_intro": _opt_str("services_intro"),
        "contact_heading": _opt_str("contact_heading"),
        "contact_subtext": _opt_str("contact_subtext"),
        "cta_band_subtext": _opt_str("cta_band_subtext"),
        # Section composition — the LLM picks which sections appear and their
        # order; resolve_layout normalizes it and drops data-less sections at
        # render time.
        "layout": _opt_layout(),
        "menu_items": _opt_menu_items(),
        "hours": _opt_hours(),
        "brand": _normalize_brand(data.get("brand")),
    }


_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _normalize_brand(value) -> dict | None:
    """Validate the optional LLM-supplied brand palette.

    Returns ``{"primary": "#rrggbb", "secondary": "#rrggbb"}`` with any valid
    subset, or ``None`` when nothing usable was provided (callers fall back
    to default colors).
    """
    if not isinstance(value, dict):
        return None
    out: dict[str, str] = {}
    for key in ("primary", "secondary"):
        raw = value.get(key)
        if isinstance(raw, str) and _HEX_COLOR_RE.match(raw.strip()):
            out[key] = raw.strip().lower()
    return out or None


# ── Category-aware defaults (de-canned copy) ────────────────────────────────

#: One shared template used to serve every business left restaurants with
#: "Request a free quote" forms and salons with estimate buttons. These
#: profiles let the deterministic layer fill in headings that match what the
#: customer actually wants from THIS kind of business. LLM-written values
#: always win — defaults only fill gaps the model left empty.
_CATEGORY_PROFILE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "food": (
        "restaurant", "cafe", "coffee", "bistro", "pizz", "taco", "diner",
        "bakery", "deli", "grill", "bbq", "sushi", "ramen", "burger",
        "food truck", "catering", "brunch", "ice cream", "tavern", "brewpub",
    ),
    "booking": (
        "salon", "spa", "barber", "massage", "nail", "tattoo", "gym",
        "fitness", "yoga", "pilates", "clinic", "dental", "dentist", "vet",
        "veterinary", "chiropractic", "physical therapy", "therapy",
        "coaching", "tutoring", "lessons",
    ),
    "trade": (
        "plumb", "hvac", "roof", "electric", "contractor", "construction",
        "landscap", "lawn", "cleaning", "pest", "garage door", "window",
        "painting", "flooring", "appliance", "handyman", "fencing",
        "septic", "drain", "mason",
    ),
}

#: Standalone "bar"/"pub" categories — matched with word boundaries so
#: "barber shop" (booking) is never misread as a bar.
_BAR_OR_PUB_RE = re.compile(r"\b(bar|pub)\b")

_CATEGORY_DEFAULTS: dict[str, dict[str, str]] = {
    "food": {
        "services_heading": "What We Serve",
        "services_intro": "Fresh and made in-house — come hungry, we'll take it from here.",
        "contact_heading": "Reserve a Table",
        "contact_subtext": "Book online or give us a call — we'll have your table ready.",
        "cta_band_subtext": "Hungry? Reserve a table and we'll take care of the rest.",
    },
    "booking": {
        "services_heading": "Our Services",
        "services_intro": "Book in minutes — your next appointment is one click away.",
        "contact_heading": "Book an Appointment",
        "contact_subtext": "Pick a time that works for you and we'll confirm right away.",
        "cta_band_subtext": "Ready when you are — book your visit today.",
    },
    "trade": {
        "services_heading": "Services We Offer",
        "services_intro": "Fast, dependable work across {city} and the surrounding area.",
        "contact_heading": "Request a Free Estimate",
        "contact_subtext": "Send us the details and we'll be in touch — usually the same day.",
        "cta_band_subtext": (
            "Tell us what you need — we'll get back to you fast with a straight "
            "answer and a fair price."
        ),
    },
    "default": {
        "services_heading": "What We Offer",
        "services_intro": "Everything you need, all in one place — right here in {city}.",
        "contact_heading": "Get in Touch",
        "contact_subtext": "Send us the details and we'll be in touch — usually the same day.",
        "cta_band_subtext": "Tell us what you need — we'll get back to you fast.",
    },
}

#: Profiles where "quote" language is wrong for the customer (a restaurant
#: wants reservations, a salon wants appointments). If the model still wrote
#: quote language for them, it is replaced deterministically. Trades are NOT
#: guarded — a free quote is exactly what a homeowner wants.
_CTA_GUARD_REPLACEMENTS = {
    "food": "Reserve a Table",
    "booking": "Book an Appointment",
}


def _category_profile(category: str) -> str:
    """Map a free-form category string to food/booking/trade/default."""
    text = (category or "").lower()
    for profile, keywords in _CATEGORY_PROFILE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return profile
    if text and _BAR_OR_PUB_RE.search(text):
        return "food"
    return "default"


def apply_category_defaults(copy: dict, category: str, city: str = "") -> dict:
    """Fill missing section headings with copy that fits the business category.

    LLM-written values always win; this only fills empty/missing fields from
    the category profile (substituting *city* where the default mentions it)
    and replaces quote language that makes no sense for food-service or
    appointment businesses — a restaurant does not send out quotes. Mutates
    and returns *copy*.
    """
    profile = _category_profile(category)
    place = (city or "").strip() or "your area"
    for key, default in _CATEGORY_DEFAULTS[profile].items():
        if not copy.get(key):
            copy[key] = default.replace("{city}", place)

    guard = _CTA_GUARD_REPLACEMENTS.get(profile)
    if guard:
        if "quote" in (copy.get("cta_text") or "").lower():
            copy["cta_text"] = guard
        if "quote" in (copy.get("contact_heading") or "").lower():
            copy["contact_heading"] = guard

    # Structural keys must always exist — the template renders with
    # StrictUndefined, and copies stored before these fields were introduced
    # simply lack them.
    for key in ("layout", "menu_items", "hours", "gallery_images"):
        if not isinstance(copy.get(key), list):
            copy[key] = []
    return copy


# ── Section composition ──────────────────────────────────────────────────────

#: Every partial under site_templates/sections/. A layout is an ordered list
#: of these names; unknown names are dropped by resolve_layout.
KNOWN_SECTIONS = (
    "hero",
    "services",
    "menu",
    "gallery",
    "about",
    "hours_location",
    "cta_band",
    "contact",
)

#: Fallback order per category profile — used when the LLM omits or mangles
#: the layout. The trade/default order is exactly the classic single-template
#: order, so legacy sites render identically.
DEFAULT_LAYOUTS: dict[str, list[str]] = {
    "food": [
        "hero", "menu", "services", "gallery",
        "about", "hours_location", "cta_band", "contact",
    ],
    "booking": [
        "hero", "services", "gallery",
        "about", "hours_location", "cta_band", "contact",
    ],
    "trade": ["hero", "services", "about", "cta_band", "contact"],
    "default": ["hero", "services", "about", "cta_band", "contact"],
}

#: Sections that only render when the copy actually carries their data. A
#: restaurant with no scraped menu items must not show an empty menu block.
_DATA_GATES: dict[str, tuple[str, ...]] = {
    "menu": ("menu_items",),
    "gallery": ("gallery_images", "about_images"),
    "hours_location": ("hours",),
    "services": ("services",),
    "about": ("about_text", "why_choose_us"),
}


def resolve_layout(copy: dict, category: str, layout_mode: str = "ai") -> list[str]:
    """Resolve the final section order for rendering.

    In ``"ai"`` mode (the default) the LLM's ``layout`` is trusted when it
    names at least three known sections including hero and contact; otherwise
    the category default applies. In ``"classic"`` mode the category's fixed
    template order (``DEFAULT_LAYOUTS``) always applies, ignoring the LLM's
    pick — the legacy single-template behaviour.

    Hero is always first and contact always last (the middle keeps the
    source's relative order), and any section without its data is dropped so
    a page never shows an empty block.
    """
    profile = _category_profile(category)
    if layout_mode == "classic":
        layout = list(DEFAULT_LAYOUTS[profile])
    else:
        raw = copy.get("layout")
        if not isinstance(raw, list):
            raw = []
        known = [name for name in dict.fromkeys(raw) if name in KNOWN_SECTIONS]

        if len(known) >= 3 and "hero" in known and "contact" in known:
            layout = known[:8]
        else:
            layout = list(DEFAULT_LAYOUTS[profile])

    # Pin hero first / contact last, keep the LLM's middle order.
    middle = [name for name in layout if name not in ("hero", "contact")]
    pinned = ["hero"] + middle + ["contact"]

    resolved: list[str] = []
    seen: set[str] = set()
    for name in pinned:
        if name in seen:
            continue
        seen.add(name)
        gate = _DATA_GATES.get(name)
        if gate and not any(copy.get(key) for key in gate):
            continue  # no data → skip the section entirely
        resolved.append(name)
    return resolved


#: Nav labels for sections with a meaningful anchor (hero/cta_band are
#: skipped; contact is always appended last).
_NAV_LABELS: dict[str, str] = {
    "services": "Services",
    "menu": "Menu",
    "gallery": "Gallery",
    "about": "About",
    "hours_location": "Hours & Location",
}


def nav_links_for(layout: list[str]) -> list[tuple[str, str]]:
    """Return (label, anchor) pairs in layout order, capped at 4 + Contact."""
    links: list[tuple[str, str]] = []
    for name in layout:
        label = _NAV_LABELS.get(name)
        if label and len(links) < 4:
            links.append((label, f"#{name}"))
    links.append(("Contact", "#contact"))
    return links


# ── Named-color safety net ───────────────────────────────────────────────────

#: Deep, professional palettes for the color names operators actually type into
#: the dashboard prompt box. Local models frequently ignore explicit "change
#: the colors to X" requests (or echo back the baseline palette), so when an
#: operator names a concrete color we apply it deterministically if the model's
#: output did not already reflect it.
_NAMED_COLOR_PALETTES: dict[str, tuple[str, str]] = {
    "navy": ("#1e3a8a", "#0f172a"),      # blue-900 / slate-900 (before "blue")
    "crimson": ("#be123c", "#881337"),   # rose-700 / rose-900
    "red": ("#b91c1c", "#7f1d1d"),       # red-700 / red-900
    "blue": ("#1d4ed8", "#1e3a8a"),      # blue-600 / blue-900
    "green": ("#15803d", "#14532d"),     # green-700 / green-900
    "teal": ("#0f766e", "#134e4a"),      # teal-700 / teal-900
    "orange": ("#c2410c", "#7c2d12"),    # orange-700 / orange-900
    "purple": ("#6d28d9", "#4c1d95"),    # violet-700 / violet-900
    "yellow": ("#a16207", "#713f12"),    # yellow-700 / yellow-900
    "gold": ("#a16207", "#713f12"),      # same family as yellow
    "black": ("#0f172a", "#334155"),     # slate-900 / slate-700
    "gray": ("#334155", "#1e293b"),      # slate-700 / slate-800
    "grey": ("#334155", "#1e293b"),
    "white": ("#e2e8f0", "#94a3b8"),     # light slate (kept legible)
}

#: Words that mark an instruction as a color-change request rather than an
#: incidental mention of a color ("remove the red text" must not re-theme).
_COLOR_INTENT_WORDS = (
    "change", "switch", "update", "make", "set", "use", "new",
    "color", "colour", "scheme", "palette", "theme",
)


def _named_color_palette(instructions: str) -> dict | None:
    """Return a canonical palette when *instructions* ask for a named color."""
    if not instructions:
        return None
    text = instructions.lower()
    if not any(re.search(rf"\b{word}\b", text) for word in _COLOR_INTENT_WORDS):
        return None
    for name, (primary, secondary) in _NAMED_COLOR_PALETTES.items():
        if re.search(rf"\b{re.escape(name)}\b", text):
            return {"primary": primary, "secondary": secondary}
    return None


def enforce_named_color(
    copy: dict, instructions: str, previous_brand: dict | None = None
) -> dict:
    """Apply a named color from *instructions* when the model ignored it.

    Mutates and returns *copy*. The override fires only when the operator
    explicitly requested a named color AND the returned palette is missing or
    identical to the previous one — a palette the model actually changed is
    always trusted as-is.
    """
    requested = _named_color_palette(instructions)
    if not requested:
        return copy
    current = _normalize_brand(copy.get("brand"))
    previous = _normalize_brand(previous_brand)
    if not current or (previous and current == previous):
        copy["brand"] = requested
    return copy


# ── Explicit-edit safety net (quoted replacement values) ─────────────────────

#: Words that mark an instruction as an edit request ("the button should stay"
#: is a mention, not an edit).
_EDIT_INTENT_WORDS = (
    "change", "make", "set", "update", "use", "say",
    "replace", "swap", "put", "write",
)

#: Copy fields an operator can point at by name, with the words they're likely
#: to use for each. Order matters only for exact-distance ties.
_EXPLICIT_EDIT_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cta_text", ("call to action", "cta", "button")),
    ("hero_headline", ("headline",)),
    ("hero_subheadline", ("sub-headline", "sub headline", "subheadline")),
    ("tagline", ("tagline",)),
    ("about_heading", ("about heading",)),
    ("contact_heading", ("contact heading", "form heading")),
)

#: Quoted replacement values — straight or curly double quotes. Single quotes
#: are deliberately NOT supported: apostrophes in copy would break matching.
_QUOTED_VALUE_RE = re.compile(r'["“]([^"”]{2,140})["”]')

#: How far around a quoted value we look for the field keyword it belongs to.
_EDIT_KEYWORD_WINDOW = 80


def _field_for_value(text: str, start: int, end: int) -> str | None:
    """Pick the copy field whose keyword sits closest to a quoted value."""
    best: tuple[int, str] | None = None
    for field, keywords in _EXPLICIT_EDIT_FIELDS:
        for kw in keywords:
            pattern = rf"\b{re.escape(kw)}\b"
            for m in re.finditer(pattern, text[:start]):
                if start - m.end() <= _EDIT_KEYWORD_WINDOW:
                    dist = start - m.end()
                    if best is None or dist < best[0]:
                        best = (dist, field)
            for m in re.finditer(pattern, text[end:]):
                if m.start() <= _EDIT_KEYWORD_WINDOW:
                    dist = m.start()
                    if best is None or dist < best[0]:
                        best = (dist, field)
    return best[1] if best else None


def apply_explicit_edits(copy: dict, instructions: str) -> dict:
    """Apply quoted replacement values from *instructions* verbatim.

    Local models frequently paraphrase or ignore exact replacement text, so a
    prompt like ``change the CTA button to "Book Now"`` is enforced here,
    after the model runs: the quoted value wins for the field whose keyword
    (button/headline/tagline/…) sits closest to it. Mutates and returns *copy*.
    """
    if not instructions:
        return copy
    text = instructions.lower()
    if not any(re.search(rf"\b{word}\b", text) for word in _EDIT_INTENT_WORDS):
        return copy
    for m in _QUOTED_VALUE_RE.finditer(instructions):
        value = m.group(1).strip()
        field = _field_for_value(text, m.start(), m.end())
        if field and copy.get(field) != value:
            copy[field] = value
    return copy


# ── Chat helper with one JSON-repair retry ───────────────────────────────────

_JSON_REMINDER = (
    "\n\nIMPORTANT: Respond with ONLY a valid JSON object. "
    "No markdown fences, no commentary."
)


def _chat_json(client, system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    last_error: Exception | None = None
    for attempt in range(2):  # initial + one repair retry
        kwargs = {
            "model": get_settings().local_llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + (_JSON_REMINDER if attempt else "")},
            ],
            "temperature": 0.4,
            "max_tokens": max_tokens,
        }
        try:
            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content
            return _extract_json(text)
        except LLMGenerationError as exc:
            last_error = exc
        except Exception as exc:  # noqa: BLE001 — SDK/network errors
            try:
                from openai import OpenAIError  # lazy: tests use fake clients
            except ImportError:
                raise
            if isinstance(exc, OpenAIError):
                # Uniform error type for every caller (no retry: the endpoint
                # is down/misconfigured, so a repair retry cannot help).
                raise LLMGenerationError(f"LLM request failed: {exc}") from exc
            raise
    raise LLMGenerationError(f"LLM did not return valid JSON after retry: {last_error}")


# ── Public generators ────────────────────────────────────────────────────────

_LANDING_SYSTEM_PROMPT = (
    "You are a direct-response web copywriter specializing in local service "
    "businesses. You write punchy, trust-building marketing copy that converts "
    "homeowners and local customers into phone calls. You ALWAYS respond with a "
    "single valid JSON object — never markdown, never commentary."
)


def generate_landing_copy(
    business_name: str,
    category: str,
    city: str,
    raw_info: str = "",
    extra_instructions: str = "",
    client=None,
    site_reference: dict | None = None,
) -> dict:
    """Generate strict-JSON landing page copy for a local business.

    Returns the validated schema: tagline, hero_headline, hero_subheadline,
    services[3-6] {title, description, icon_name}, about_heading, about_text,
    why_choose_us[3], cta_text, and an optional ``brand`` palette
    ({"primary", "secondary"} hex colors) the LLM picks to match the business.

    ``site_reference`` (optional) is a dict from ``site_ref.scrape_site_reference``
    — when present, the LLM uses the lead's existing website copy as its
    factual basis instead of inventing content. ``raw_info`` remains as a
    plain-text fallback for callers that already scraped text themselves.

    ``extra_instructions`` (optional) appends a custom prompt adjustment to the
    user prompt — used by the dashboard's regenerate-with-prompt endpoint.
    """
    llm = client if client is not None else build_client()

    from .site_ref import reference_copy_block  # lazy: avoids import cycles

    reference_text = ""
    reference_instructions = ""
    if site_reference and reference_copy_block(site_reference):
        reference_text = reference_copy_block(site_reference)
        reference_instructions = (
            "Use the reference above as your factual basis. Keep every concrete "
            "claim accurate — services offered, service area, specialties and "
            "differentiators must match what they already publish. Rewrite it in a "
            "modern, confident, professional voice; do not copy their wording "
            "verbatim and do not invent facts that are not supported by the "
            "reference or clearly implied by the business name/category.\n\n"
        )
    elif raw_info:
        reference_text = raw_info[:3000]

    user_prompt = (
        f"Write landing page copy for a local business.\n\n"
        f"Business name: {business_name}\n"
        f"Category: {category or 'local service business'}\n"
        f"City: {city}\n\n"
        f"{reference_instructions}"
        f"Reference material from their current website (may be empty):\n"
        f"{reference_text or '(none available — use your knowledge of this category)'}\n\n"
        "Match every heading to what a customer of THIS kind of business actually "
        "wants: restaurants, cafes and bars want reservations or orders — never "
        "quote/estimate language; salons, spas, gyms and clinics want "
        "appointments; trades (plumbing, roofing, electrical) want free "
        "estimates. The contact_heading must be the action a customer of this "
        "category takes next.\n\n"
        "Return JSON with exactly these keys:\n"
        '{\n'
        '  "tagline": "short brand hook",\n'
        '  "hero_headline": "6-10 words, benefit-driven",\n'
        '  "hero_subheadline": "1-2 sentences highlighting local trust",\n'
        '  "services": [{"title": "...", "description": "1 sentence", '
        '"icon_name": "wrench|shield|clock|phone|star"}] (3-6 items),\n'
        '  "services_heading": "short heading for the services section, matched to the category",\n'
        '  "services_intro": "one sentence under that heading",\n'
        '  "about_heading": "...",\n'
        '  "about_text": "2 short paragraphs about the business and its local roots",\n'
        '  "why_choose_us": ["badge 1", "badge 2", "badge 3"],\n'
        '  "cta_text": "e.g. Get Your Free Quote — matched to the category (a restaurant says Reserve a Table, not Get Your Free Quote)",\n'
        '  "contact_heading": "heading for the contact form — what the customer actually wants from this business",\n'
        '  "contact_subtext": "one short sentence under the contact heading",\n'
        '  "cta_band_subtext": "one punchy line for the call-to-action band",\n'
        '  "layout": ["hero", "..."] — ordered section list for THIS business, chosen from: hero, services, menu, gallery, about, hours_location, cta_band, contact. ALWAYS start with "hero" and end with "contact". Order the middle sections for conversion: a restaurant or cafe leads with "menu" (then services), a trade or service business leads with "services", include "gallery" for visual businesses (restaurants, salons, contractors) and "hours_location" when hours are known. 3-8 sections total.\n'
        '  "menu_items": [{"name": "...", "description": "short", "price": "$12"}] — ONLY real dishes/drinks actually listed in the reference material, with prices exactly as published there. Empty array [] when the reference has no menu. NEVER invent dishes or prices.\n'
        '  "hours": [{"day": "Monday", "hours": "9:00am-5:00pm"}] — ONLY if opening hours are stated in the reference material. Empty array [] otherwise. NEVER invent hours.\n'
        '  "brand": {"primary": "#rrggbb", "secondary": "#rrggbb"} — pick two hex '
        'colors that fit the business brand/industry (deep, professional tones; '
        'avoid neon), e.g. navy + steel blue for plumbing\n'
        '}'
    )
    if extra_instructions:
        user_prompt += (
            "\n\nAdditional instructions from the operator — these OVERRIDE "
            "every default above where they conflict (e.g. a requested color "
            "scheme MUST be reflected in brand.primary/secondary as #rrggbb hex "
            f"values):\n{extra_instructions}"
        )
    # Thinking models (e.g. Qwen3) spend part of the token budget on internal
    # reasoning before emitting content — 1400 was fully consumed by reasoning,
    # leaving zero tokens for the JSON body.
    data = _chat_json(llm, _LANDING_SYSTEM_PROMPT, user_prompt, max_tokens=4096)
    copy = validate_landing_copy(data)
    # Pre-save syntax QA: fix broken syntax in the model's own output before
    # the deterministic safety nets run — they must be the last word, so a
    # sloppy QA response can never revert them.
    copy = syntax_check_copy(copy, business_name, category, city, client=llm)
    # Category-aware safety net: fill any headings the model left blank with
    # copy that matches what this kind of business actually sells, and kill
    # quote language on food/booking businesses.
    apply_category_defaults(copy, category, city)
    return copy


_REFINE_SYSTEM_PROMPT = (
    "You are a direct-response web copywriter specializing in local service "
    "businesses. You refine an existing landing page's copy on request: you "
    "change ONLY what the operator asks for and keep everything else — "
    "structure, tone, facts, and wording of untouched fields — as close to the "
    "current copy as possible. If the operator asks to change colors or the "
    "color scheme, you MUST return new brand.primary/secondary values as "
    "#rrggbb hex codes in the requested color family — never keep the old "
    "palette when a change is asked for. If the operator gives exact replacement "
    "text in quotes, use it verbatim in the matching field — do not paraphrase "
    "it. Before responding, re-read the operator's instruction and verify your "
    "output actually reflects every requested change; if one is missing or "
    "wrong, fix it. You ALWAYS respond with a single valid JSON object — never "
    "markdown, never commentary."
)

# Schema keys passed to the LLM as the refinement baseline (asset paths such
# as logo_url/hero_image_url/about_images/gallery_images are persisted
# extensions handled by the caller, not part of the LLM contract).
_COPY_SCHEMA_KEYS = (
    "tagline",
    "hero_headline",
    "hero_subheadline",
    "services",
    "services_heading",
    "services_intro",
    "about_heading",
    "about_text",
    "why_choose_us",
    "cta_text",
    "contact_heading",
    "contact_subtext",
    "cta_band_subtext",
    "layout",
    "menu_items",
    "hours",
    "brand",
)


def refine_landing_copy(
    current_copy: dict,
    business_name: str,
    category: str,
    city: str,
    instructions: str,
    client=None,
) -> dict:
    """Update an existing landing-page copy dict following operator instructions.

    The current copy (schema fields only) is shown to the model as the
    baseline; it returns the complete updated JSON in the same validated
    schema. Unlike ``generate_landing_copy`` this never scrapes or invents —
    it is a surgical edit of what already exists, used for fine-tuning a
    generated site before the customer pays. If the model drops the ``brand``
    palette, the previous one is carried over so fine-tuning never loses the
    business's colors.
    """
    llm = client if client is not None else build_client()

    baseline = {
        key: current_copy.get(key)
        for key in _COPY_SCHEMA_KEYS
        if current_copy.get(key) is not None
    }
    user_prompt = (
        f"Update the landing page copy for {business_name} "
        f"({category or 'local service business'}, {city}).\n\n"
        "Current copy (JSON):\n"
        f"{json.dumps(baseline, ensure_ascii=False)}\n\n"
        f"Operator instructions: {instructions.strip()}\n\n"
        "Apply ONLY the requested changes; keep every other field intact and "
        "do not invent new facts. Return the COMPLETE updated JSON object with "
        "exactly these keys:\n"
        '{\n'
        '  "tagline": "...",\n'
        '  "hero_headline": "...",\n'
        '  "hero_subheadline": "...",\n'
        '  "services": [{"title": "...", "description": "1 sentence", '
        '"icon_name": "wrench|shield|clock|phone|star"}] (3-6 items),\n'
        '  "services_heading": "...",\n'
        '  "services_intro": "...",\n'
        '  "about_heading": "...",\n'
        '  "about_text": "2 short paragraphs",\n'
        '  "why_choose_us": ["badge 1", "badge 2", "badge 3"],\n'
        '  "cta_text": "...",\n'
        '  "contact_heading": "...",\n'
        '  "contact_subtext": "...",\n'
        '  "cta_band_subtext": "...",\n'
        '  "layout": ["hero", "..."] — keep the current section order unless the operator asks to change it,\n'
        '  "menu_items": [...] — keep unchanged unless asked (only real menu items; never invent dishes or prices),\n'
        '  "hours": [...] — keep unchanged unless asked (never invent hours),\n'
        '  "brand": {"primary": "#rrggbb", "secondary": "#rrggbb"} — keep the '
        'existing colors unless asked to change them (when asked, return new '
        '#rrggbb hex values in the requested color family)\n'
        '}'
    )
    data = _chat_json(llm, _REFINE_SYSTEM_PROMPT, user_prompt, max_tokens=4096)
    refined = validate_landing_copy(data)
    # Pre-save syntax QA: runs before the deterministic safety nets below so a
    # weak model that echoes the baseline during QA can never revert an
    # operator's explicit (quoted or named-color) edit.
    refined = syntax_check_copy(refined, business_name, category, city, client=llm)

    # Older copies predate the section-heading fields — fill any the model
    # dropped with category-appropriate defaults.
    apply_category_defaults(refined, category, city)

    # Factual sections (menu/hours) are scraped data, not creative copy — a
    # weak model that drops them must not silently delete the business's real
    # menu or hours during a surgical edit. Carry them over when the refined
    # output lost them but the current copy had them.
    for key in ("menu_items", "hours"):
        if not refined.get(key) and current_copy.get(key):
            refined[key] = list(current_copy[key])

    if not refined.get("brand"):
        previous = _normalize_brand(current_copy.get("brand"))
        if previous:
            refined["brand"] = previous
    # Safety net for weak local models: if the operator asked for a named
    # color but the model kept (or dropped) the old palette, apply it here.
    enforce_named_color(refined, instructions, current_copy.get("brand"))
    # Second safety net: quoted replacement values ("change the CTA to ...")
    # are applied verbatim so exact text can never be paraphrased away.
    apply_explicit_edits(refined, instructions)
    return refined


# ── Pre-save syntax QA pass ──────────────────────────────────────────────────

_SYNTAX_CHECK_SYSTEM_PROMPT = (
    "You are a meticulous QA editor for marketing copy stored as JSON. You fix "
    "ONLY broken syntax and rendering problems — you never rewrite style, tone, "
    "or facts. Problems to fix: leftover template placeholders such as {city}, "
    "[TODO] or {{ ... }}; truncated, garbled or duplicated words and sentences; "
    "HTML tags or markdown (e.g. <b>, **bold**) embedded inside text fields; "
    "invalid hex color values in the brand palette. Preserve every key exactly "
    "as given, keep all facts (names, numbers, prices, addresses) unchanged, "
    "and make the smallest edit that fixes each problem. If there is nothing "
    "to fix, return the JSON exactly as received. You ALWAYS respond with a "
    "single valid JSON object — never markdown, never commentary."
)


def syntax_check_copy(
    copy: dict,
    business_name: str,
    category: str = "",
    city: str = "",
    client=None,
) -> dict:
    """Model-based syntax QA pass that runs before copy is saved and uploaded.

    The validated copy is shown to the model, which returns the complete JSON
    with only broken-syntax fixes applied (leftover placeholders, truncated or
    garbled sentences, HTML/markdown inside text fields, invalid hex colors).
    The result is accepted when it parses as a JSON object and re-validates
    against the landing-copy schema (which enforces every required field); any
    key the model dropped — layout choices, asset references, optional
    sections — is restored from the original, so a sloppy response can apply
    its fixes but never lose content. If validation fails outright, the
    original copy is returned unchanged: this pass can never break a working
    generation. Asset references are always taken from the original: they are
    uploaded file keys, not copy text.
    """
    original = dict(copy)
    llm = client if client is not None else build_client()
    user_prompt = (
        f"Business name: {business_name}\n"
        f"Category: {category or 'local service business'}\n"
        f"City: {city}\n\n"
        "Landing page copy JSON:\n"
        f"{json.dumps(copy, ensure_ascii=False)}\n\n"
        "Return the COMPLETE copy JSON with only syntax fixes applied."
    )
    try:
        data = _chat_json(
            llm, _SYNTAX_CHECK_SYSTEM_PROMPT, user_prompt, max_tokens=4096
        )
    except LLMGenerationError:
        return original  # the QA pass must never break a working generation
    if not isinstance(data, dict):
        return original
    try:
        validated = validate_landing_copy(data)
    except LLMGenerationError:
        return original  # required content missing/broken — keep what we had
    # Keys the model dropped are restored from the original (layout choices,
    # asset references, optional sections). Asset references in particular are
    # uploaded file keys, never copy text — the model's version is discarded
    # even when it "fixed" them.
    for key in original:
        if key not in validated:
            validated[key] = original[key]
    if validated.get("brand") is None and isinstance(original.get("brand"), dict):
        validated["brand"] = original["brand"]
    return validated


_PITCH_SYSTEM_PROMPT = (
    "You are a friendly web developer writing a short cold outreach email to a "
    "local business owner. Tone: helpful, peer-level, non-salesy. You reference "
    "their specific website problems in plain language and offer a free improved "
    "version they can click and see immediately. You ALWAYS respond with a single "
    "valid JSON object — never markdown, never commentary."
)


def generate_pitch_email(
    business_name: str,
    flaws: list[str],
    preview_url: str,
    client=None,
) -> dict:
    """Generate ``{"subject", "body"}`` for a cold outreach pitch email."""
    llm = client if client is not None else build_client()
    flaw_text = "\n".join(f"- {f}" for f in flaws) if flaws else "- (no specific issues)"
    user_prompt = (
        f"Write a short cold outreach email to the owner of {business_name}.\n\n"
        f"I audited their current website and found these problems:\n{flaw_text}\n\n"
        f"I built them a free improved version at: {preview_url}\n\n"
        "Rules: under 150 words, no hype, sign off as 'Aaron from SoloSpark'.\n"
        "Return JSON with exactly these keys:\n"
        '{\n'
        '  "subject": "short, specific subject line",\n'
        '  "body": "plain-text email body (no markdown) that mentions the '
        'specific problems above and includes the preview URL"\n'
        '}'
    )
    data = _chat_json(llm, _PITCH_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
    subject = _require_str(data, "subject")
    body = _require_str(data, "body")
    return {"subject": subject, "body": body}
