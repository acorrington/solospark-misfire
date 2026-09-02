"""Tests for app.deployer — site rendering + R2 deployment (offline fakes)."""

from __future__ import annotations

import pytest

from app import deployer
from app.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Isolate the cached Settings snapshot per test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


BUSINESS = {
    "name": "Acme Plumbing",
    "slug": "acme-plumbing",
    "category": "Plumber",
    "address": "123 Main St, Eugene, OR 97401",
    "phone": "(541) 555-0142",
}

COPY = {
    "tagline": "Fast, honest plumbing for Eugene",
    "hero_headline": "Your Leaks, Fixed Fast",
    "hero_subheadline": "Acme Plumbing keeps Eugene homes dry.",
    "about_heading": "Who We Are",
    "about_text": "We have been fixing pipes for 20 years.\n\nLocally owned and operated.",
    "cta_text": "Get My Free Quote",
    "services": [
        {"title": "Emergency Repairs", "description": "24/7 leak response.", "icon_name": "wrench"},
        {"title": "Drain Cleaning", "description": "Hydro jetting.", "icon_name": "clock"},
        {"title": "Water Heaters", "description": "Install & repair.", "icon_name": "star"},
    ],
    "why_choose_us": ["Licensed & Insured", "Fast Response", "Local Expertise"],
}


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {"ETag": '"fake"'}


# ── render_site_html ─────────────────────────────────────────────────────────


def test_render_contains_business_identity_and_call_link():
    html = deployer.render_site_html(BUSINESS, COPY)
    assert "Acme Plumbing" in html
    assert "(541) 555-0142" in html
    assert 'href="tel:+5415550142"' in html


def test_render_includes_tailwind_viewport_and_title():
    html = deployer.render_site_html(BUSINESS, COPY)
    assert "cdn.tailwindcss.com" in html
    assert '<meta name="viewport" content="width=device-width, initial-scale=1.0" />' in html
    assert "<title>Acme Plumbing | Plumber</title>" in html


def test_render_renders_all_services_with_svg_icons():
    html = deployer.render_site_html(BUSINESS, COPY)
    assert "Emergency Repairs" in html
    assert "Drain Cleaning" in html
    assert "Water Heaters" in html
    # each service card carries an inline SVG icon
    assert html.count("<svg") >= len(COPY["services"]) + 2  # services + header phone + badges


def test_render_unknown_icon_falls_back_to_star():
    copy = {**COPY, "services": [{"title": "X", "description": "", "icon_name": "rocket"}]}
    html = deployer.render_site_html(BUSINESS, copy)
    assert "X" in html
    assert "<svg" in html  # fallback icon rendered


def test_render_trust_badges_present():
    from markupsafe import escape

    html = deployer.render_site_html(BUSINESS, COPY)
    for badge in COPY["why_choose_us"]:
        assert str(escape(badge)) in html  # autoescaped: "&" -> "&amp;"


def test_render_about_text_paragraphs():
    html = deployer.render_site_html(BUSINESS, COPY)
    assert "We have been fixing pipes for 20 years." in html
    assert "Locally owned and operated." in html


def test_render_lead_form_targets_api_and_carries_slug():
    html = deployer.render_site_html(BUSINESS, COPY)
    assert 'action="/api/forms/submit"' in html
    assert '<input type="hidden" name="business_slug" value="acme-plumbing" />' in html
    for field in ("name", "phone", "message"):
        assert f'name="{field}"' in html


def test_render_form_action_overridable():
    biz = {**BUSINESS, "form_action": "https://app.solospark.net/api/forms/submit"}
    html = deployer.render_site_html(biz, COPY)
    assert 'action="https://app.solospark.net/api/forms/submit"' in html


def test_render_footer_has_current_year_and_address():
    from datetime import date

    html = deployer.render_site_html(BUSINESS, COPY)
    assert f"&copy; {date.today().year} Acme Plumbing" in html
    assert "123 Main St, Eugene, OR 97401" in html


def test_render_escaping_neutralizes_script_in_copy():
    biz = {**BUSINESS, "name": "Acme <script>alert(1)</script>"}
    html = deployer.render_site_html(biz, COPY)
    assert "<script>alert(1)</script>" not in html.replace("<script src=", "")
    assert "&lt;script&gt;" in html


def test_render_missing_optional_fields_ok():
    biz = {"name": "No Phone Co", "slug": "no-phone-co"}
    html = deployer.render_site_html(biz, COPY)
    assert "No Phone Co" in html
    assert "tel:" not in html


def test_render_fills_missing_section_headings_for_old_copies():
    # COPY predates the section-heading fields — rendering must fill them from
    # the category profile (StrictUndefined would otherwise raise).
    html = deployer.render_site_html(BUSINESS, dict(COPY))
    assert "Request a Free Estimate" in html  # trade default for a Plumber
    assert "Services We Offer" in html


def test_render_restaurant_old_copy_gets_reservation_heading():
    biz = {**BUSINESS, "name": "Luigi's Pizzeria", "slug": "luigis-pizzeria", "category": "Restaurant"}
    copy = {**COPY, "hero_headline": "Wood-fired pizza done right"}
    html = deployer.render_site_html(biz, copy)
    assert "Reserve a Table" in html
    assert "Request a free quote" not in html


def test_render_keeps_llm_written_contact_heading():
    biz = {**BUSINESS, "category": "Restaurant"}
    copy = {**COPY, "contact_heading": "Book Your Private Dining Room"}
    html = deployer.render_site_html(biz, copy)
    assert "Book Your Private Dining Room" in html


# ── deploy_to_r2 ─────────────────────────────────────────────────────────────


def test_deploy_uploads_correct_object_and_returns_url():
    fake = FakeS3()
    url = deployer.deploy_to_r2("acme-plumbing", "<html>hi</html>", s3_client=fake)

    base = get_settings().r2_public_base_url.rstrip("/")
    assert url == f"{base}/acme-plumbing/index.html"
    assert len(fake.puts) == 1
    kwargs = fake.puts[0]
    assert kwargs["Bucket"] == "solospark-previews"
    assert kwargs["Key"] == "acme-plumbing/index.html"
    assert kwargs["Body"] == b"<html>hi</html>"
    assert kwargs["ContentType"].startswith("text/html")


def test_deploy_returns_configured_public_base(monkeypatch):
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://sites.example.com")
    get_settings.cache_clear()
    fake = FakeS3()
    url = deployer.deploy_to_r2("joe-s-plumbing", "<html></html>", s3_client=fake)
    assert url == "https://sites.example.com/joe-s-plumbing/index.html"


def test_preview_url_for():
    base = get_settings().r2_public_base_url.rstrip("/")
    assert deployer.preview_url_for("acme-plumbing") == f"{base}/acme-plumbing/index.html"


def test_build_s3_client_raises_without_credentials(monkeypatch):
    for var in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="R2 credentials are not configured"):
        deployer.build_s3_client()


def test_build_s3_client_creates_boto3_client(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")
    get_settings.cache_clear()
    client = deployer.build_s3_client()
    assert hasattr(client, "put_object")


# ── brand palette (regenerate feature) ───────────────────────────────────────


def _lum(hexv):
    return sum(int(hexv[i : i + 2], 16) for i in (1, 3, 5))


def test_build_brand_palette_lowercases_and_shades():
    p = deployer.build_brand_palette({"primary": "#1D4ED8", "secondary": "#0F172A"})
    assert p["primary"] == "#1d4ed8"
    assert p["secondary"] == "#0f172a"
    for key in ("primary", "primary_dark", "primary_light", "secondary"):
        assert deployer._HEX_COLOR_RE.match(p[key])
    # dark is darker, light is lighter than the base color
    assert _lum(p["primary_dark"]) < _lum(p["primary"]) < _lum(p["primary_light"])


def test_build_brand_palette_defaults_when_missing_or_invalid():
    p = deployer.build_brand_palette(None)
    assert p["primary"] == deployer.DEFAULT_BRAND_PRIMARY
    assert p["secondary"] == deployer.DEFAULT_BRAND_SECONDARY

    bad = deployer.build_brand_palette({"primary": "not-a-color", "secondary": "#zzz"})
    assert bad["primary"] == deployer.DEFAULT_BRAND_PRIMARY
    assert bad["secondary"] == deployer.DEFAULT_BRAND_SECONDARY


def test_build_brand_palette_partial_keeps_valid_key():
    p = deployer.build_brand_palette({"primary": "#7c2d12"})
    assert p["primary"] == "#7c2d12"
    assert p["secondary"] == deployer.DEFAULT_BRAND_SECONDARY


def test_shade_extremes_stay_in_range():
    assert deployer._shade("#000000", -1.0) == "#000000"
    assert deployer._shade("#ffffff", 1.0) == "#ffffff"
    assert deployer._shade("#808080", -2.0) == "#000000"  # strength clamped


# ── deploy_site_assets (regenerate feature) ──────────────────────────────────


def test_deploy_site_assets_uploads_relative_keys_and_content_types():
    fake = FakeS3()
    assets = [
        {"key": "logo.png", "data": b"png-bytes", "content_type": "image/png"},
        {"key": "hero.jpg", "data": b"jpg-bytes"},  # no ctype → octet-stream fallback
    ]
    returned = deployer.deploy_site_assets("acme-plumbing", assets, s3_client=fake)

    assert returned == ["assets/logo.png", "assets/hero.jpg"]
    assert len(fake.puts) == 2
    first, second = fake.puts
    assert first["Bucket"] == "solospark-previews"
    assert first["Key"] == "acme-plumbing/assets/logo.png"
    assert first["Body"] == b"png-bytes"
    assert first["ContentType"] == "image/png"
    assert second["Key"] == "acme-plumbing/assets/hero.jpg"
    assert second["ContentType"] == "application/octet-stream"


def test_deploy_site_assets_skips_malformed_entries():
    fake = FakeS3()
    returned = deployer.deploy_site_assets(
        "acme-plumbing",
        [
            {"key": "no-data.png"},          # missing bytes
            {"data": b"x"},                  # missing key
            {"key": "", "data": b"x"},       # empty key
            {"key": "ok.webp", "data": b"w"},
        ],
        s3_client=fake,
    )
    assert returned == ["assets/ok.webp"]
    assert [p["Key"] for p in fake.puts] == ["acme-plumbing/assets/ok.webp"]


# ── delete_site_objects (site-delete feature) ────────────────────────────────


class FakeS3Delete:
    """Fake S3 client supporting paginated list_objects_v2 + bulk delete."""

    PAGE_SIZE = 2

    def __init__(self, keys):
        self.all_keys = list(keys)
        self.list_calls = []
        self.deleted = []

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(kwargs)
        prefix = kwargs.get("Prefix", "")
        scoped = [k for k in self.all_keys if k.startswith(prefix)]
        start = int(kwargs["ContinuationToken"]) if "ContinuationToken" in kwargs else 0
        page = scoped[start : start + self.PAGE_SIZE]
        result = {"IsTruncated": False, "Contents": [{"Key": k} for k in page]}
        if start + self.PAGE_SIZE < len(self.all_keys):
            result["IsTruncated"] = True
            result["NextContinuationToken"] = str(start + self.PAGE_SIZE)
        return result

    def delete_objects(self, **kwargs):
        self.deleted.extend(obj["Key"] for obj in kwargs["Delete"]["Objects"])
        return {}


def test_delete_site_objects_removes_all_pages():
    fake = FakeS3Delete(
        [
            "acme-plumbing/index.html",
            "acme-plumbing/assets/logo.png",
            "acme-plumbing/assets/hero.jpg",
            "other-site/index.html",  # different prefix — must never be touched
        ]
    )
    removed = deployer.delete_site_objects("acme-plumbing", s3_client=fake)

    assert removed == 3
    assert fake.deleted == [
        "acme-plumbing/index.html",
        "acme-plumbing/assets/logo.png",
        "acme-plumbing/assets/hero.jpg",
    ]
    # 3 keys at page size 2 → two list calls, the second carrying the token
    assert len(fake.list_calls) == 2
    assert fake.list_calls[0]["Prefix"] == "acme-plumbing/"
    assert "ContinuationToken" not in fake.list_calls[0]
    assert fake.list_calls[1].get("ContinuationToken") == "2"
    assert not any(k.startswith("other-site/") for k in fake.deleted)


def test_delete_site_objects_blank_slug_is_noop():
    fake = FakeS3Delete([])
    assert deployer.delete_site_objects("", s3_client=fake) == 0
    assert deployer.delete_site_objects(None, s3_client=fake) == 0
    assert fake.list_calls == []
    assert fake.deleted == []


def test_delete_site_objects_empty_prefix_returns_zero():
    fake = FakeS3Delete([])
    assert deployer.delete_site_objects("ghost-site", s3_client=fake) == 0
    assert fake.deleted == []


# ── render with brand + assets (regenerate feature) ─────────────────────────


def test_render_with_brand_palette_assets_and_trust_row():
    biz = {**BUSINESS, "city": "Eugene", "rating": 4.6, "review_count": 30}
    copy = {
        **COPY,
        "brand": {"primary": "#1e3a8a", "secondary": "#0f172a"},
        "logo_url": "assets/logo.png",
        "hero_image_url": "assets/hero.jpg",
        "about_images": ["assets/about1.webp"],
    }
    html = deployer.render_site_html(biz, copy)

    # assets referenced with bucket-relative paths
    assert 'src="assets/logo.png"' in html
    assert 'src="assets/hero.jpg"' in html
    assert 'src="assets/about1.webp"' in html
    # brand color wired into the Tailwind config
    assert "#1e3a8a" in html
    # trust row rendered from rating/review_count/city
    assert "4.6" in html
    assert "30" in html
    assert "<title>Acme Plumbing | Plumber — Eugene</title>" in html


def test_render_old_copy_uses_default_brand_and_no_assets():
    html = deployer.render_site_html(BUSINESS, COPY)  # pre-feature copy shape
    assert deployer.DEFAULT_BRAND_PRIMARY in html
    assert "assets/" not in html


# ── Section composition (AI-picked layouts) ──────────────────────────────────

RESTAURANT_BIZ = {
    "name": "Luigi's Pizzeria",
    "slug": "luigis-pizzeria",
    "category": "Restaurant",
    "address": "450 SE Main St, Portland, OR 97214",
    "phone": "(503) 555-0188",
}

RESTAURANT_COPY = {
    **COPY,
    "tagline": "Wood-fired pizza in Portland since 1987",
    "hero_headline": "Real Pizza, Baked Over Live Fire",
    "hero_subheadline": "Hand-stretched dough and San Marzano tomatoes.",
    # 5 sections so every labeled one fits the nav's 4-link cap
    "layout": ["hero", "menu", "hours_location", "services", "contact"],
    "menu_items": [
        {"name": "Margherita Pizza", "description": "San Marzano tomato, mozzarella", "price": "$14"},
        {"name": "Burrata Salad", "description": "", "price": "$12"},
    ],
    "hours": [{"day": "Monday", "hours": "11am-9pm"}],
    "gallery_images": ["assets/gallery-1.jpg", "assets/gallery-2.jpg"],
}


def test_render_restaurant_shows_menu_hours_and_nav():
    html = deployer.render_site_html(RESTAURANT_BIZ, RESTAURANT_COPY)
    # menu section with dishes and prices
    assert 'id="menu"' in html
    assert "Our Menu" in html
    assert "Margherita Pizza" in html
    assert "$14" in html
    # hours & location section with the day row
    assert 'id="hours_location"' in html
    assert "Hours &amp; Location" in html
    assert "11am-9pm" in html
    # dynamic nav includes the new anchors, Contact stays last
    assert 'href="#menu"' in html
    assert 'href="#hours_location"' in html
    assert html.rindex('href="#contact"') > html.find('href="#menu"')


def test_render_restaurant_without_menu_data_omits_menu_section():
    # no menu_items → the data gate drops the section entirely (no empty block)
    copy = {k: v for k, v in RESTAURANT_COPY.items() if k != "menu_items"}
    html = deployer.render_site_html(RESTAURANT_BIZ, copy)
    assert 'id="menu"' not in html
    assert "Our Menu" not in html
    assert 'href="#menu"' not in html


def test_render_trade_legacy_copy_keeps_classic_order():
    # plumber with pre-feature copy (no layout/menu/hours/gallery keys):
    # classic template order, no new sections, no gallery anchor
    html = deployer.render_site_html(BUSINESS, COPY)
    assert 'id="menu"' not in html
    assert 'id="hours_location"' not in html
    assert 'id="gallery"' not in html
    services_i = html.find('id="services"')
    about_i = html.find('id="about"')
    contact_i = html.find('id="contact"')
    assert 0 <= services_i < about_i < contact_i
    # classic nav: Services / About / Contact
    assert 'href="#services"' in html
    assert 'href="#about"' in html
    assert "Why Us" not in html


def test_render_gallery_hides_about_image_duplicate():
    copy = {
        **COPY,
        "layout": ["hero", "services", "gallery", "about", "cta_band", "contact"],
        "about_images": ["assets/about.jpg"],
        "gallery_images": ["assets/gallery-1.jpg"],
    }
    html = deployer.render_site_html(BUSINESS, copy)
    assert 'src="assets/gallery-1.jpg"' in html
    # the about section hides its photo when the gallery shows it
    assert 'src="assets/about.jpg"' not in html


def test_render_gallery_falls_back_to_about_images():
    copy = {
        **COPY,
        "layout": ["hero", "services", "gallery", "about", "cta_band", "contact"],
        "about_images": ["assets/about.jpg"],
    }
    html = deployer.render_site_html(BUSINESS, copy)
    assert 'id="gallery"' in html
    # the photo renders exactly once — inside the gallery fallback
    assert html.count('src="assets/about.jpg"') == 1


# ── check_html_syntax (pre-upload gate) ──────────────────────────────────────


def test_check_html_syntax_passes_rendered_pages():
    # every layout variant must clear the gate or no site could be deployed
    plain = deployer.render_site_html(BUSINESS, COPY)
    assert deployer.check_html_syntax(plain) == []

    restaurant = {
        **COPY,
        "layout": ["hero", "menu", "hours_location", "services", "contact"],
        "menu_items": [{"name": "Margherita", "description": "", "price": "$14"}],
        "hours": [{"day": "Monday", "hours": "9am-5pm"}],
    }
    assert deployer.check_html_syntax(deployer.render_site_html(BUSINESS, restaurant)) == []

    gallery = {
        **COPY,
        "layout": ["hero", "services", "gallery", "about", "cta_band", "contact"],
        "gallery_images": ["assets/g1.jpg"],
    }
    assert deployer.check_html_syntax(deployer.render_site_html(BUSINESS, gallery)) == []


def test_check_html_syntax_detects_unclosed_tag():
    html = "<!DOCTYPE html><html><head><title>t</title></head><body><div>hi"
    problems = deployer.check_html_syntax(html)
    assert any("unclosed" in p for p in problems)


def test_check_html_syntax_detects_mismatched_tag():
    html = (
        "<!DOCTYPE html><html><head><title>t</title></head>"
        "<body><div><span></div></span></body></html>"
    )
    problems = deployer.check_html_syntax(html)
    assert any("mismatched" in p for p in problems)


def test_check_html_syntax_detects_jinja_leftovers():
    html = (
        "<!DOCTYPE html><html><head><title>t</title></head>"
        "<body>{{ city }}</body></html>"
    )
    problems = deployer.check_html_syntax(html)
    assert any("Jinja" in p for p in problems)


def test_check_html_syntax_ignores_tags_inside_comments():
    html = (
        "<!DOCTYPE html><html><head><title>t</title></head>"
        "<body><!-- <div> draft --></body></html>"
    )
    assert deployer.check_html_syntax(html) == []

