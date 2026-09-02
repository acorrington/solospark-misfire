"""Central environment configuration for SoloSpark.

Loads `.env` from the project root (if present) and exposes typed,
lazy-read accessors so modules never call `os.getenv` directly.
Every value can be overridden by a real environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at import time so tests can run without it
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _get(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Snapshot of all runtime settings."""

    # Server
    port: int = 8000
    secret_key: str = "solospark-local-secret-key"
    database_url: str = "sqlite:///solospark.db"

    # Local LLM (OpenAI-compatible endpoint, e.g. LM Studio / Unsloth)
    local_llm_url: str = "http://localhost:1234/v1"
    local_llm_model: str = "local-model"
    local_llm_api_key: str = "lm-studio"

    # Google Places API (New)
    places_api_key: str | None = None

    # Cloudflare R2 / AWS S3
    r2_endpoint_url: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket_name: str = "solospark-previews"
    r2_public_base_url: str = "https://preview.solospark.net"

    # Outreach email
    resend_api_key: str | None = None
    outreach_from_email: str = "aaron@solosparkmail.com"
    business_physical_address: str = "SoloSpark LLC, Roseburg, OR 97470"
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    outreach_rate_limit_per_hour: int = 5

    # Stripe
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) a Settings snapshot from the environment."""
    return Settings(
        port=_get_int("PORT", 8000),
        secret_key=_get("SECRET_KEY", "solospark-local-secret-key") or "solospark-local-secret-key",
        database_url=_get("DATABASE_URL", "sqlite:///solospark.db") or "sqlite:///solospark.db",
        local_llm_url=_get("LOCAL_LLM_URL", "http://localhost:1234/v1") or "http://localhost:1234/v1",
        local_llm_model=_get("LOCAL_LLM_MODEL", "local-model") or "local-model",
        local_llm_api_key=_get("LOCAL_LLM_API_KEY", "lm-studio") or "lm-studio",
        places_api_key=_get("PLACES_API_KEY"),
        r2_endpoint_url=_get("R2_ENDPOINT_URL"),
        r2_access_key_id=_get("R2_ACCESS_KEY_ID"),
        r2_secret_access_key=_get("R2_SECRET_ACCESS_KEY"),
        r2_bucket_name=_get("R2_BUCKET_NAME", "solospark-previews") or "solospark-previews",
        r2_public_base_url=_get("R2_PUBLIC_BASE_URL", "https://preview.solospark.net")
        or "https://preview.solospark.net",
        resend_api_key=_get("RESEND_API_KEY"),
        outreach_from_email=_get("OUTREACH_FROM_EMAIL", "aaron@solosparkmail.com")
        or "aaron@solosparkmail.com",
        business_physical_address=_get("BUSINESS_PHYSICAL_ADDRESS", "SoloSpark LLC, Roseburg, OR 97470")
        or "SoloSpark LLC, Roseburg, OR 97470",
        smtp_host=_get("SMTP_HOST"),
        smtp_port=_get_int("SMTP_PORT", 587),
        smtp_user=_get("SMTP_USER"),
        smtp_password=_get("SMTP_PASSWORD"),
        outreach_rate_limit_per_hour=_get_int("OUTREACH_RATE_LIMIT_PER_HOUR", 5),
        stripe_secret_key=_get("STRIPE_SECRET_KEY"),
        stripe_webhook_secret=_get("STRIPE_WEBHOOK_SECRET"),
    )


def reload_settings() -> Settings:
    """Clear the cache and rebuild settings (used by tests after env changes)."""
    get_settings.cache_clear()
    return get_settings()
