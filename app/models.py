"""Phase 1 — Database architecture & core data models.

SQLite + SQLAlchemy 2.0 schema tracking businesses, audit results, generated
copy, outreach state, and subscription/billing state.
"""

from __future__ import annotations

import enum
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import get_settings

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Enums ────────────────────────────────────────────────────────────────────


class DealStage(str, enum.Enum):
    """Sales pipeline stage for a discovered business."""

    DISCOVERED = "discovered"
    AUDITED = "audited"
    MOCKUP_READY = "mockup_ready"
    PITCH_APPROVED = "pitch_approved"
    CONTACTED = "contacted"
    REPLIED = "replied"
    WON = "won"
    LOST = "lost"


class SubscriptionStatus(str, enum.Enum):
    """Stripe subscription state for a closed client."""

    UNPAID = "unpaid"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"


# ── Models ───────────────────────────────────────────────────────────────────


class Business(Base):
    """A discovered local business and its journey through the pipeline."""

    __tablename__ = "businesses"

    id = Column(Integer, primary_key=True, index=True)
    place_id = Column(String(255), unique=True, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, index=True)
    category = Column(String(100))
    address = Column(String(255))
    phone = Column(String(50))
    current_website = Column(String(255), nullable=True)
    contact_email = Column(String(255), nullable=True)

    rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)

    # Audit results — audit_flags holds a JSON list of human-readable flags.
    audit_flags = Column(Text, default="")
    is_bad_site = Column(Boolean, default=False)
    no_website = Column(Boolean, default=False)  # LEAD_NO_WEBSITE marker

    # Generated content — generated_copy holds the LLM copy as JSON text.
    generated_copy = Column(Text, nullable=True)
    preview_url = Column(String(255), nullable=True)

    # Pipeline & billing
    stage = Column(SAEnum(DealStage), default=DealStage.DISCOVERED, index=True)
    stripe_customer_id = Column(String(255), nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    subscription_status = Column(
        SAEnum(SubscriptionStatus), default=SubscriptionStatus.UNPAID
    )

    # Outreach safety — set when the owner opts out via CAN-SPAM unsubscribe.
    opted_out = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # ── JSON helpers (stored as text per spec; parsed on demand) ──────────

    def audit_flags_list(self) -> list[str]:
        """Return the stored audit flags as a list of strings."""
        if not self.audit_flags:
            return []
        try:
            value = json.loads(self.audit_flags)
            return value if isinstance(value, list) else [str(value)]
        except (json.JSONDecodeError, TypeError):
            return [self.audit_flags]

    def set_audit_flags(self, flags: list[str]) -> None:
        self.audit_flags = json.dumps(list(flags))

    def generated_copy_dict(self) -> dict[str, Any] | None:
        """Return the stored LLM copy as a dict (or None if not generated)."""
        if not self.generated_copy:
            return None
        try:
            value = json.loads(self.generated_copy)
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def set_generated_copy(self, copy: dict[str, Any]) -> None:
        self.generated_copy = json.dumps(copy, ensure_ascii=False)

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return f"<Business id={self.id} name={self.name!r} stage={self.stage}>"


class OutreachEmail(Base):
    """Ledger of drafted/sent outreach emails.

    Serves two purposes:
      * dry-run review queue (status="queued") for manual approval, and
      * deduplication — a business with a "sent" row must not be re-emailed.
    """

    __tablename__ = "outreach_emails"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(
        Integer, ForeignKey("businesses.id"), index=True, nullable=False
    )
    recipient = Column(String(255), nullable=False)
    subject = Column(String(500))
    body = Column(Text)
    status = Column(String(20), default="queued")  # queued | sent | failed
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    sent_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return f"<OutreachEmail id={self.id} recipient={self.recipient!r} {self.status}>"


# ── Engine / session factory (lazy so importing never touches disk) ─────────


_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def init_db(db_url: str | None = None) -> tuple[Engine, sessionmaker]:
    """Create the engine + schema and return (engine, session_factory).

    Defaults to ``DATABASE_URL`` from the environment (sqlite:///solospark.db).
    """
    url = db_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, session_factory


def get_engine() -> Engine:
    """Return the lazily-initialized module-level engine."""
    global _engine, _session_factory
    if _engine is None:
        _engine, _session_factory = init_db()
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the lazily-initialized module-level session factory."""
    get_engine()
    assert _session_factory is not None
    return _session_factory


def reinit_db(db_url: str) -> tuple[Engine, sessionmaker]:
    """Replace the module-level engine (used by tests and CLI --db-url)."""
    global _engine, _session_factory
    _engine, _session_factory = init_db(db_url)
    return _engine, _session_factory


def get_db():
    """FastAPI dependency yielding a session that is always closed."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()
