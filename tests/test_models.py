"""Phase 1 tests — schema, enums, JSON helpers, dedup constraints."""

import json

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import (
    Base,
    Business,
    DealStage,
    OutreachEmail,
    SubscriptionStatus,
    get_engine,
    get_session_factory,
    init_db,
    reinit_db,
)


@pytest.fixture()
def db(tmp_path):
    """Fresh in-test database per test; resets the module-level factory."""
    engine, factory = reinit_db(f"sqlite:///{tmp_path / 'test.db'}")
    session = factory()
    yield session
    session.close()


def _make_business(**overrides) -> Business:
    defaults = dict(
        place_id="ChIJ-test-1",
        name="Acme Plumbing",
        slug="acme-plumbing",
        category="plumber",
        address="1 Main St, Eugene, OR",
        phone="(541) 555-0100",
    )
    defaults.update(overrides)
    return Business(**defaults)


def test_business_defaults(db):
    biz = _make_business()
    db.add(biz)
    db.commit()

    assert biz.id is not None
    assert biz.stage == DealStage.DISCOVERED
    assert biz.subscription_status == SubscriptionStatus.UNPAID
    assert biz.is_bad_site is False
    assert biz.no_website is False
    assert biz.rating == 0.0
    assert biz.review_count == 0
    assert biz.created_at is not None
    assert biz.updated_at is not None


def test_audit_flags_json_roundtrip(db):
    biz = _make_business()
    db.add(biz)
    db.commit()

    assert biz.audit_flags_list() == []
    biz.set_audit_flags(["Missing Mobile Viewport", "Stale Copyright (2012)"])
    db.commit()

    fresh = db.get(Business, biz.id)
    flags = fresh.audit_flags_list()
    assert flags == ["Missing Mobile Viewport", "Stale Copyright (2012)"]


def test_generated_copy_json_roundtrip(db):
    biz = _make_business()
    copy = {
        "tagline": "Honest work, fast service",
        "hero_headline": "Eugene's trusted plumbing team since 1998",
        "services": [{"title": "Drain Cleaning", "description": "x", "icon": "wrench"}],
    }
    db.add(biz)
    db.commit()

    assert biz.generated_copy_dict() is None
    biz.set_generated_copy(copy)
    db.commit()

    fresh = db.get(Business, biz.id)
    assert fresh.generated_copy_dict() == copy


def test_place_id_unique(db):
    db.add(_make_business())
    db.commit()
    db.add(_make_business(name="Other Plumbing"))  # same place_id
    with pytest.raises(IntegrityError):
        db.commit()


def test_slug_unique(db):
    db.add(_make_business())
    db.commit()
    db.add(
        _make_business(place_id="ChIJ-test-2", slug="acme-plumbing")
    )  # same slug, different place
    with pytest.raises(IntegrityError):
        db.commit()


def test_outreach_email_ledger(db):
    biz = _make_business()
    db.add(biz)
    db.commit()

    email = OutreachEmail(
        business_id=biz.id, recipient="owner@acme.com", subject="Hi", body="Hello"
    )
    db.add(email)
    db.commit()

    assert email.status == "queued"
    rows = (
        db.query(OutreachEmail).filter(OutreachEmail.business_id == biz.id).all()
    )
    assert len(rows) == 1
    assert rows[0].recipient == "owner@acme.com"


def test_init_db_creates_tables(tmp_path):
    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    engine, factory = init_db(url)
    session = factory()
    try:
        # Empty tables exist and are queryable.
        assert session.query(Business).count() == 0
        assert session.query(OutreachEmail).count() == 0
    finally:
        session.close()


def test_lazy_accessors_share_reinitialized_engine(tmp_path):
    """get_engine()/get_session_factory() return the reinitialized instances."""
    engine, factory = reinit_db(f"sqlite:///{tmp_path / 'lazy.db'}")
    assert get_engine() is engine
    assert get_session_factory() is factory
    session = get_session_factory()()
    try:
        assert session.query(Business).count() == 0
    finally:
        session.close()
