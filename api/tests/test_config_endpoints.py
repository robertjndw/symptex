import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.symptex_db import get_symptex_db
from app.db.symptex_models import SymptexBase, SymptexConfig
from app.routers import config


def _build_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    SymptexBase.metadata.create_all(bind=engine)

    app = FastAPI()
    app.include_router(config.router, prefix="/api/v1")

    def _override_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_symptex_db] = _override_db
    return TestClient(app), TestingSessionLocal


def test_config_upsert_creates_then_updates_existing_case():
    client, session_local = _build_client()

    create_response = client.post(
        "/api/v1/config",
        json={
            "caseId": 11,
            "model": "model-a",
            "talkativeness": "ausgewogen",
            "condition": "default",
        },
    )

    assert create_response.status_code == 200
    assert create_response.json() == {"caseId": 11, "updated": False}

    db = session_local()
    created = db.query(SymptexConfig).filter(SymptexConfig.case_id == 11).first()
    assert created is not None
    assert created.model == "model-a"
    assert created.talkativeness == "ausgewogen"
    assert created.condition == "default"
    db.close()

    update_response = client.post(
        "/api/v1/config",
        json={
            "caseId": 11,
            "model": "model-b",
            "talkativeness": "kurz angebunden",
            "condition": "alzheimer",
        },
    )

    assert update_response.status_code == 200
    assert update_response.json() == {"caseId": 11, "updated": True}

    db = session_local()
    updated = db.query(SymptexConfig).filter(SymptexConfig.case_id == 11).first()
    assert updated is not None
    assert updated.model == "model-b"
    assert updated.talkativeness == "kurz angebunden"
    assert updated.condition == "alzheimer"
    db.close()


def test_get_config_returns_case_config_when_present():
    client, _ = _build_client()

    client.post(
        "/api/v1/config",
        json={
            "caseId": 44,
            "model": "model-a",
            "talkativeness": "ausgewogen",
            "condition": "default",
        },
    )

    get_response = client.get("/api/v1/config", params={"caseId": 44})

    assert get_response.status_code == 200
    assert get_response.json() == {
        "caseId": 44,
        "model": "model-a",
        "talkativeness": "ausgewogen",
        "condition": "default",
    }


def test_get_config_returns_404_when_case_config_missing():
    client, _ = _build_client()

    get_response = client.get("/api/v1/config", params={"caseId": 999})

    assert get_response.status_code == 404
    assert "No config found for case 999" in get_response.text


def test_config_rejects_payload_with_extra_or_missing_or_wrong_keys():
    client, session_local = _build_client()

    extra_key_response = client.post(
        "/api/v1/config",
        json={
            "caseId": 22,
            "model": "model-a",
            "talkativeness": "ausgewogen",
            "condition": "default",
            "unexpected": "value",
        },
    )
    assert extra_key_response.status_code == 422

    missing_key_response = client.post(
        "/api/v1/config",
        json={
            "caseId": 22,
            "model": "model-a",
            "talkativeness": "ausgewogen",
        },
    )
    assert missing_key_response.status_code == 422

    snake_case_response = client.post(
        "/api/v1/config",
        json={
            "case_id": 22,
            "model": "model-a",
            "talkativeness": "ausgewogen",
            "condition": "default",
        },
    )
    assert snake_case_response.status_code == 422

    db = session_local()
    assert db.query(SymptexConfig).count() == 0
    db.close()


def test_delete_config_removes_existing_and_is_idempotent_for_missing_case():
    client, session_local = _build_client()

    client.post(
        "/api/v1/config",
        json={
            "caseId": 33,
            "model": "model-a",
            "talkativeness": "ausgewogen",
            "condition": "default",
        },
    )

    delete_existing = client.delete("/api/v1/config/33")
    assert delete_existing.status_code == 200

    db = session_local()
    assert db.query(SymptexConfig).filter(SymptexConfig.case_id == 33).first() is None
    db.close()

    delete_missing = client.delete("/api/v1/config/33")
    assert delete_missing.status_code == 200
