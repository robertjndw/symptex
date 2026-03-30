import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from app.routers import chat, dev_chat
from chains import llm


def _configure_ollama_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_OLLAMA_MODELS", "model-a,model-b")
    monkeypatch.setenv("LLM_OLLAMA_MODEL", "model-b")
    monkeypatch.setenv("DEV_FRONTEND_KEY", "dev-secret")
    llm.clear_llm_config_cache()


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(dev_chat.router, prefix="/api/v1")
    return TestClient(app)


def test_runtime_chat_uses_configured_model(monkeypatch):
    _configure_ollama_env(monkeypatch)

    async def fake_execute_chat(_db, **kwargs):
        assert kwargs["model"] == "model-b"
        return PlainTextResponse("ok", status_code=200)

    monkeypatch.setattr(chat, "execute_chat", fake_execute_chat)
    client = _build_client()

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hallo",
            "condition": "default",
            "talkativeness": "ausgewogen",
            "patient_file_id": 1,
            "session_id": "session-1",
        },
    )

    assert response.status_code == 200
    assert response.text == "ok"


def test_runtime_eval_uses_configured_model(monkeypatch):
    _configure_ollama_env(monkeypatch)

    async def fake_execute_eval(*, model, messages):
        assert model == "model-b"
        assert messages == []
        return PlainTextResponse("ok", status_code=200)

    monkeypatch.setattr(chat, "execute_eval", fake_execute_eval)
    client = _build_client()

    response = client.post("/api/v1/eval", json={"messages": []})

    assert response.status_code == 200
    assert response.text == "ok"


def test_dev_eval_rejects_missing_dev_key(monkeypatch):
    _configure_ollama_env(monkeypatch)
    client = _build_client()

    response = client.post(
        "/api/v1/dev/eval",
        json={"model": "model-a", "messages": []},
    )

    assert response.status_code == 401
    assert "Missing required header: X-Dev-Frontend-Key" in response.text


def test_dev_eval_rejects_invalid_dev_key(monkeypatch):
    _configure_ollama_env(monkeypatch)
    client = _build_client()

    response = client.post(
        "/api/v1/dev/eval",
        headers={"X-Dev-Frontend-Key": "wrong-key"},
        json={"model": "model-a", "messages": []},
    )

    assert response.status_code == 403
    assert "Invalid development frontend key." in response.text


def test_dev_eval_rejects_invalid_model(monkeypatch):
    _configure_ollama_env(monkeypatch)
    client = _build_client()

    response = client.post(
        "/api/v1/dev/eval",
        headers={"X-Dev-Frontend-Key": "dev-secret"},
        json={"model": "does-not-exist", "messages": []},
    )

    assert response.status_code == 400
    assert "Invalid model 'does-not-exist'" in response.text


def test_dev_chat_uses_selected_model(monkeypatch):
    _configure_ollama_env(monkeypatch)

    async def fake_execute_chat(_db, **kwargs):
        assert kwargs["model"] == "model-a"
        return PlainTextResponse("ok", status_code=200)

    monkeypatch.setattr(dev_chat, "execute_chat", fake_execute_chat)
    client = _build_client()

    response = client.post(
        "/api/v1/dev/chat",
        headers={"X-Dev-Frontend-Key": "dev-secret"},
        json={
            "message": "Hallo",
            "model": "model-a",
            "condition": "default",
            "talkativeness": "ausgewogen",
            "patient_file_id": 1,
            "session_id": "session-1",
        },
    )

    assert response.status_code == 200
    assert response.text == "ok"


def test_dev_eval_uses_selected_model(monkeypatch):
    _configure_ollama_env(monkeypatch)

    async def fake_execute_eval(*, model, messages):
        assert model == "model-a"
        assert len(messages) == 3
        return PlainTextResponse("ok", status_code=200)

    monkeypatch.setattr(dev_chat, "execute_eval", fake_execute_eval)
    client = _build_client()

    response = client.post(
        "/api/v1/dev/eval",
        headers={"X-Dev-Frontend-Key": "dev-secret"},
        json={
            "model": "model-a",
            "messages": [
                {"role": "user", "output": "Hallo"},
                {"role": "patient", "output": "Ich habe Schmerzen."},
                {"role": "assistant", "output": "Seit wann?"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.text == "ok"
