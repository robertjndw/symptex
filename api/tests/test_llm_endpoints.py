import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import chat
from chains import llm


def _configure_ollama_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_OLLAMA_MODELS", "model-a,model-b")
    llm.clear_llm_config_cache()


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1")
    return TestClient(app)


def test_available_models_endpoint_returns_active_provider(monkeypatch):
    _configure_ollama_env(monkeypatch)
    client = _build_client()

    response = client.get("/api/v1/available-models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["default_model"] == "model-a"
    assert payload["models"] == [
        {"id": "model-a", "label": "model-a", "default": True},
        {"id": "model-b", "label": "model-b", "default": False},
    ]


def test_chat_rejects_unknown_model_before_db_access(monkeypatch):
    _configure_ollama_env(monkeypatch)
    client = _build_client()

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hallo",
            "model": "does-not-exist",
            "condition": "default",
            "talkativeness": "ausgewogen",
            "patient_file_id": 1,
            "session_id": "session-1",
        },
    )

    assert response.status_code == 400
    assert "Invalid model 'does-not-exist'" in response.text


def test_eval_rejects_unknown_model(monkeypatch):
    _configure_ollama_env(monkeypatch)
    client = _build_client()

    response = client.post(
        "/api/v1/eval",
        json={"model": "does-not-exist", "messages": []},
    )

    assert response.status_code == 400
    assert "Invalid model 'does-not-exist'" in response.text


def test_eval_accepts_valid_model_and_streams(monkeypatch):
    _configure_ollama_env(monkeypatch)

    async def fake_eval_history(messages, model):
        assert model == "model-a"
        yield "ok"

    monkeypatch.setattr(chat, "eval_history", fake_eval_history)
    client = _build_client()

    response = client.post(
        "/api/v1/eval",
        json={"model": "model-a", "messages": []},
    )

    assert response.status_code == 200
    assert "ok" in response.text


def test_eval_accepts_mixed_patient_and_assistant_roles(monkeypatch):
    _configure_ollama_env(monkeypatch)

    async def fake_eval_history(messages, model):
        # user + patient + legacy assistant are all forwarded as LC messages
        assert len(messages) == 3
        yield "ok"

    monkeypatch.setattr(chat, "eval_history", fake_eval_history)
    client = _build_client()

    response = client.post(
        "/api/v1/eval",
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
    assert "ok" in response.text
