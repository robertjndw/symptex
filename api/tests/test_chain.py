import os
import asyncio
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.db import get_db
from app.db.models import ChatMessage, ChatSession, PatientFile
from app.routers import chat
from chains import llm


class _FakeQuery:
    def __init__(self, first_value=None, all_value=None):
        self._first_value = first_value
        self._all_value = all_value or []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self._first_value

    def all(self):
        return list(self._all_value)

    def delete(self):
        return 1


class _FakeDB:
    def __init__(self):
        self.session = None
        self.patient = SimpleNamespace(id=3)

    def query(self, model):
        if model is PatientFile:
            return _FakeQuery(first_value=self.patient)
        if model is ChatSession:
            return _FakeQuery(first_value=self.session)
        if model is ChatMessage:
            return _FakeQuery(all_value=[])
        return _FakeQuery()

    def add(self, obj):
        if isinstance(obj, ChatSession):
            self.session = obj

    def commit(self):
        return None

    def refresh(self, obj):
        return None

    def rollback(self):
        return None


def _configure_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_OLLAMA_MODELS", "model-a,model-b")
    llm.clear_llm_config_cache()


def _build_client(fake_db):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1")

    def _override_db():
        yield fake_db

    app.dependency_overrides[get_db] = _override_db
    return TestClient(app)


def test_chat_rejects_invalid_model(monkeypatch):
    _configure_llm_env(monkeypatch)
    fake_db = _FakeDB()
    client = _build_client(fake_db)

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hallo",
            "model": "invalid-model",
            "condition": "default",
            "talkativeness": "ausgewogen",
            "patient_file_id": 3,
            "session_id": "session-1",
        },
    )

    assert response.status_code == 400


def test_chat_rejects_invalid_condition(monkeypatch):
    _configure_llm_env(monkeypatch)
    fake_db = _FakeDB()
    client = _build_client(fake_db)

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hallo",
            "model": "model-a",
            "condition": "invalid-condition",
            "talkativeness": "ausgewogen",
            "patient_file_id": 3,
            "session_id": "session-1",
        },
    )

    assert response.status_code == 400


def test_chat_rejects_empty_message(monkeypatch):
    _configure_llm_env(monkeypatch)
    fake_db = _FakeDB()
    client = _build_client(fake_db)

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "   ",
            "model": "model-a",
            "condition": "default",
            "talkativeness": "ausgewogen",
            "patient_file_id": 3,
            "session_id": "session-1",
        },
    )

    assert response.status_code == 400


def test_chat_valid_request_streams_response(monkeypatch):
    _configure_llm_env(monkeypatch)

    async def fake_stream_response(*args, **kwargs):
        yield "Antwort"

    monkeypatch.setattr(chat, "stream_response", fake_stream_response)
    monkeypatch.setattr(chat, "has_anamdocs", lambda *_: False)
    monkeypatch.setattr(chat, "format_patient_details", lambda _: "mocked-patient-details")

    fake_db = _FakeDB()
    client = _build_client(fake_db)

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hallo",
            "model": "model-a",
            "condition": "default",
            "talkativeness": "ausgewogen",
            "patient_file_id": 3,
            "session_id": "session-1",
        },
    )

    assert response.status_code == 200
    assert "Antwort" in response.text


def test_stream_response_strips_think_tags_across_chunks(monkeypatch):
    class _FakeModel:
        async def astream(self, *_args, **_kwargs):
            yield (
                "messages",
                (SimpleNamespace(content="<thi"), {"langgraph_node": chat.TARGET_NODE}),
            )
            yield (
                "messages",
                (SimpleNamespace(content="nk>hidden"), {"langgraph_node": chat.TARGET_NODE}),
            )
            yield (
                "messages",
                (
                    SimpleNamespace(content=" text</thin"),
                    {"langgraph_node": chat.TARGET_NODE},
                ),
            )
            yield (
                "messages",
                (
                    SimpleNamespace(content="k>Hello there"),
                    {"langgraph_node": chat.TARGET_NODE},
                ),
            )

    monkeypatch.setattr(chat, "build_symptex_model", lambda _docs_cache: _FakeModel())

    async def _collect() -> str:
        parts = []
        async for chunk in chat.stream_response(
            model="model-a",
            condition="default",
            talkativeness="ausgewogen",
            patient_details="details",
            previous_messages=[],
            docs_available=False,
            docs_cache=SimpleNamespace(),
        ):
            parts.append(chunk)
        return "".join(parts)

    output = asyncio.run(_collect())
    assert output == "Hello there"


def test_stream_response_keeps_normal_text_with_partial_tag_prefix(monkeypatch):
    class _FakeModel:
        async def astream(self, *_args, **_kwargs):
            yield (
                "messages",
                (SimpleNamespace(content="normal text "), {"langgraph_node": chat.TARGET_NODE}),
            )
            yield (
                "messages",
                (SimpleNamespace(content="<t"), {"langgraph_node": chat.TARGET_NODE}),
            )

    monkeypatch.setattr(chat, "build_symptex_model", lambda _docs_cache: _FakeModel())

    async def _collect() -> str:
        parts = []
        async for chunk in chat.stream_response(
            model="model-a",
            condition="default",
            talkativeness="ausgewogen",
            patient_details="details",
            previous_messages=[],
            docs_available=False,
            docs_cache=SimpleNamespace(),
        ):
            parts.append(chunk)
        return "".join(parts)

    output = asyncio.run(_collect())
    assert output == "normal text <t"
