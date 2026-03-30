import os
import asyncio
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.db import get_db
from app.db.models import Case, ChatMessage, ChatSession, PatientFile
from app.routers import chat
from app.services import chat_execution
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
        self.case = SimpleNamespace(
            id=3,
            patient_file_id=3,
            patient_file=self.patient,
            symptex_config=None,
        )

    def query(self, model):
        if model is Case:
            return _FakeQuery(first_value=self.case)
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
    monkeypatch.setenv("LLM_OLLAMA_MODEL", "model-a")
    llm.clear_llm_config_cache()


def _configure_runtime_defaults(monkeypatch, *, condition: str, talkativeness: str):
    monkeypatch.setenv("SYMPTEX_DEFAULT_CONDITION", condition)
    monkeypatch.setenv("SYMPTEX_DEFAULT_TALKATIVENESS", talkativeness)


def _build_client(fake_db):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1")

    def _override_db():
        yield fake_db

    app.dependency_overrides[get_db] = _override_db
    return TestClient(app)


def _runtime_chat_payload(*, case_id: int = 3, message: str = "Hallo"):
    return {
        "message": message,
        "case_id": case_id,
        "session_id": "session-1",
    }


def test_chat_uses_case_symptex_config_when_enabled(monkeypatch):
    _configure_llm_env(monkeypatch)
    _configure_runtime_defaults(monkeypatch, condition="default", talkativeness="ausgewogen")
    captured = {}

    async def fake_stream_response(*args, **kwargs):
        captured.update(kwargs)
        yield "Antwort"

    monkeypatch.setattr(chat_execution, "stream_response", fake_stream_response)
    monkeypatch.setattr(chat_execution, "has_anamdocs", lambda *_: False)
    monkeypatch.setattr(chat_execution, "format_patient_details", lambda _: "mocked-patient-details")

    fake_db = _FakeDB()
    fake_db.case.symptex_config = SimpleNamespace(
        enabled=True,
        llm_model="model-b",
        condition="alzheimer",
        talkativeness="ausschweifend",
    )
    client = _build_client(fake_db)

    response = client.post("/api/v1/chat", json=_runtime_chat_payload())

    assert response.status_code == 200
    assert captured["model"] == "model-b"
    assert captured["condition"] == "alzheimer"
    assert captured["talkativeness"] == "ausschweifend"


def test_chat_uses_defaults_when_symptex_config_missing(monkeypatch):
    _configure_llm_env(monkeypatch)
    _configure_runtime_defaults(monkeypatch, condition="verdraengung", talkativeness="kurz angebunden")
    captured = {}

    async def fake_stream_response(*args, **kwargs):
        captured.update(kwargs)
        yield "Antwort"

    monkeypatch.setattr(chat_execution, "stream_response", fake_stream_response)
    monkeypatch.setattr(chat_execution, "has_anamdocs", lambda *_: False)
    monkeypatch.setattr(chat_execution, "format_patient_details", lambda _: "mocked-patient-details")

    fake_db = _FakeDB()
    client = _build_client(fake_db)

    response = client.post("/api/v1/chat", json=_runtime_chat_payload())

    assert response.status_code == 200
    assert captured["model"] == "model-a"
    assert captured["condition"] == "verdraengung"
    assert captured["talkativeness"] == "kurz angebunden"


def test_chat_uses_defaults_when_symptex_config_disabled(monkeypatch):
    _configure_llm_env(monkeypatch)
    _configure_runtime_defaults(monkeypatch, condition="default", talkativeness="ausgewogen")
    captured = {}

    async def fake_stream_response(*args, **kwargs):
        captured.update(kwargs)
        yield "Antwort"

    monkeypatch.setattr(chat_execution, "stream_response", fake_stream_response)
    monkeypatch.setattr(chat_execution, "has_anamdocs", lambda *_: False)
    monkeypatch.setattr(chat_execution, "format_patient_details", lambda _: "mocked-patient-details")

    fake_db = _FakeDB()
    fake_db.case.symptex_config = SimpleNamespace(
        enabled=False,
        llm_model="model-b",
        condition="alzheimer",
        talkativeness="ausschweifend",
    )
    client = _build_client(fake_db)

    response = client.post("/api/v1/chat", json=_runtime_chat_payload())

    assert response.status_code == 200
    assert captured["model"] == "model-a"
    assert captured["condition"] == "default"
    assert captured["talkativeness"] == "ausgewogen"


def test_chat_falls_back_per_field_for_invalid_symptex_config_values(monkeypatch):
    _configure_llm_env(monkeypatch)
    _configure_runtime_defaults(monkeypatch, condition="verdraengung", talkativeness="kurz angebunden")
    captured = {}

    async def fake_stream_response(*args, **kwargs):
        captured.update(kwargs)
        yield "Antwort"

    monkeypatch.setattr(chat_execution, "stream_response", fake_stream_response)
    monkeypatch.setattr(chat_execution, "has_anamdocs", lambda *_: False)
    monkeypatch.setattr(chat_execution, "format_patient_details", lambda _: "mocked-patient-details")

    fake_db = _FakeDB()
    fake_db.case.symptex_config = SimpleNamespace(
        enabled=True,
        llm_model="not-available",
        condition="not-supported",
        talkativeness="too-chatty",
    )
    client = _build_client(fake_db)

    response = client.post("/api/v1/chat", json=_runtime_chat_payload())

    assert response.status_code == 200
    assert captured["model"] == "model-a"
    assert captured["condition"] == "verdraengung"
    assert captured["talkativeness"] == "kurz angebunden"


def test_chat_rejects_empty_message(monkeypatch):
    _configure_llm_env(monkeypatch)
    fake_db = _FakeDB()
    client = _build_client(fake_db)

    response = client.post("/api/v1/chat", json=_runtime_chat_payload(message="   "))

    assert response.status_code == 400


def test_chat_returns_404_when_case_not_found(monkeypatch):
    _configure_llm_env(monkeypatch)
    fake_db = _FakeDB()
    fake_db.case = None
    client = _build_client(fake_db)

    response = client.post("/api/v1/chat", json=_runtime_chat_payload(case_id=999))

    assert response.status_code == 404
    assert "Case not found" in response.text


def test_stream_response_strips_think_tags_across_chunks(monkeypatch):
    class _FakeModel:
        async def astream(self, *_args, **_kwargs):
            yield (
                "messages",
                (
                    SimpleNamespace(content="<thi"),
                    {"langgraph_node": chat_execution.TARGET_NODE},
                ),
            )
            yield (
                "messages",
                (
                    SimpleNamespace(content="nk>hidden"),
                    {"langgraph_node": chat_execution.TARGET_NODE},
                ),
            )
            yield (
                "messages",
                (
                    SimpleNamespace(content=" text</thin"),
                    {"langgraph_node": chat_execution.TARGET_NODE},
                ),
            )
            yield (
                "messages",
                (
                    SimpleNamespace(content="k>Hello there"),
                    {"langgraph_node": chat_execution.TARGET_NODE},
                ),
            )

    monkeypatch.setattr(chat_execution, "build_symptex_model", lambda _docs_cache: _FakeModel())

    async def _collect() -> str:
        parts = []
        async for chunk in chat_execution.stream_response(
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
                (
                    SimpleNamespace(content="normal text "),
                    {"langgraph_node": chat_execution.TARGET_NODE},
                ),
            )
            yield (
                "messages",
                (SimpleNamespace(content="<t"), {"langgraph_node": chat_execution.TARGET_NODE}),
            )

    monkeypatch.setattr(chat_execution, "build_symptex_model", lambda _docs_cache: _FakeModel())

    async def _collect() -> str:
        parts = []
        async for chunk in chat_execution.stream_response(
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
