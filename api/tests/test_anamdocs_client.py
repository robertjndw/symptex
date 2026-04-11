import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.anamdocs_client import AnamDocsClient, ListAnamDocsError


class FakeResponse:
    def __init__(self, status_code: int, json_data=None, content: bytes = b"", json_error: bool = False):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("Invalid JSON")
        return self._json_data


def _sample_payload():
    return [
        {
            "id": 7,
            "category": "Labor",
            "originalName": "report.pdf",
            "anamnesisId": 3,
            "path": "anamnesis-docs/1/report.pdf",
        }
    ]


def test_list_anamdocs_401_with_debug_disabled_raises():
    client = AnamDocsClient(api_base_url="http://ilvi.local", debug_login_enabled=False)
    session = Mock()
    session.get.side_effect = [FakeResponse(401)]
    client._session = session

    with pytest.raises(ListAnamDocsError, match="Unexpected status while listing AnamDocs: HTTP 401"):
        client.list_anamdocs(1)

    assert session.get.call_count == 1


def test_list_anamdocs_401_then_debug_login_success_retries_once():
    client = AnamDocsClient(
        api_base_url="http://ilvi.local",
        debug_login_enabled=True,
        debug_login_tum_id="ADMIN1234",
        debug_login_role="admin",
        debug_login_first_name="Symptex",
        debug_login_last_name="Debug",
    )
    session = Mock()
    session.get.side_effect = [
        FakeResponse(401),
        FakeResponse(200),
        FakeResponse(200, json_data=_sample_payload()),
    ]
    client._session = session

    docs = client.list_anamdocs(1)

    assert len(docs) == 1
    assert docs[0].id == 7
    assert docs[0].original_name == "report.pdf"
    assert session.get.call_count == 3

    debug_call = session.get.call_args_list[1]
    assert "/auth/debug-login" in debug_call.args[0]
    assert debug_call.kwargs["params"] == {
        "tumID": "ADMIN1234",
        "firstName": "Symptex",
        "lastName": "Debug",
        "role": "admin",
    }
    assert "redirect_uri" not in debug_call.kwargs["params"]


def test_list_anamdocs_401_then_debug_login_failure_returns_401_error():
    client = AnamDocsClient(api_base_url="http://ilvi.local", debug_login_enabled=True)
    session = Mock()
    session.get.side_effect = [
        FakeResponse(401),
        FakeResponse(500),
    ]
    client._session = session

    with pytest.raises(ListAnamDocsError, match="Unexpected status while listing AnamDocs: HTTP 401"):
        client.list_anamdocs(1)

    assert session.get.call_count == 2


def test_list_anamdocs_second_401_after_retry_does_not_loop():
    client = AnamDocsClient(api_base_url="http://ilvi.local", debug_login_enabled=True)
    session = Mock()
    session.get.side_effect = [
        FakeResponse(401),
        FakeResponse(200),
        FakeResponse(401),
    ]
    client._session = session

    with pytest.raises(ListAnamDocsError, match="Unexpected status while listing AnamDocs: HTTP 401"):
        client.list_anamdocs(1)

    assert session.get.call_count == 3


def test_from_env_debug_login_defaults(monkeypatch):
    monkeypatch.setenv("ILUVI_API_BASE_URL", "http://ilvi.local")
    monkeypatch.delenv("ILUVI_DEBUG_LOGIN_ENABLED", raising=False)
    monkeypatch.delenv("ILUVI_DEBUG_LOGIN_TUM_ID", raising=False)
    monkeypatch.delenv("ILUVI_DEBUG_LOGIN_ROLE", raising=False)
    monkeypatch.delenv("ILUVI_DEBUG_LOGIN_FIRST_NAME", raising=False)
    monkeypatch.delenv("ILUVI_DEBUG_LOGIN_LAST_NAME", raising=False)

    client = AnamDocsClient.from_env()

    assert client.debug_login_enabled is False
    assert client.debug_login_tum_id == "ADMIN1234"
    assert client.debug_login_role == "admin"
    assert client.debug_login_first_name == "Symptex"
    assert client.debug_login_last_name == "Debug"


def test_from_env_debug_login_overrides(monkeypatch):
    monkeypatch.setenv("ILUVI_API_BASE_URL", "http://ilvi.local")
    monkeypatch.setenv("ILUVI_DEBUG_LOGIN_ENABLED", "true")
    monkeypatch.setenv("ILUVI_DEBUG_LOGIN_TUM_ID", "CUSTOM123")
    monkeypatch.setenv("ILUVI_DEBUG_LOGIN_ROLE", "instructor")
    monkeypatch.setenv("ILUVI_DEBUG_LOGIN_FIRST_NAME", "Local")
    monkeypatch.setenv("ILUVI_DEBUG_LOGIN_LAST_NAME", "Runner")

    client = AnamDocsClient.from_env()

    assert client.debug_login_enabled is True
    assert client.debug_login_tum_id == "CUSTOM123"
    assert client.debug_login_role == "instructor"
    assert client.debug_login_first_name == "Local"
    assert client.debug_login_last_name == "Runner"


def test_from_env_accepts_quoted_boolean(monkeypatch):
    monkeypatch.setenv("ILUVI_API_BASE_URL", "http://ilvi.local")
    monkeypatch.setenv("ILUVI_DEBUG_LOGIN_ENABLED", "'true'")

    client = AnamDocsClient.from_env()

    assert client.debug_login_enabled is True
