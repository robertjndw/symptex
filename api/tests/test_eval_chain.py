import asyncio
import json
import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.utils.eval_json import EVAL_CATEGORIES, get_eval_categories_with_overall
from chains import eval_chain


def _build_valid_payload() -> dict:
    return {
        category: {
            "score": 4,
            "message": f"Bewertung fuer {category}",
            "verbesserungsvorschlag": f"Verbesserung fuer {category}",
        }
        for category in get_eval_categories_with_overall()
    }


class _FakeChain:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def ainvoke(self, payload: dict):
        self.calls.append(payload)
        if not self._responses:
            raise AssertionError("No fake response left for ainvoke().")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class _FakePrompt:
    def __init__(self, chain: _FakeChain):
        self.chain = chain
        self.last_llm = None

    def __or__(self, llm):
        self.last_llm = llm
        return self.chain


def _build_raw_response(content: object, *, tool_calls=None, additional_kwargs=None):
    return SimpleNamespace(
        content=content,
        response_metadata={"provider": "fake"},
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=[],
    )


def test_role_map_for_eval_prefixes_roles_and_preserves_other_messages():
    original_system = SystemMessage(content="System note")
    mapped = eval_chain._role_map_for_eval(
        [
            HumanMessage(content="Wie geht es Ihnen?"),
            AIMessage(content="Nicht gut."),
            original_system,
        ]
    )

    assert mapped[0].content == "Arzt: Wie geht es Ihnen?"
    assert mapped[1].content == "Patient: Nicht gut."
    assert mapped[2] is original_system


def test_get_eval_prompt_contains_categories_and_required_keys():
    prompt = eval_chain.get_eval_prompt()
    formatted_messages = prompt.format_messages(messages=[])
    system_content = formatted_messages[0].content

    for category in EVAL_CATEGORIES:
        assert category in system_content
    for key in get_eval_categories_with_overall():
        assert key in system_content
    assert "genau drei Feldern" in system_content
    assert "verbesserungsvorschlag" in system_content
    assert "Gib ausschließlich ein JSON-Objekt zurück" in system_content


def test_eval_history_structured_provider_returns_normalized_json(monkeypatch):
    payload = _build_valid_payload()
    chain = _FakeChain(
        [
            {
                "raw": _build_raw_response(json.dumps(payload, ensure_ascii=False)),
                "parsed": payload,
                "parsing_error": None,
            }
        ]
    )
    prompt = _FakePrompt(chain)
    build_calls: list[str] = []

    monkeypatch.setattr(eval_chain, "get_eval_prompt", lambda: prompt)
    monkeypatch.setattr(eval_chain, "get_llm_config", lambda: SimpleNamespace(provider="chatai"))
    monkeypatch.setattr(
        eval_chain,
        "_build_eval_llm",
        lambda model: build_calls.append(model) or object(),
    )

    result = asyncio.run(
        eval_chain.eval_history(
            [
                HumanMessage(content="Hallo"),
                AIMessage(content="Guten Tag"),
            ],
            model="model-a",
        )
    )

    assert build_calls == ["model-a"]
    assert len(chain.calls) == 1
    call_messages = chain.calls[0]["messages"]
    assert call_messages[0].content == "Arzt: Hallo"
    assert call_messages[1].content == "Patient: Guten Tag"
    assert json.loads(result) == payload
    assert result.startswith("{\n")


def test_eval_history_ollama_succeeds_on_first_attempt(monkeypatch):
    payload = _build_valid_payload()
    chain = _FakeChain([_build_raw_response(json.dumps(payload, ensure_ascii=False))])
    prompt = _FakePrompt(chain)
    llm_calls: list[tuple[str, float | None]] = []

    monkeypatch.setattr(eval_chain, "get_eval_prompt", lambda: prompt)
    monkeypatch.setattr(eval_chain, "get_llm_config", lambda: SimpleNamespace(provider="ollama"))
    monkeypatch.setattr(
        eval_chain,
        "get_llm",
        lambda model, temperature=None: llm_calls.append((model, temperature)) or object(),
    )

    result = asyncio.run(
        eval_chain.eval_history([HumanMessage(content="Wie geht es Ihnen?")], model="model-a")
    )

    assert llm_calls == [("model-a", eval_chain.OLLAMA_EVAL_TEMPERATURE)]
    assert len(chain.calls) == 1
    assert json.loads(result) == payload


def test_eval_history_ollama_retries_and_succeeds_on_second_attempt(monkeypatch):
    payload = _build_valid_payload()
    first_response = _build_raw_response("kein json")
    second_response = _build_raw_response(json.dumps(payload, ensure_ascii=False))
    chain = _FakeChain([first_response, second_response])
    prompt = _FakePrompt(chain)

    monkeypatch.setattr(eval_chain, "get_eval_prompt", lambda: prompt)
    monkeypatch.setattr(eval_chain, "get_llm_config", lambda: SimpleNamespace(provider="ollama"))
    monkeypatch.setattr(eval_chain, "get_llm", lambda *_args, **_kwargs: object())

    result = asyncio.run(
        eval_chain.eval_history([HumanMessage(content="Wie geht es Ihnen?")], model="model-a")
    )

    assert len(chain.calls) == 2
    second_attempt_messages = chain.calls[1]["messages"]
    assert isinstance(second_attempt_messages[-1], SystemMessage)
    assert second_attempt_messages[-1].content == eval_chain.OLLAMA_JSON_RETRY_INSTRUCTION
    assert json.loads(result) == payload


def test_eval_history_ollama_retries_when_first_json_is_missing_required_category(monkeypatch):
    payload = _build_valid_payload()
    invalid_payload = dict(payload)
    invalid_payload.pop("Gesprächsführung übernehmen")

    first_response = _build_raw_response(json.dumps(invalid_payload, ensure_ascii=False))
    second_response = _build_raw_response(json.dumps(payload, ensure_ascii=False))
    chain = _FakeChain([first_response, second_response])
    prompt = _FakePrompt(chain)

    monkeypatch.setattr(eval_chain, "get_eval_prompt", lambda: prompt)
    monkeypatch.setattr(eval_chain, "get_llm_config", lambda: SimpleNamespace(provider="ollama"))
    monkeypatch.setattr(eval_chain, "get_llm", lambda *_args, **_kwargs: object())

    result = asyncio.run(
        eval_chain.eval_history([HumanMessage(content="Wie geht es Ihnen?")], model="model-a")
    )

    assert len(chain.calls) == 2
    second_attempt_messages = chain.calls[1]["messages"]
    assert second_attempt_messages[-1].content == eval_chain.OLLAMA_JSON_RETRY_INSTRUCTION
    assert json.loads(result) == payload


def test_eval_history_ollama_retries_when_first_attempt_has_empty_output(monkeypatch):
    payload = _build_valid_payload()
    first_response = _build_raw_response("")
    second_response = _build_raw_response(json.dumps(payload, ensure_ascii=False))
    chain = _FakeChain([first_response, second_response])
    prompt = _FakePrompt(chain)

    monkeypatch.setattr(eval_chain, "get_eval_prompt", lambda: prompt)
    monkeypatch.setattr(eval_chain, "get_llm_config", lambda: SimpleNamespace(provider="ollama"))
    monkeypatch.setattr(eval_chain, "get_llm", lambda *_args, **_kwargs: object())

    result = asyncio.run(
        eval_chain.eval_history([HumanMessage(content="Wie geht es Ihnen?")], model="model-a")
    )

    assert len(chain.calls) == 2
    second_attempt_messages = chain.calls[1]["messages"]
    assert second_attempt_messages[-1].content == eval_chain.OLLAMA_JSON_RETRY_INSTRUCTION
    assert json.loads(result) == payload


def test_eval_history_ollama_recovers_from_tool_call_args_when_content_empty(monkeypatch):
    payload = _build_valid_payload()
    response = _build_raw_response(
        "",
        tool_calls=[{"args": payload}],
    )
    chain = _FakeChain([response])
    prompt = _FakePrompt(chain)

    monkeypatch.setattr(eval_chain, "get_eval_prompt", lambda: prompt)
    monkeypatch.setattr(eval_chain, "get_llm_config", lambda: SimpleNamespace(provider="ollama"))
    monkeypatch.setattr(eval_chain, "get_llm", lambda *_args, **_kwargs: object())

    result = asyncio.run(
        eval_chain.eval_history([HumanMessage(content="Wie geht es Ihnen?")], model="model-a")
    )

    assert json.loads(result) == payload


def test_eval_history_returns_friendly_error_on_terminal_parse_failure(monkeypatch):
    chain = _FakeChain(
        [
            _build_raw_response("kein json"),
            _build_raw_response("immer noch kein json"),
        ]
    )
    prompt = _FakePrompt(chain)

    monkeypatch.setattr(eval_chain, "get_eval_prompt", lambda: prompt)
    monkeypatch.setattr(eval_chain, "get_llm_config", lambda: SimpleNamespace(provider="ollama"))
    monkeypatch.setattr(eval_chain, "get_llm", lambda *_args, **_kwargs: object())

    result = asyncio.run(
        eval_chain.eval_history([HumanMessage(content="Wie geht es Ihnen?")], model="model-a")
    )

    assert result.startswith("Entschuldigung, es ist ein Fehler aufgetreten:")
    assert len(chain.calls) == 2


def test_log_response_diagnostics_respects_raw_logging_gate(monkeypatch, caplog):
    response = _build_raw_response("SENSITIVE_RAW_CONTENT")

    monkeypatch.delenv(eval_chain.SYMPTEX_EVAL_LOG_RAW_ENV_VAR, raising=False)
    caplog.set_level(logging.WARNING, logger=eval_chain.logger.name)
    eval_chain._log_response_diagnostics(
        stage="test",
        raw_response=response,
        level=logging.WARNING,
    )
    assert caplog.records == []

    caplog.clear()
    monkeypatch.setenv(eval_chain.SYMPTEX_EVAL_LOG_RAW_ENV_VAR, "true")
    eval_chain._log_response_diagnostics(
        stage="test",
        raw_response=response,
        level=logging.WARNING,
    )
    assert any("SENSITIVE_RAW_CONTENT" in record.getMessage() for record in caplog.records)
