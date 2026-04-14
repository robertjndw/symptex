import json
import logging
import os
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

from app.utils.eval_json import (
    EVAL_CATEGORIES,
    build_eval_response_schema,
    extract_eval_payload,
    get_eval_categories_with_overall,
    normalize_eval_result,
)
from chains.llm import get_llm, get_llm_config

logger = logging.getLogger(__name__)

SYMPTEX_EVAL_LOG_RAW_ENV_VAR = "SYMPTEX_EVAL_LOG_RAW"
RAW_LOG_TRUE_VALUES = {"1", "true", "yes", "on"}
LOG_CONTENT_LIMIT = 4000
OLLAMA_EVAL_TEMPERATURE = 0.2
OLLAMA_MAX_ATTEMPTS = 2
OLLAMA_JSON_RETRY_INSTRUCTION = (
    "WICHTIG: Antworte jetzt ausschließlich mit einem gültigen JSON-Objekt "
    "mit den geforderten Schlüsseln und Feldern. Kein weiterer Text."
)


class EmptyEvalOutputError(ValueError):
    """Raised when Ollama returns no usable content or tool-call payload."""


EVAL_SYSTEM_PROMPT_TEMPLATE = """
            Ziel: Du bist ein medizinischer Prüfer und bewertest die klinische Gesprächsführung eines Doktors während der Anamneseerhebung anhand definierter klinischer Indikatoren (CRI-HT) auf Deutsch.
            Die Bewertung erfolgt auf einer Skala von 1 bis 5 für jede Kategorie.

            Bewertungskriterien:
{criteria}

            Bewertungsskala:
            1: Kriterium nicht erfüllt
            2: Kriterium eher nicht erfüllt
            3: Teilerfüllung
            4: Kriterium weitgehend erfüllt
            5: Vollständig erfüllt

            Anweisung:
            Analysiere den vorgelegten Arzt-Patienten-Dialog und vergib für jedes der 8 Kriterien eine Punktzahl (1-5).
            Begründe jede Bewertung mit konkreten Beispielen aus dem Dialog.
            Die Bewertung soll konstruktiv sein und Verbesserungspotenziale aufzeigen.

            Gib ausschließlich ein JSON-Objekt zurück und keinen zusätzlichen Text, kein Markdown und keine Code-Fences.
            Das JSON muss exakt diese obersten Schlüssel enthalten:
{json_keys}

            Jeder dieser Schlüssel hat als Wert ein JSON-Objekt mit genau drei Feldern:
            * score: Integer von 1 bis 5
            * message: String mit Begründung; bei den 8 Kriterien inkl. konkreten Beispielen aus dem Dialog, bei Gesamtbewertung inkl. Stärken
            * verbesserungsvorschlag: String mit konkretem Verbesserungsvorschlag für diese Kategorie
            """

def get_eval_prompt() -> ChatPromptTemplate:
    criteria = _to_prompt_bullets(EVAL_CATEGORIES)
    json_keys = _to_prompt_bullets(get_eval_categories_with_overall())
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                EVAL_SYSTEM_PROMPT_TEMPLATE.format(criteria=criteria, json_keys=json_keys)
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

async def eval_history(messages: list, model: str) -> str:
    mapped_messages: list[BaseMessage] = []
    provider = "unresolved"
    try:
        prompt = get_eval_prompt()
        mapped_messages = _role_map_for_eval(messages)
        logger.debug("Evaluating %d mapped messages for model=%s", len(mapped_messages), model)
        llm_config = get_llm_config()
        provider = llm_config.provider

        if provider == "ollama":
            payload = await _run_ollama_eval_with_retries(
                prompt=prompt,
                mapped_messages=mapped_messages,
                model=model,
            )
        else:
            payload = await _run_structured_eval(
                prompt=prompt,
                mapped_messages=mapped_messages,
                model=model,
            )

        return _normalize_and_serialize_eval_payload(payload)
    except Exception as error:
        logger.error(
            "Error in eval_history | provider=%s | model=%s | error=%s",
            provider,
            model,
            str(error),
        )
        return f"Entschuldigung, es ist ein Fehler aufgetreten: {str(error)}"

def _normalize_and_serialize_eval_payload(payload: dict) -> str:
    normalized = normalize_eval_result(payload)
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def _build_ollama_attempt_messages(mapped_messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    return [
        list(mapped_messages),
        list(mapped_messages) + [HumanMessage(content=OLLAMA_JSON_RETRY_INSTRUCTION)],
    ]

def _is_raw_eval_logging_enabled() -> bool:
    return os.getenv(SYMPTEX_EVAL_LOG_RAW_ENV_VAR, "").strip().lower() in RAW_LOG_TRUE_VALUES


def _extract_raw_text_for_log(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)
    return str(content or "")


def _truncate_for_log(text: str, limit: int = LOG_CONTENT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _log_response_diagnostics(
    *,
    stage: str,
    raw_response: object,
    level: int,
    attempt: int | None = None,
    parsing_error: object | None = None,
) -> None:
    if not _is_raw_eval_logging_enabled():
        return

    raw_content = getattr(raw_response, "content", raw_response)
    raw_text = _extract_raw_text_for_log(raw_content)
    raw_metadata = getattr(raw_response, "response_metadata", None)
    raw_additional_kwargs = getattr(raw_response, "additional_kwargs", None)
    raw_tool_calls = getattr(raw_response, "tool_calls", None)
    raw_invalid_tool_calls = getattr(raw_response, "invalid_tool_calls", None)

    tool_call_count = len(raw_tool_calls) if isinstance(raw_tool_calls, list) else 0
    invalid_tool_call_count = len(raw_invalid_tool_calls) if isinstance(raw_invalid_tool_calls, list) else 0

    logger.log(
        level,
        "Eval raw LLM output | stage=%s | attempt=%s | parsing_error=%r | content_len=%d | "
        "content=%s | metadata=%r | additional_kwargs=%r | tool_call_count=%d | invalid_tool_call_count=%d | "
        "tool_calls=%r | invalid_tool_calls=%r",
        stage,
        attempt if attempt is not None else "-",
        parsing_error,
        len(raw_text),
        _truncate_for_log(raw_text),
        raw_metadata,
        raw_additional_kwargs,
        tool_call_count,
        invalid_tool_call_count,
        raw_tool_calls,
        raw_invalid_tool_calls,
    )


def _extract_ollama_tool_call_args(raw_response: object) -> dict | str | None:
    tool_calls = getattr(raw_response, "tool_calls", None)
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            args = tool_call.get("args")
            if isinstance(args, dict):
                return dict(args)
            if isinstance(args, str) and args.strip():
                return args

    additional_kwargs = getattr(raw_response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        kw_tool_calls = additional_kwargs.get("tool_calls")
        if isinstance(kw_tool_calls, list):
            for item in kw_tool_calls:
                if not isinstance(item, dict):
                    continue
                function_payload = item.get("function")
                if not isinstance(function_payload, dict):
                    continue
                arguments = function_payload.get("arguments")
                if isinstance(arguments, str) and arguments.strip():
                    return arguments

    return None


def _extract_payload_from_tool_call_args(tool_call_args: dict | str) -> dict:
    if isinstance(tool_call_args, dict):
        return tool_call_args
    try:
        return extract_eval_payload(tool_call_args)
    except Exception as tool_call_error:
        raise ValueError(
            "Ollama eval payload recovery failed after parser and tool-call fallback."
        ) from tool_call_error


def _extract_payload_with_ollama_fallback(raw_response: object) -> dict:
    raw_content = getattr(raw_response, "content", raw_response)
    raw_text = _extract_raw_text_for_log(raw_content)
    tool_call_args = _extract_ollama_tool_call_args(raw_response)

    if not raw_text.strip():
        if tool_call_args is None:
            raise EmptyEvalOutputError(
                "Ollama eval returned empty output (no content and no tool-call payload)."
            )
        return _extract_payload_from_tool_call_args(tool_call_args)

    if raw_text.strip():
        try:
            return extract_eval_payload(raw_text)
        except Exception:
            # Fall back to additional sources below.
            pass

    try:
        return extract_eval_payload(raw_response)
    except Exception as primary_error:
        if tool_call_args is None:
            raise primary_error
        return _extract_payload_from_tool_call_args(tool_call_args)


def _log_structured_eval_parse_diagnostics(response: object) -> None:
    if not isinstance(response, dict):
        return
    if not {"raw", "parsed", "parsing_error"}.issubset(response.keys()):
        return

    parsed = response.get("parsed")
    parsing_error = response.get("parsing_error")
    if isinstance(parsed, dict) and parsing_error is None:
        return

    logger.warning(
        "Structured eval parser diagnostics | parsed_type=%s | parsing_error=%r",
        type(parsed).__name__,
        parsing_error,
    )
    _log_response_diagnostics(
        stage="structured-parse-diagnostics",
        raw_response=response.get("raw"),
        level=logging.WARNING,
        parsing_error=parsing_error,
    )


def _role_map_for_eval(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    mapped_messages: list[BaseMessage] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(msg, HumanMessage):
            mapped_messages.append(HumanMessage(content=f"Arzt: {content}"))
        elif isinstance(msg, AIMessage):
            mapped_messages.append(AIMessage(content=f"Patient: {content}"))
        else:
            mapped_messages.append(msg)
    return mapped_messages


def _build_eval_llm(model: str):
    llm = get_llm(model)
    schema = build_eval_response_schema()

    if isinstance(llm, ChatOpenAI):
        return llm.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)

    return llm.with_structured_output(schema, method="json_schema", include_raw=True)


def _to_prompt_bullets(items: Sequence[str]) -> str:
    return "\n".join(f"            * {item}" for item in items)

async def _run_ollama_eval_with_retries(
    *,
    prompt: ChatPromptTemplate,
    mapped_messages: list[BaseMessage],
    model: str,
) -> dict:
    llm = get_llm(model, temperature=OLLAMA_EVAL_TEMPERATURE)
    chain = prompt | llm
    attempts = _build_ollama_attempt_messages(mapped_messages)[:OLLAMA_MAX_ATTEMPTS]
    last_error: Exception | None = None

    for attempt, attempt_messages in enumerate(attempts, start=1):
        response = await chain.ainvoke({"messages": attempt_messages})
        try:
            payload = _extract_payload_with_ollama_fallback(response)
            # Validate payload shape during retries so malformed-but-parseable JSON can trigger retry.
            normalize_eval_result(payload)
            return payload
        except Exception as parse_exc:
            last_error = parse_exc
            stage = (
                "ollama-empty-output"
                if isinstance(parse_exc, EmptyEvalOutputError)
                else "ollama-parse-failure"
            )
            error_kind = "empty output" if stage == "ollama-empty-output" else "parse/validation"
            logger.warning(
                "Ollama eval %s failed | attempt=%d/%d | error=%s",
                error_kind,
                attempt,
                len(attempts),
                parse_exc,
            )
            _log_response_diagnostics(
                stage=stage,
                raw_response=response,
                level=logging.WARNING,
                attempt=attempt,
                parsing_error=parse_exc,
            )
            if attempt < len(attempts):
                logger.warning(
                    "Retrying Ollama eval with reinforcement prompt | next_attempt=%d",
                    attempt + 1,
                )

    raise last_error or ValueError("Ollama eval parsing failed without a specific error.")


async def _run_structured_eval(
    *,
    prompt: ChatPromptTemplate,
    mapped_messages: list[BaseMessage],
    model: str,
) -> dict:
    llm = _build_eval_llm(model)
    chain = prompt | llm
    response = await chain.ainvoke({"messages": mapped_messages})
    _log_structured_eval_parse_diagnostics(response)
    return extract_eval_payload(response)
