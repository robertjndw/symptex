import json
import logging
import os
from typing import AsyncGenerator

from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from app.db.db_utils import has_anamdocs
from app.db.models import Case, ChatMessage, ChatSession, PatientFile
from app.services.anamdocs_client import AnamDocsClient
from app.utils.stream_filters import StreamDelimitedBlockFilter
from chains.chat_chain import build_symptex_model
from chains.document_bundle_cache import DocumentBundleCache
from chains.eval_chain import eval_history
from chains.formatting import format_patient_details
from chains.llm import (
    InvalidModelError,
    LLMConfigurationError,
    get_llm_config,
    get_runtime_model,
    validate_requested_model,
)

logger = logging.getLogger(__name__)

TARGET_NODE = "patient_model_final"
ALLOWED_CONDITIONS = ("default", "alzheimer", "schwerhoerig", "verdraengung")
ALLOWED_CONDITIONS_SET = set(ALLOWED_CONDITIONS)
ALLOWED_TALKATIVENESS = ("kurz angebunden", "ausgewogen", "ausschweifend")
ALLOWED_TALKATIVENESS_SET = set(ALLOWED_TALKATIVENESS)
DEFAULT_CONDITION = "default"
DEFAULT_TALKATIVENESS = "ausgewogen"
DEFAULT_CONDITION_ENV = "SYMPTEX_DEFAULT_CONDITION"
DEFAULT_TALKATIVENESS_ENV = "SYMPTEX_DEFAULT_TALKATIVENESS"
MODEL_INTERNAL_BLOCK_START = "<think>"
MODEL_INTERNAL_BLOCK_END = "</think>"


def validate_model_selection(requested_model: str) -> tuple[str | None, PlainTextResponse | None]:
    try:
        return validate_requested_model(requested_model), None
    except InvalidModelError as exc:
        logger.error("Model validation error: %s", exc)
        return None, PlainTextResponse(str(exc), status_code=400)
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error while validating model: %s", exc)
        return None, PlainTextResponse(str(exc), status_code=500)


def get_allowed_chat_parameters() -> tuple[dict[str, list[str]] | None, PlainTextResponse | None]:
    try:
        models = list(get_llm_config().models)
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error while resolving allowed chat parameters: %s", exc)
        return None, PlainTextResponse(str(exc), status_code=500)

    return {
        "models": models,
        "conditions": list(ALLOWED_CONDITIONS),
        "talkativeness": list(ALLOWED_TALKATIVENESS),
    }, None


def _read_default_condition() -> str:
    configured = os.getenv(DEFAULT_CONDITION_ENV, "").strip()
    if not configured:
        return DEFAULT_CONDITION
    if configured in ALLOWED_CONDITIONS_SET:
        return configured
    logger.warning(
        "Invalid %s=%r. Falling back to %r.",
        DEFAULT_CONDITION_ENV,
        configured,
        DEFAULT_CONDITION,
    )
    return DEFAULT_CONDITION


def _read_default_talkativeness() -> str:
    configured = os.getenv(DEFAULT_TALKATIVENESS_ENV, "").strip()
    if not configured:
        return DEFAULT_TALKATIVENESS
    if configured in ALLOWED_TALKATIVENESS_SET:
        return configured
    logger.warning(
        "Invalid %s=%r. Falling back to %r.",
        DEFAULT_TALKATIVENESS_ENV,
        configured,
        DEFAULT_TALKATIVENESS,
    )
    return DEFAULT_TALKATIVENESS


def _validate_condition_and_talkativeness(
    condition: str,
    talkativeness: str,
) -> PlainTextResponse | None:
    if condition not in ALLOWED_CONDITIONS_SET:
        logger.error("Invalid condition: %s", condition)
        return PlainTextResponse(f"Invalid condition: {condition}", status_code=400)

    if talkativeness not in ALLOWED_TALKATIVENESS_SET:
        logger.error("Invalid talkativeness: %s", talkativeness)
        return PlainTextResponse(f"Invalid talkativeness: {talkativeness}", status_code=400)

    return None


def _resolve_runtime_chat_parameters(medical_case: Case) -> tuple[str, str, str]:
    effective_model = get_runtime_model()
    effective_condition = _read_default_condition()
    effective_talkativeness = _read_default_talkativeness()

    config = medical_case.symptex_config
    if not config:
        logger.info("No SymptexConfig for case_id=%s. Using defaults.", medical_case.id)
        return effective_model, effective_condition, effective_talkativeness

    candidate_model = (config.llm_model or "").strip()
    if candidate_model:
        try:
            effective_model = validate_requested_model(candidate_model)
        except (InvalidModelError, LLMConfigurationError) as exc:
            logger.warning(
                "Invalid SymptexConfig.llm_model=%r for case_id=%s. Using fallback runtime model %r. Reason: %s",
                candidate_model,
                medical_case.id,
                effective_model,
                exc,
            )
    else:
        logger.warning(
            "Empty SymptexConfig.llm_model for case_id=%s. Using fallback runtime model %r.",
            medical_case.id,
            effective_model,
        )

    candidate_condition = (config.condition or "").strip()
    if candidate_condition in ALLOWED_CONDITIONS_SET:
        effective_condition = candidate_condition
    else:
        logger.warning(
            "Invalid SymptexConfig.condition=%r for case_id=%s. Using fallback condition %r.",
            candidate_condition,
            medical_case.id,
            effective_condition,
        )

    candidate_talkativeness = (config.talkativeness or "").strip()
    if candidate_talkativeness in ALLOWED_TALKATIVENESS_SET:
        effective_talkativeness = candidate_talkativeness
    else:
        logger.warning(
            "Invalid SymptexConfig.talkativeness=%r for case_id=%s. Using fallback talkativeness %r.",
            candidate_talkativeness,
            medical_case.id,
            effective_talkativeness,
        )

    return effective_model, effective_condition, effective_talkativeness

#todo refactor
async def execute_chat(
    db: Session,
    *,
    model: str | None = None,
    message: str,
    condition: str | None = None,
    talkativeness: str | None = None,
    case_id: int,
    session_id: str,
    use_case_config: bool = False,
) -> StreamingResponse | PlainTextResponse:
    if not message or not message.strip():
        logger.error("Empty message received")
        return PlainTextResponse("Message cannot be empty", status_code=400)

    medical_case = db.query(Case).filter(Case.id == case_id).first()
    if not medical_case:
        return PlainTextResponse("Case not found", status_code=404)

    if use_case_config:
        try:
            effective_model, effective_condition, effective_talkativeness = _resolve_runtime_chat_parameters(
                medical_case
            )
        except LLMConfigurationError as exc:
            logger.error("LLM configuration error while resolving runtime chat parameters: %s", exc)
            return PlainTextResponse(str(exc), status_code=500)
    else:
        effective_model = (model or "").strip()
        effective_condition = (condition or "").strip()
        effective_talkativeness = (talkativeness or "").strip()

        if not effective_model:
            logger.error("Model cannot be empty")
            return PlainTextResponse("Model cannot be empty.", status_code=400)

    validation_error = _validate_condition_and_talkativeness(
        condition=effective_condition,
        talkativeness=effective_talkativeness,
    )
    if validation_error is not None:
        return validation_error

    patient_file = medical_case.patient_file
    if not patient_file:
        patient_file = db.query(PatientFile).filter(PatientFile.id == medical_case.patient_file_id).first()
    if not patient_file:
        return PlainTextResponse("Patient not found", status_code=404)
    patient_details = format_patient_details(patient_file)

    # TODO: implement proper session management
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        session = ChatSession(id=session_id, patient_file_id=patient_file.id)
        db.add(session)
        db.commit()
        db.refresh(session)

    previous_messages = []
    chat_history = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    for msg in chat_history:
        if msg.role == "user":
            previous_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "patient":
            previous_messages.append(AIMessage(content=msg.content))

    user_message = ChatMessage(session_id=session.id, role="user", content=message)
    db.add(user_message)
    db.commit()

    docs_cache = DocumentBundleCache(
        client=AnamDocsClient.from_env(),
        patient_file_id=patient_file.id,
    )

    try:
        llm_response = ""

        async def generate_and_store():
            nonlocal llm_response
            attach_docs_flag = {"value": False}
            try:
                messages = previous_messages + [HumanMessage(content=message)]
                # Existence flag only: docs may exist even if loading/summarization fails this turn.
                docs_available = has_anamdocs(db, patient_file.id)
                logger.info("Beginning text streaming")
                async for chunk in stream_response(
                    model=effective_model,
                    condition=effective_condition,
                    talkativeness=effective_talkativeness,
                    patient_details=patient_details,
                    previous_messages=messages,
                    docs_available=docs_available,
                    docs_cache=docs_cache,
                    attach_docs_flag=attach_docs_flag,
                ):
                    llm_response += chunk
                    yield chunk

                logger.info("Text streaming ended")
                if attach_docs_flag.get("value"):
                    docs = docs_cache.get_frontend_docs()
                    if docs:
                        # TODO(protocol): This app currently mixes plain text streaming with a
                        # trailing raw JSON control payload. Coalesced chunks can make frontend
                        # detection ambiguous. Migrate backend+frontend together to an explicit
                        # framed event protocol (e.g. EVENT:/SSE/NDJSON) with incremental parsing.
                        yield "\n" + json.dumps({"event": "attach_docs", "docs": docs})

                llm_message = ChatMessage(
                    session_id=session.id,
                    role="patient",
                    content=llm_response,
                )
                db.add(llm_message)
                db.commit()
            except Exception:
                db.rollback()
                raise

        return StreamingResponse(generate_and_store(), media_type="text/plain")
    except Exception as exc:
        db.rollback()
        logger.error("Error in execute_chat: %s", str(exc))
        return PlainTextResponse("Internal server error", status_code=500)


async def execute_eval(*, model: str, messages: list) -> StreamingResponse | PlainTextResponse:
    async def generate_eval():
        try:
            lc_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["output"]))
                elif msg["role"] in {"patient", "assistant"}:
                    lc_messages.append(AIMessage(content=msg["output"]))
            async for chunk in eval_history(lc_messages, model=model):
                yield chunk
        except Exception as exc:
            logger.error("Error generating evaluation: %s", str(exc))
            yield f"Entschuldigung, es ist ein Fehler aufgetreten: {str(exc)}"

    try:
        return StreamingResponse(generate_eval(), media_type="text/plain")
    except Exception as exc:
        logger.error("Error rating chat: %s", str(exc))
        return PlainTextResponse("Error rating chat", status_code=500)


async def stream_response(
    model: str,
    condition: str,
    talkativeness: str,
    patient_details: str,
    previous_messages: list,
    docs_available: bool,
    docs_cache: DocumentBundleCache,
    attach_docs_flag: dict | None = None,
) -> AsyncGenerator[str, None]:
    initial_state = {
        "messages": previous_messages,
        "model": model,
        "condition": condition,
        "talkativeness": talkativeness,
        "patient_details": patient_details,
        # Existence flag from DB/backend; not overwritten by in-turn load outcomes.
        "docs_available": docs_available,
    }
    symptex_model = build_symptex_model(docs_cache)
    internal_block_filter = StreamDelimitedBlockFilter(
        start_delimiter=MODEL_INTERNAL_BLOCK_START,
        end_delimiter=MODEL_INTERNAL_BLOCK_END,
    )

    try:
        async for mode, chunk in symptex_model.astream(
            initial_state,
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                msg, metadata = chunk
                node_name = metadata.get("langgraph_node")
                if node_name == TARGET_NODE and msg.content and not isinstance(msg, HumanMessage):
                    clean_chunk = internal_block_filter.consume(msg.content)
                    if clean_chunk:
                        yield clean_chunk
            elif mode == "values":
                state = chunk
                if attach_docs_flag is not None and state.get("attach_docs"):
                    attach_docs_flag["value"] = True
        trailing_chunk = internal_block_filter.flush()
        if trailing_chunk:
            yield trailing_chunk
    except Exception as exc:
        logger.error("Error while streaming response: %s", str(exc))
        yield f"Entschuldigung, es ist ein Fehler aufgetreten: {str(exc)}"
