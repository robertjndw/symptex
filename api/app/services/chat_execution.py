import json
import logging
import os
from typing import AsyncGenerator

from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from app.db.db_utils import has_anamdocs
from app.db.models import Case, ChatMessage, ChatSession, PatientFile
from app.db.symptex_models import SymptexConfig
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

async def execute_chat(
    db: Session,
    *,
    symptex_db: Session | None = None,
    model: str | None = None,
    message: str,
    condition: str | None = None,
    talkativeness: str | None = None,
    case_id: int,
    session_id: str,
    use_case_config: bool = False,
) -> StreamingResponse | PlainTextResponse:
    message_error = _validate_chat_message(message)
    if message_error is not None:
        return message_error

    medical_case, case_error = _get_case_or_404(db, case_id)
    if case_error is not None:
        return case_error

    chat_parameters, parameters_error = _resolve_effective_chat_parameters(
        symptex_db=symptex_db,
        medical_case=medical_case,
        use_case_config=use_case_config,
        model=model,
        condition=condition,
        talkativeness=talkativeness,
    )
    if parameters_error is not None:
        return parameters_error
    effective_model, effective_condition, effective_talkativeness = chat_parameters

    logger.info(
        "Resolved chat LLM parameters: model=%s condition=%s talkativeness=%s use_case_config=%s case_id=%s session_id=%s",
        effective_model,
        effective_condition,
        effective_talkativeness,
        use_case_config,
        case_id,
        session_id,
    )

    patient_file, patient_error = _get_patient_file_or_404(db, medical_case)
    if patient_error is not None:
        return patient_error
    patient_details = format_patient_details(patient_file)

    session, session_error = _get_or_create_chat_session(
        db,
        session_id=session_id,
        patient_file=patient_file,
        medical_case=medical_case,
    )
    if session_error is not None:
        return session_error

    previous_messages = _load_previous_messages(db, session)
    _save_user_message(db, session=session, message=message)

    docs_cache = _build_docs_cache(patient_file)
    return _build_chat_streaming_response(
        db=db,
        message=message,
        effective_model=effective_model,
        effective_condition=effective_condition,
        effective_talkativeness=effective_talkativeness,
        patient_file=patient_file,
        patient_details=patient_details,
        session=session,
        previous_messages=previous_messages,
        docs_cache=docs_cache,
    )

async def execute_eval(*, model: str, messages: list) -> PlainTextResponse:
    try:
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["output"]))
            elif msg["role"] in {"patient", "assistant"}:
                lc_messages.append(AIMessage(content=msg["output"]))

        evaluation_text = await eval_history(lc_messages, model=model)
        return PlainTextResponse(evaluation_text, status_code=200)
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


def _get_symptex_config(symptex_db: Session, case_id: int) -> SymptexConfig | None:
    return (
        symptex_db.query(SymptexConfig)
        .filter(SymptexConfig.case_id == case_id)
        .first()
    )


def _resolve_runtime_chat_parameters(*, symptex_db: Session, medical_case: Case) -> tuple[str, str, str]:
    effective_model = get_runtime_model()
    effective_condition = _read_default_condition()
    effective_talkativeness = _read_default_talkativeness()

    config = _get_symptex_config(symptex_db, medical_case.id)
    if not config:
        logger.info("No SymptexConfig for case_id=%s. Using defaults.", medical_case.id)
        return effective_model, effective_condition, effective_talkativeness

    candidate_model = (config.model or "").strip()
    if candidate_model:
        try:
            effective_model = validate_requested_model(candidate_model)
        except (InvalidModelError, LLMConfigurationError) as exc:
            logger.warning(
                "Invalid SymptexConfig.model=%r for case_id=%s. Using fallback runtime model %r. Reason: %s",
                candidate_model,
                medical_case.id,
                effective_model,
                exc,
            )
    else:
        logger.warning(
            "Empty SymptexConfig.model for case_id=%s. Using fallback runtime model %r.",
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

def _validate_chat_message(message: str) -> PlainTextResponse | None:
    if not message or not message.strip():
        logger.error("Empty message received")
        return PlainTextResponse("Message cannot be empty", status_code=400)
    return None


def _get_case_or_404(db: Session, case_id: int) -> tuple[Case | None, PlainTextResponse | None]:
    medical_case = db.query(Case).filter(Case.id == case_id).first()
    if not medical_case:
        return None, PlainTextResponse("Case not found", status_code=404)
    return medical_case, None


def _resolve_effective_chat_parameters(
    *,
    symptex_db: Session | None,
    medical_case: Case,
    use_case_config: bool,
    model: str | None,
    condition: str | None,
    talkativeness: str | None,
) -> tuple[tuple[str, str, str] | None, PlainTextResponse | None]:
    if use_case_config:
        if symptex_db is None:
            logger.error("Missing symptex_db dependency while use_case_config=True")
            return None, PlainTextResponse("Internal server error", status_code=500)
        try:
            effective_model, effective_condition, effective_talkativeness = _resolve_runtime_chat_parameters(
                symptex_db=symptex_db,
                medical_case=medical_case,
            )
        except LLMConfigurationError as exc:
            logger.error("LLM configuration error while resolving runtime chat parameters: %s", exc)
            return None, PlainTextResponse(str(exc), status_code=500)
    else:
        effective_model = (model or "").strip()
        effective_condition = (condition or "").strip()
        effective_talkativeness = (talkativeness or "").strip()

        if not effective_model:
            logger.error("Model cannot be empty")
            return None, PlainTextResponse("Model cannot be empty.", status_code=400)

    validation_error = _validate_condition_and_talkativeness(
        condition=effective_condition,
        talkativeness=effective_talkativeness,
    )
    if validation_error is not None:
        return None, validation_error

    return (effective_model, effective_condition, effective_talkativeness), None


def _get_patient_file_or_404(
    db: Session,
    medical_case: Case,
) -> tuple[PatientFile | None, PlainTextResponse | None]:
    patient_file = medical_case.patient_file
    if not patient_file:
        patient_file = db.query(PatientFile).filter(PatientFile.id == medical_case.patient_file_id).first()
    if not patient_file:
        return None, PlainTextResponse("Patient not found", status_code=404)
    return patient_file, None


def _get_or_create_chat_session(
    db: Session,
    *,
    session_id: str,
    patient_file: PatientFile,
    medical_case: Case,
) -> tuple[ChatSession | None, PlainTextResponse | None]:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session_patient_file_id = getattr(session, "patient_file_id", None)
        session_case_id = getattr(session, "case_id", None)
        if session_patient_file_id != patient_file.id or session_case_id != medical_case.id:
            logger.warning(
                "Session ownership mismatch for session_id=%s requested_case_id=%s requested_patient_file_id=%s "
                "actual_case_id=%s actual_patient_file_id=%s",
                session_id,
                medical_case.id,
                patient_file.id,
                session_case_id,
                session_patient_file_id,
            )
            return None, PlainTextResponse("Session does not belong to this case.", status_code=409)
        return session, None

    session = ChatSession(id=session_id, patient_file_id=patient_file.id, case_id=medical_case.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session, None


def _load_previous_messages(db: Session, session: ChatSession) -> list[HumanMessage | AIMessage]:
    previous_messages: list[HumanMessage | AIMessage] = []
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
    return previous_messages


def _save_user_message(db: Session, *, session: ChatSession, message: str) -> None:
    user_message = ChatMessage(session_id=session.id, role="user", content=message)
    db.add(user_message)
    db.commit()


def _build_docs_cache(patient_file: PatientFile) -> DocumentBundleCache:
    return DocumentBundleCache(
        client=AnamDocsClient.from_env(),
        patient_file_id=patient_file.id,
    )


async def _generate_and_store_chat_response(
    *,
    db: Session,
    message: str,
    effective_model: str,
    effective_condition: str,
    effective_talkativeness: str,
    patient_file: PatientFile,
    patient_details: str,
    session: ChatSession,
    previous_messages: list[HumanMessage | AIMessage],
    docs_cache: DocumentBundleCache,
) -> AsyncGenerator[str, None]:
    llm_response = ""
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


def _build_chat_streaming_response(
    *,
    db: Session,
    message: str,
    effective_model: str,
    effective_condition: str,
    effective_talkativeness: str,
    patient_file: PatientFile,
    patient_details: str,
    session: ChatSession,
    previous_messages: list[HumanMessage | AIMessage],
    docs_cache: DocumentBundleCache,
) -> StreamingResponse | PlainTextResponse:
    try:
        return StreamingResponse(
            _generate_and_store_chat_response(
                db=db,
                message=message,
                effective_model=effective_model,
                effective_condition=effective_condition,
                effective_talkativeness=effective_talkativeness,
                patient_file=patient_file,
                patient_details=patient_details,
                session=session,
                previous_messages=previous_messages,
                docs_cache=docs_cache,
            ),
            media_type="text/plain",
        )
    except Exception as exc:
        db.rollback()
        logger.error("Error in execute_chat: %s", str(exc))
        return PlainTextResponse("Internal server error", status_code=500)
    
