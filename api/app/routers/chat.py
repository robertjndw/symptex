import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.db.db_utils import has_anamdocs
from app.db.models import ChatMessage, ChatSession, PatientFile
from app.services.anamdocs_client import AnamDocsClient
from chains.chat_chain import build_symptex_model
from chains.document_bundle_cache import DocumentBundleCache
from chains.eval_chain import eval_history
from chains.formatting import format_patient_details
from chains.llm import (
    InvalidModelError,
    LLMConfigurationError,
    get_available_models,
    validate_requested_model,
)

logger = logging.getLogger(__name__)
router = APIRouter()

TARGET_NODE = "patient_model_final"
ALLOWED_CONDITIONS = {"default", "alzheimer", "schwerhoerig", "verdraengung"}
ALLOWED_TALKATIVENESS = {"kurz angebunden", "ausgewogen", "ausschweifend"}


class ChatRequest(BaseModel):
    message: str
    model: str
    condition: str
    talkativeness: str
    patient_file_id: int
    session_id: str


class RateRequest(BaseModel):
    model: str
    messages: list


def _validate_model(requested_model: str) -> tuple[str | None, PlainTextResponse | None]:
    try:
        return validate_requested_model(requested_model), None
    except InvalidModelError as exc:
        logger.error("Model validation error: %s", exc)
        return None, PlainTextResponse(str(exc), status_code=400)
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error while validating model: %s", exc)
        return None, PlainTextResponse(str(exc), status_code=500)


@router.get("/available-models")
async def available_models():
    try:
        return JSONResponse(get_available_models(), status_code=200)
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error while listing models: %s", exc)
        return PlainTextResponse(str(exc), status_code=500)


@router.post("/chat")
async def chat_with_llm(request: ChatRequest, db: Session = Depends(get_db)):
    logger.debug("Received chat request: %s", request)

    if not request.message or not request.message.strip():
        logger.error("Empty message received")
        return PlainTextResponse("Message cannot be empty", status_code=400)

    model, model_error = _validate_model(request.model)
    if model_error is not None:
        return model_error

    if request.condition not in ALLOWED_CONDITIONS:
        logger.error("Invalid condition: %s", request.condition)
        return PlainTextResponse(f"Invalid condition: {request.condition}", status_code=400)

    if request.talkativeness not in ALLOWED_TALKATIVENESS:
        logger.error("Invalid talkativeness: %s", request.talkativeness)
        return PlainTextResponse(f"Invalid talkativeness: {request.talkativeness}", status_code=400)

    patient_file = db.query(PatientFile).filter(PatientFile.id == request.patient_file_id).first()
    if not patient_file:
        return PlainTextResponse("Patient not found", status_code=404)
    patient_details = format_patient_details(patient_file)

    #todo implement proper session management
    session = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
    if not session:
        session = ChatSession(id=request.session_id, patient_file_id=request.patient_file_id)
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

    user_message = ChatMessage(session_id=session.id, role="user", content=request.message)
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
                messages = previous_messages + [HumanMessage(content=request.message)]
                # Existence flag only: docs may exist even if loading/summarization fails this turn.
                docs_available = has_anamdocs(db, patient_file.id)
                logger.info("Beginning text streaming")
                async for chunk in stream_response(
                    model=model,
                    condition=request.condition,
                    talkativeness=request.talkativeness,
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
        logger.error("Error in chat_with_llm endpoint: %s", str(exc))
        return PlainTextResponse("Internal server error", status_code=500)


@router.post("/reset/{session_id}")
async def reset_memory(session_id: str, db: Session = Depends(get_db)):
    try:
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        db.query(ChatSession).filter(ChatSession.id == session_id).delete()
        db.commit()
        return PlainTextResponse(f"Chat data deleted for session {session_id}", status_code=200)
    except Exception as exc:
        logger.error("Error deleting session %s: %s", session_id, str(exc))
        db.rollback()
        return PlainTextResponse("Error deleting session", status_code=500)


@router.post("/eval")
async def eval_chat(request: RateRequest):
    from langchain_core.messages import AIMessage, HumanMessage

    model, model_error = _validate_model(request.model)
    if model_error is not None:
        return model_error

    async def generate_eval():
        try:
            lc_messages = []
            for msg in request.messages:
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

    try:
        async for mode, chunk in symptex_model.astream(
            initial_state,
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                msg, metadata = chunk
                node_name = metadata.get("langgraph_node")
                if node_name == TARGET_NODE and msg.content and not isinstance(msg, HumanMessage):
                    yield msg.content
            elif mode == "values":
                state = chunk
                if attach_docs_flag is not None and state.get("attach_docs"):
                    attach_docs_flag["value"] = True
    except Exception as exc:
        logger.error("Error while streaming response: %s", str(exc))
        yield f"Entschuldigung, es ist ein Fehler aufgetreten: {str(exc)}"
