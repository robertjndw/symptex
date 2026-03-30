import logging

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.db.models import ChatMessage, ChatSession
from app.services.chat_execution import execute_chat, execute_eval
from chains.llm import LLMConfigurationError, get_runtime_model

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    condition: str
    talkativeness: str
    patient_file_id: int
    session_id: str


class RateRequest(BaseModel):
    messages: list


@router.post("/chat")
async def chat_with_llm(request: ChatRequest, db: Session = Depends(get_db)):
    logger.debug("Received runtime chat request: %s", request)
    try:
        model = get_runtime_model()
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error while resolving runtime model: %s", exc)
        return PlainTextResponse(str(exc), status_code=500)

    return await execute_chat(
        db,
        model=model,
        message=request.message,
        condition=request.condition,
        talkativeness=request.talkativeness,
        patient_file_id=request.patient_file_id,
        session_id=request.session_id,
    )


@router.post("/eval")
async def eval_chat(request: RateRequest):
    try:
        model = get_runtime_model()
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error while resolving runtime model: %s", exc)
        return PlainTextResponse(str(exc), status_code=500)

    return await execute_eval(model=model, messages=request.messages)


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
