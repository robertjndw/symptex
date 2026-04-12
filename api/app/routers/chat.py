import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.db.symptex_db import get_symptex_db
from app.db.symptex_models import ChatMessage, ChatSession
from app.services.chat_execution import (
    execute_chat,
    execute_eval,
    execute_eval_for_session,
    get_allowed_chat_parameters,
    load_chat_history_for_session,
)
from chains.llm import LLMConfigurationError, get_runtime_model

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    case_id: int
    session_id: str


class RateRequest(BaseModel):
    session_id: str | None = None
    case_id: int | None = None
    messages: list | None = None


class ChatHistoryMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    session_id: str
    case_id: int
    messages: list[ChatHistoryMessageResponse]


@router.post("/chat")
async def chat_with_llm(
    request: ChatRequest,
    db: Session = Depends(get_db),
    symptex_db: Session = Depends(get_symptex_db),
):
    logger.debug("Received runtime chat request: %s", request)
    return await execute_chat(
        db,
        symptex_db=symptex_db,
        message=request.message,
        case_id=request.case_id,
        session_id=request.session_id,
        use_case_config=True,
    )


@router.get("/chat/options")
async def get_chat_options():
    options, error = get_allowed_chat_parameters()
    if error is not None:
        return error
    return options


@router.post("/eval")
async def eval_chat(
    request: RateRequest,
    symptex_db: Session = Depends(get_symptex_db),
):
    try:
        model = get_runtime_model()
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error while resolving runtime model: %s", exc)
        return PlainTextResponse(str(exc), status_code=500)

    if request.session_id and request.case_id is not None:
        return await execute_eval_for_session(
            model=model,
            symptex_db=symptex_db,
            session_id=request.session_id,
            case_id=request.case_id,
        )

    if request.messages is not None:
        logger.warning("Deprecated /eval request payload with raw messages received.")
        return await execute_eval(model=model, messages=request.messages)

    return PlainTextResponse("Provide either session_id and case_id, or messages.", status_code=400)


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str = Query(...),
    case_id: int = Query(...),
    symptex_db: Session = Depends(get_symptex_db),
):
    messages, error = load_chat_history_for_session(
        symptex_db=symptex_db,
        session_id=session_id,
        case_id=case_id,
    )
    if error is not None:
        return error

    return ChatHistoryResponse(
        session_id=session_id,
        case_id=case_id,
        messages=[
            ChatHistoryMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
            )
            for msg in messages
        ],
    )


@router.post("/reset/{session_id}")
async def reset_memory(session_id: str, symptex_db: Session = Depends(get_symptex_db)):
    try:
        symptex_db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        symptex_db.query(ChatSession).filter(ChatSession.id == session_id).delete()
        symptex_db.commit()
        return PlainTextResponse(f"Chat data deleted for session {session_id}", status_code=200)
    except Exception as exc:
        logger.error("Error deleting session %s: %s", session_id, str(exc))
        symptex_db.rollback()
        return PlainTextResponse("Error deleting session", status_code=500)
