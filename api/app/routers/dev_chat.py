import logging
import os

from fastapi import APIRouter, Depends, Header
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.db import get_db
from app.services.chat_execution import execute_chat, execute_eval, validate_model_selection

logger = logging.getLogger(__name__)
router = APIRouter()

DEV_FRONTEND_KEY_HEADER = "X-Dev-Frontend-Key"


class DevChatRequest(BaseModel):
    message: str
    model: str
    condition: str
    talkativeness: str
    patient_file_id: int
    session_id: str


class DevRateRequest(BaseModel):
    model: str
    messages: list


def _validate_dev_frontend_key(provided_key: str | None) -> PlainTextResponse | None:
    expected_key = os.getenv("DEV_FRONTEND_KEY", "").strip()
    if not expected_key:
        logger.error("Missing required DEV_FRONTEND_KEY environment variable.")
        return PlainTextResponse("Server misconfiguration: DEV_FRONTEND_KEY is required.", status_code=500)
    if not provided_key:
        return PlainTextResponse(f"Missing required header: {DEV_FRONTEND_KEY_HEADER}", status_code=401)
    if provided_key != expected_key:
        return PlainTextResponse("Invalid development frontend key.", status_code=403)
    return None


@router.post("/dev/chat")
async def dev_chat_with_llm(
    request: DevChatRequest,
    db: Session = Depends(get_db),
    dev_frontend_key: str | None = Header(default=None, alias=DEV_FRONTEND_KEY_HEADER),
):
    auth_error = _validate_dev_frontend_key(dev_frontend_key)
    if auth_error is not None:
        return auth_error

    model, model_error = validate_model_selection(request.model)
    if model_error is not None:
        return model_error

    return await execute_chat(
        db,
        model=model,
        message=request.message,
        condition=request.condition,
        talkativeness=request.talkativeness,
        patient_file_id=request.patient_file_id,
        session_id=request.session_id,
    )


@router.post("/dev/eval")
async def dev_eval_chat(
    request: DevRateRequest,
    dev_frontend_key: str | None = Header(default=None, alias=DEV_FRONTEND_KEY_HEADER),
):
    auth_error = _validate_dev_frontend_key(dev_frontend_key)
    if auth_error is not None:
        return auth_error

    model, model_error = validate_model_selection(request.model)
    if model_error is not None:
        return model_error

    return await execute_eval(model=model, messages=request.messages)
