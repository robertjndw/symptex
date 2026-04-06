import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy.orm import Session

from app.db.symptex_db import get_symptex_db
from app.db.symptex_models import SymptexConfig

logger = logging.getLogger(__name__)
router = APIRouter()


class SymptexConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=False)

    case_id: StrictInt = Field(alias="caseId")
    model: StrictStr
    talkativeness: StrictStr
    condition: StrictStr


@router.get("/config")
async def get_config(
    case_id: int = Query(alias="caseId"),
    db: Session = Depends(get_symptex_db),
):
    try:
        config = db.query(SymptexConfig).filter(SymptexConfig.case_id == case_id).first()
        if config is None:
            return PlainTextResponse(f"No config found for case {case_id}", status_code=404)

        return {
            "caseId": config.case_id,
            "model": config.model,
            "talkativeness": config.talkativeness,
            "condition": config.condition,
        }
    except Exception as exc:
        logger.error("Failed to read SymptexConfig for case_id=%s: %s", case_id, exc)
        return PlainTextResponse("Error while reading config", status_code=500)


@router.post("/config")
async def upsert_config(
    request: SymptexConfigRequest,
    db: Session = Depends(get_symptex_db),
):
    try:
        existing = db.query(SymptexConfig).filter(SymptexConfig.case_id == request.case_id).first()
        if existing is not None:
            existing.model = request.model
            existing.talkativeness = request.talkativeness
            existing.condition = request.condition
            db.commit()
            logger.info("Updated SymptexConfig for case_id=%s", request.case_id)
            return {"caseId": request.case_id, "updated": True}

        config = SymptexConfig(
            case_id=request.case_id,
            model=request.model,
            talkativeness=request.talkativeness,
            condition=request.condition,
        )
        db.add(config)
        db.commit()
        logger.info("Created SymptexConfig for case_id=%s", request.case_id)
        return {"caseId": request.case_id, "updated": False}
    except Exception as exc:
        logger.error("Failed to upsert SymptexConfig for case_id=%s: %s", request.case_id, exc)
        db.rollback()
        return PlainTextResponse("Error while storing config", status_code=500)


@router.delete("/config/{case_id}")
async def delete_config(case_id: int, db: Session = Depends(get_symptex_db)):
    try:
        config = db.query(SymptexConfig).filter(SymptexConfig.case_id == case_id).first()
        if config is None:
            logger.info("No SymptexConfig to delete for case_id=%s", case_id)
            return PlainTextResponse(f"No config found for case {case_id}", status_code=200)

        db.delete(config)
        db.commit()
        logger.info("Deleted SymptexConfig for case_id=%s", case_id)
        return PlainTextResponse(f"Config deleted for case {case_id}", status_code=200)
    except Exception as exc:
        logger.error("Failed to delete SymptexConfig for case_id=%s: %s", case_id, exc)
        db.rollback()
        return PlainTextResponse("Error while deleting config", status_code=500)
