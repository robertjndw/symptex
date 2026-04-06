import datetime

from sqlalchemy import Column, DateTime, Integer, String

from app.db.symptex_db import SymptexBase


class SymptexConfig(SymptexBase):
    __tablename__ = "symptex_config"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )
    deleted_at = Column(DateTime, nullable=True, index=True)

    case_id = Column(Integer, nullable=False, unique=True, index=True)
    model = Column(String, nullable=False)
    condition = Column(String, nullable=False)
    talkativeness = Column(String, nullable=False)
