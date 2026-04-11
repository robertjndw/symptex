import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.db.symptex_db import SymptexBase


class ChatSession(SymptexBase):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    patient_file_id = Column(Integer)
    case_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(SymptexBase):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey('chat_sessions.id'))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    session = relationship("ChatSession", back_populates="messages")


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
