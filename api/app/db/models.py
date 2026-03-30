from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from app.db.db import Base
import datetime

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    patient_file_id = Column(Integer, ForeignKey('patient_files.id'))
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey('chat_sessions.id'))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    session = relationship("ChatSession", back_populates="messages")

class PatientFile(Base):
    __tablename__ = "patient_files"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    birth_date = Column(Date)
    height = Column(Integer)
    weight = Column(Float)
    gender_identity = Column(String)
    gender_medical = Column(String)
    ethnic_origin = Column(String)
    anamneses = relationship("Anamnesis", back_populates="patient_file")
    cases = relationship("Case", back_populates="patient_file")
    

class Anamnesis(Base):
    __tablename__ = "anamneses"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String)
    answer = Column(String)
    patient_file_id = Column(Integer, ForeignKey("patient_files.id"))
    patient_file = relationship("PatientFile", back_populates="anamneses")
    anam_docs = relationship("AnamDoc", back_populates="anamnesis")

class AnamDoc(Base):
    __tablename__ = "anam_docs"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    storage_key = Column(String, nullable=False, unique=True, index=True)

    anamnesis_id = Column(
        Integer,
        ForeignKey("anamneses.id", onupdate="CASCADE", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    anamnesis = relationship("Anamnesis", back_populates="anam_docs")


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )
    deleted_at = Column(DateTime, nullable=True, index=True)

    title = Column(String, nullable=False)
    treatment_reason = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=True)
    due_date = Column(DateTime, nullable=True)
    marked = Column(Boolean, default=False, nullable=False)
    time_budget = Column(Float, nullable=False)
    money_budget = Column(Float, nullable=False)
    diagnosis = Column(String, nullable=False)
    treatment = Column(Text, nullable=True)
    is_draft = Column(Boolean, nullable=False)

    lecture_id = Column(Integer, nullable=False)
    patient_file_id = Column(Integer, ForeignKey("patient_files.id"), nullable=False)
    symptex_config_id = Column(Integer, nullable=True)

    patient_file = relationship("PatientFile", back_populates="cases")
    symptex_config = relationship(
        "SymptexConfig",
        back_populates="case",
        uselist=False,
        foreign_keys="SymptexConfig.case_id",
    )


class SymptexConfig(Base):
    __tablename__ = "symptex_configs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )
    deleted_at = Column(DateTime, nullable=True, index=True)

    enabled = Column(Boolean, default=True, nullable=False)
    llm_model = Column(String, nullable=False)
    condition = Column(String, nullable=False)
    talkativeness = Column(String, nullable=False)
    case_id = Column(
        Integer,
        ForeignKey("cases.id", onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    case = relationship(
        "Case",
        back_populates="symptex_config",
        foreign_keys=[case_id],
    )
