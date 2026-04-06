import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DEFAULT_ILVI_DATABASE_URL = "postgresql://ilvi:ilvi@postgres:5432/ilvi"
ILVI_DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_ILVI_DATABASE_URL).strip() or DEFAULT_ILVI_DATABASE_URL

SYMPTEX_DATABASE_URL = os.getenv("SYMPTEX_DATABASE_URL", ILVI_DATABASE_URL).strip() or ILVI_DATABASE_URL

symptex_engine = create_engine(SYMPTEX_DATABASE_URL)
SymptexSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=symptex_engine)
SymptexBase = declarative_base()


def get_symptex_db():
    db = SymptexSessionLocal()
    try:
        yield db
    finally:
        db.close()
