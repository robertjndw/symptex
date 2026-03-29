# API entry point
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI

from app.db import models
from app.db.db import engine


def configure_logging() -> None:
    requested_level = os.getenv("LOG_LEVEL", "INFO").upper()
    resolved_level = getattr(logging, requested_level, None)
    invalid_level = not isinstance(resolved_level, int)
    log_level = logging.INFO if invalid_level else resolved_level

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(funcName)s() - %(message)s",
    )
    # Ensure all module loggers inherit the same runtime level.
    logging.getLogger().setLevel(log_level)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).setLevel(log_level)

    logger = logging.getLogger(__name__)
    if invalid_level:
        logger.warning("Invalid LOG_LEVEL=%r. Falling back to INFO.", requested_level)
    logger.info("Configured logging level: %s", logging.getLevelName(log_level))


load_dotenv()
configure_logging()

from app.routers import chat  # noqa: E402

app = FastAPI(
    title="Symptex LangChain Server",
    version="1.0",
    description="API server for Symptex, a LangChain-based chat application for patient simulation",
)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Include chat router
app.include_router(chat.router, prefix="/api/v1")

# Init database schema
models.Base.metadata.create_all(bind=engine)
