# API entry point
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import models
from app.db.db import engine
from app.db import symptex_models
from app.db.symptex_db import symptex_engine
from app.utils.env import read_bool_env

def is_dev_mode_enabled() -> bool:
    return read_bool_env("SYMPTEX_DEV_MODE", default=False)

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


def get_cors_allow_origins() -> list[str]:
    configured = os.getenv("SYMPTEX_CORS_ALLOW_ORIGINS", os.getenv("CORS_ALLOW_ORIGINS", "*")).strip()
    if configured == "":
        return ["*"]
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or ["*"]


def configure_cors(app: FastAPI) -> None:
    origins = get_cors_allow_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logging.getLogger(__name__).info("Configured CORS allow_origins=%s allow_credentials=%s", origins, False)


load_dotenv()
configure_logging()

from app.routers import chat, config  # noqa: E402

app = FastAPI(
    title="Symptex LangChain Server",
    version="1.0",
    description="API server for Symptex, a LangChain-based chat application for patient simulation",
)
configure_cors(app)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Include chat router
app.include_router(chat.router, prefix="/api/v1")
app.include_router(config.router, prefix="/api/v1")
if is_dev_mode_enabled():
    from app.routers import dev_chat  # noqa: E402

    app.include_router(dev_chat.router, prefix="/api/v1")
    logging.getLogger(__name__).info("Development mode enabled: /dev/chat and /dev/eval endpoints are active.")

# Init database schema
models.Base.metadata.create_all(bind=engine)
symptex_models.SymptexBase.metadata.create_all(bind=symptex_engine)
