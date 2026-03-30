import logging
import os


def read_bool_env(name: str, default: bool = False, *, logger: logging.Logger | None = None) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False

    active_logger = logger or logging.getLogger(__name__)
    active_logger.warning("Invalid boolean value for %s=%r. Falling back to %s.", name, raw_value, default)
    return default
