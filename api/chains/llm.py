import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger("llm")
logger.setLevel(logging.DEBUG)

load_dotenv()

SUPPORTED_LLM_PROVIDERS = {"chatai", "ollama"}


class LLMConfigError(ValueError):
    pass


class LLMConfigurationError(LLMConfigError):
    pass


class InvalidModelError(LLMConfigError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    models: tuple[str, ...]
    default_model: str
    temperature: float
    top_p: float
    max_retries: int
    chatai_base_url: str | None = None
    chatai_api_key: str | None = None
    ollama_base_url: str | None = None

    def to_available_models_payload(self) -> dict[str, Any]:
        models_payload = [
            {"id": model_id, "label": model_id, "default": model_id == self.default_model}
            for model_id in self.models
        ]
        return {
            "provider": self.provider,
            "default_model": self.default_model,
            "models": models_payload,
        }


def _read_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    raise LLMConfigurationError(f"Required environment variable is missing: {name}")


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r. Falling back to %.2f.", name, raw_value, default)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
        if parsed < 0:
            logger.warning("Invalid %s=%r (must be >= 0). Falling back to %d.", name, raw_value, default)
            return default
        return parsed
    except ValueError:
        logger.warning("Invalid %s=%r. Falling back to %d.", name, raw_value, default)
        return default


def _parse_models(raw_models: str, env_name: str) -> tuple[str, ...]:
    parsed_models = [item.strip() for item in raw_models.split(",") if item.strip()]
    deduped_models = tuple(dict.fromkeys(parsed_models))
    if not deduped_models:
        raise LLMConfigurationError(f"Environment variable {env_name} must contain at least one model.")
    return deduped_models


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise LLMConfigurationError(f"Invalid LLM_PROVIDER={provider!r}. Expected one of: {supported}.")

    temperature = _read_float_env("LLM_TEMPERATURE", default=0.7)
    top_p = _read_float_env("LLM_TOP_P", default=0.8)
    max_retries = _read_int_env("LLM_MAX_RETRIES", default=2)

    if provider == "chatai":
        models = _parse_models(_read_required_env("LLM_CHATAI_MODELS"), "LLM_CHATAI_MODELS")
        return LLMConfig(
            provider=provider,
            models=models,
            default_model=models[0],
            temperature=temperature,
            top_p=top_p,
            max_retries=max_retries,
            chatai_base_url=_read_required_env("LLM_CHATAI_BASE_URL"),
            chatai_api_key=_read_required_env("LLM_CHATAI_API_KEY"),
        )

    models = _parse_models(_read_required_env("LLM_OLLAMA_MODELS"), "LLM_OLLAMA_MODELS")
    return LLMConfig(
        provider=provider,
        models=models,
        default_model=models[0],
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries,
        ollama_base_url=_read_required_env("LLM_OLLAMA_BASE_URL"),
    )


def clear_llm_config_cache() -> None:
    get_llm_config.cache_clear()


def get_available_models() -> dict[str, Any]:
    return get_llm_config().to_available_models_payload()


def validate_requested_model(model: str) -> str:
    normalized_model = (model or "").strip()
    if not normalized_model:
        raise InvalidModelError("Model cannot be empty.")

    config = get_llm_config()
    if normalized_model not in config.models:
        models = ", ".join(config.models)
        raise InvalidModelError(
            f"Invalid model '{normalized_model}' for provider '{config.provider}'. "
            f"Available models: {models}."
        )
    return normalized_model


def get_llm(model: str) -> ChatOpenAI | ChatOllama:
    validated_model = validate_requested_model(model)
    config = get_llm_config()

    if config.provider == "chatai":
        return ChatOpenAI(
            api_key=config.chatai_api_key,
            base_url=config.chatai_base_url,
            model=validated_model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_retries=config.max_retries,
        )

    return ChatOllama(
        model=validated_model,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        top_p=config.top_p,
        max_retries=config.max_retries,
    )
