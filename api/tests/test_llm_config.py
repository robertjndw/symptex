import pytest

from chains import llm


LLM_ENV_KEYS = [
    "LLM_PROVIDER",
    "LLM_CHATAI_BASE_URL",
    "LLM_CHATAI_API_KEY",
    "LLM_CHATAI_MODELS",
    "LLM_OLLAMA_BASE_URL",
    "LLM_OLLAMA_MODELS",
    "LLM_TEMPERATURE",
    "LLM_TOP_P",
    "LLM_MAX_RETRIES",
]


@pytest.fixture(autouse=True)
def reset_llm_env(monkeypatch):
    for key in LLM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    llm.clear_llm_config_cache()
    yield
    llm.clear_llm_config_cache()


def test_ollama_config_parses_and_deduplicates_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_OLLAMA_MODELS", " model-a,model-b,model-a ,, ")

    config = llm.get_llm_config()

    assert config.provider == "ollama"
    assert config.models == ("model-a", "model-b")
    assert config.default_model == "model-a"


def test_invalid_provider_raises_configuration_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(llm.LLMConfigurationError):
        llm.get_llm_config()


def test_missing_required_chatai_values_raise(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "chatai")
    monkeypatch.setenv("LLM_CHATAI_BASE_URL", "https://chat-ai.academiccloud.de/v1")
    monkeypatch.setenv("LLM_CHATAI_MODELS", "qwen3-235b-a22b")

    with pytest.raises(llm.LLMConfigurationError):
        llm.get_llm_config()


def test_chatai_factory_uses_requested_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "chatai")
    monkeypatch.setenv("LLM_CHATAI_BASE_URL", "https://example.org/v1")
    monkeypatch.setenv("LLM_CHATAI_API_KEY", "dummy")
    monkeypatch.setenv("LLM_CHATAI_MODELS", "m1,m2")

    chat_model = llm.get_llm("m2")

    assert chat_model.model_name == "m2"
    assert chat_model.openai_api_base == "https://example.org/v1"


def test_ollama_factory_uses_requested_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_OLLAMA_MODELS", "foo,bar")

    chat_model = llm.get_llm("bar")

    assert chat_model.model == "bar"
    assert chat_model.base_url == "http://localhost:11434"


def test_invalid_requested_model_raises_model_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_OLLAMA_MODELS", "foo,bar")

    with pytest.raises(llm.InvalidModelError) as exc:
        llm.validate_requested_model("baz")

    assert "Available models: foo, bar." in str(exc.value)
