import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

logger = logging.getLogger('llm')
logger.setLevel(logging.DEBUG)

# Load env variables for LangSmith to work
load_dotenv()

# Set up env variables
CHATAI_API_URL = os.environ.get("CHATAI_API_URL")
CHATAI_API_KEY = os.environ.get("CHATAI_API_KEY")
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_SELECTED", "false").lower() == "true"

if not CHATAI_API_URL or not CHATAI_API_KEY:
    logger.error("CHATAI environment variable not set, setting to default")
    raise ValueError("ERROR: Environment variables not set")

#todo update with ollama models
def get_external_llm(model: str) -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        api_key=CHATAI_API_KEY,
        base_url=CHATAI_API_URL,
        model=model,
        temperature=0.7,
        top_p=0.8,
        #max_tokens=1024,
        max_retries=2,
    )

def get_local_llm() -> ChatOllama:
    return ChatOllama(
        model = "qwen2.5:7b-instruct",
        base_url="http://host.docker.internal:11434",
        temperature=0.7,
        top_p=0.8,
        max_retries=2,
    )

def get_llm(model: str):
    logger.info("Local llm selected: %s", LOCAL_LLM_ENABLED)
    if LOCAL_LLM_ENABLED:
        return get_local_llm()
    else:
        return get_external_llm(model=model)
