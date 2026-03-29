import logging
import os
from typing import Annotated

import langsmith as ls
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from chains.prompts.patient_prompts import get_prompt

# Load env variables for LangSmith to work
load_dotenv()

# Set up logging
logger = logging.getLogger('symptex_chain')
logger.setLevel(logging.DEBUG)

class CustomState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model: str
    condition: str
    talkativeness: str
        
# Set up env variables
CHATAI_API_URL = os.environ.get("CHATAI_API_URL")
CHATAI_API_KEY = os.environ.get("CHATAI_API_KEY")
if not CHATAI_API_URL or not CHATAI_API_KEY:
    logger.error("CHATAI environment variable not set, setting to default")
    raise ValueError("ERROR: Environment variables not set")

def get_llm(model: str) -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(
        openai_api_base=CHATAI_API_URL,
        openai_api_key=CHATAI_API_KEY,
        model=model,
        temperature=0.7,
        top_p=0.8,
        #max_tokens=1024,
        max_retries=2,
    )

@ls.traceable(
    run_type="llm",
    name="Patient LLM Call Decorator",
)
async def call_patient_model(state: CustomState):
    # Extract model, condition and talkativeness
    model = state.get("model")
    condition = state.get("condition")
    talkativeness = state.get("talkativeness")

    logger.debug(f"Calling patient model {model} with condition {condition} and talkativeness {talkativeness}")

    # Get appropriate prompt
    prompt = get_prompt(condition, talkativeness)
    chain = prompt | get_llm(model)

    try:
        # Invoke the chain
        response = await chain.ainvoke(state)
        logger.debug("Received response from patient model")

        return {"messages": response}
    except Exception as e:
        logger.error("Error calling patient model: %s", str(e))
        raise

# Define new graph
workflow = StateGraph(state_schema=CustomState)

# Define patient llm node
workflow.add_node("patient_model", call_patient_model)

# Set entrypoint as 'patient_model'
workflow.add_edge(START, "patient_model")
workflow.add_edge("patient_model", END)

# Add memory
memory = InMemorySaver()

# Compile into LangChain runnable
symptex_model = workflow.compile(checkpointer=memory)
