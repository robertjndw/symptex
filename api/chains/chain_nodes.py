import logging
import re
import traceback

from langchain_core.messages import ToolMessage
from langgraph.graph import add_messages

import chains.prompts.orchestrator_prompts as orchestrator_prompts
import chains.prompts.patient_prompts as patient_prompts
from chains.custom_state import CustomState
from chains.llm import get_llm
from chains.prompts import summarizer_prompts

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def make_load_docs_node(load_patient_docs_tool):
    async def load_docs_node(state: CustomState) -> CustomState:
        tool_calls = state.get("tool_calls", []) or []
        if not tool_calls:
            return state  # nothing to do

        tool_messages: list[ToolMessage] = []
        tool_output_text = None

        for tc in tool_calls:
            if tc.get("name") != "load_patient_docs":
                logger.warning("Unknown tool called: %s", tc.get("name"))
                continue


            try:
                tool_output_text = await load_patient_docs_tool.ainvoke({})
            except Exception as e:
                logger.exception("Error running load_patient_docs tool")
                tool_output_text = f"[Error loading documents: {e}]"

            tool_messages.append(
                ToolMessage(
                    name="load_patient_docs",
                    content=tool_output_text,
                    tool_call_id=tc.get("id"),
                )
            )

        state["tool_calls"] = []

        if not tool_output_text:
            logger.info("No docs found")
            state["attach_docs"] = False
            return state

        state["attach_docs"] = True
        
        # add tool messages to history for next LLM call
        state["messages"] = add_messages(state["messages"], tool_messages)

        return state
    return load_docs_node

def make_orchestrator_node(tool_list: list):
    async def orchestrator_node(state: CustomState) -> CustomState:
        model = state["model"]
        prompt = orchestrator_prompts.get_prompt()
        logger.info("Starting orchestrator node execution")
        llm_with_tools = get_llm(model).bind_tools(tool_list)
        logger.info("Tools bound to orchestrator")
        chain = prompt | llm_with_tools

        try:
            response = await chain.ainvoke(state)
            logger.debug("Initial LLM call succeeded")
            tool_calls = getattr(response, "tool_calls", []) or []
            state["tool_calls"] = tool_calls

            if tool_calls:
                logger.debug("Orchestrator requested tool calls: %s", tool_calls)
                state["hard_error"] = False
                return state

            content = getattr(response, "content", "") or ""
            stripped = strip_think_tags(content).strip()

            if stripped == "NO_TOOL":
                logger.debug("Orchestrator returned NO_TOOL (no-op).")
                state["hard_error"] = False
                return state

            logger.debug("Orchestrator output something other than a summary: %s", stripped)
            state["hard_error"] = False

            return state

        except Exception as e:
            logger.error("Error in orchestrator_node: %s", e)
            logger.error("Traceback:\n%s", traceback.format_exc())
            fallback = {
                "role": "ai",
                "content": f"Entschuldigung, ein Fehler ist aufgetreten: {str(e)}",
            }

            state["messages"] = add_messages(state.get("messages", []), [fallback])
            state["tool_calls"] = []
            state["attach_docs"] = False
            state["hard_error"] = True

            return state
    return orchestrator_node

#todo check if summary works with more than one file
#todo make the summary stay in the chat history so that the patient keeps the context
async def summary_node(state: CustomState) -> CustomState:
    model = state["model"]
    prompt = summarizer_prompts.get_prompt()
    logger.info("Starting summarizer node execution")
    logger.info("Tools bound to orchestrator")
    llm = get_llm(model)
    chain = prompt | llm
    response = await chain.ainvoke(state)
    stripped = strip_think_tags(response.content).strip()
    state["messages"] = remove_tool_messages(state["messages"])
    state["docs_summary"] = stripped
    state["attach_docs"] = True
    logger.debug("Summarizer state: %s",state)
    return state

async def patient_model_final(state: CustomState) -> dict:
    model = state["model"]
    condition = state["condition"]
    talkativeness = state["talkativeness"]
    patient_details = state["patient_details"]
    docs_available = state.get("docs_available", False)
    docs_summary = state.get("docs_summary", "")
    llm = get_llm(model)
    logger.info("Calling final patient model %s", model)

    prompt = patient_prompts.get_prompt(
        patient_condition=condition,
        talkativeness=talkativeness, 
        patient_details=patient_details, 
        docs_available=docs_available,
        docs_summary=docs_summary)

    chain = prompt | llm
    response = await chain.ainvoke(state)

    return {
        "messages": [response]
    }


def tool_branching_node(state: CustomState) -> str:
    """Routing function used by add_conditional_edges."""
    if state.get("hard_error", False):
        return "abort"
    tool_calls = state.get("tool_calls", []) or []
    return "has_tool_calls" if tool_calls else "no_tool_calls"

def docs_branching_node(state: CustomState) -> str:
    """Routing function used by add_conditional_edges."""
    logger.info("Starting load docs branching")
    if state.get("hard_error", False):
        logger.info("Hard error aborting")
        return "abort"
    if state["attach_docs"]:
        logger.info("Branching to summary node")
        return "summary"
    logger.info("Branching to final node")
    return "end"

def remove_tool_messages(messages):
    return [m for m in messages if not isinstance(m, ToolMessage)]
