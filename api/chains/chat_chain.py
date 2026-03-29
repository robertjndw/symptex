import logging

from langgraph.graph import START, StateGraph, END

from chains.chain_nodes import patient_model_final, tool_branching_node, make_orchestrator_node, make_load_docs_node, \
    summary_node, docs_branching_node
from chains.chain_tools import make_load_patient_files_tool
from chains.custom_state import CustomState
from chains.document_bundle_cache import DocumentBundleCache

logger = logging.getLogger(__name__)
#todo consider adding ls.traceable to LLM-calling nodes

#todo update file loading and system prompt
def build_symptex_model(document_bundle_cache: DocumentBundleCache):
    logger.info("Building symptex model")
    workflow = StateGraph(state_schema=CustomState)

    load_patient_docs_tool = make_load_patient_files_tool(document_bundle_cache)
    workflow.add_node("orchestrator_node", make_orchestrator_node([load_patient_docs_tool]))
    workflow.add_node("load_docs", make_load_docs_node(load_patient_docs_tool))
    workflow.add_node("summary_node", summary_node)
    workflow.add_node("patient_model_final", patient_model_final)

    workflow.add_edge(START, "orchestrator_node")
    workflow.add_conditional_edges(
        "orchestrator_node",
        tool_branching_node,
        {
            "abort": END,
            "has_tool_calls": "load_docs",
            "no_tool_calls": "patient_model_final",
        },
    )
    workflow.add_conditional_edges(
        "load_docs",
        docs_branching_node,
        {
            "abort": END,
            "summary": "summary_node",
            "end": "patient_model_final",
        },
    )
    workflow.add_edge("summary_node", "patient_model_final")
    workflow.add_edge("patient_model_final", END)
    return workflow.compile()
