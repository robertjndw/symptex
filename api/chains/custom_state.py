from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class CustomState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model: str
    condition: str
    talkativeness: str
    patient_details: str
    patient_doc_md: list[dict]
    tool_calls: list
    docs_present: bool
    attach_docs: bool
    hard_error: bool
    docs_summary: str