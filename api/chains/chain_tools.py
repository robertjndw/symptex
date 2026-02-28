from langchain_core.tools import tool

from chains.document_bundle_cache import DocumentBundleCache


def make_load_patient_files_tool(document_bundle_cache: DocumentBundleCache):
    @tool("load_patient_docs")
    def load_patient_docs_tool() -> str:
        """
        Loads and returns the full text of all available medical documents
        (e.g., doctor's letters, reports, imaging findings, lab results).

        The Orchestrator Assistant should call this tool when the doctor requests
        medical records, findings, medications, test results, imaging, or related concepts.
        """
        bundle = document_bundle_cache.ensure_loaded()
        return bundle["combined_tool_text"]

    return load_patient_docs_tool
