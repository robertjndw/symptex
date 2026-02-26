from langchain_core.tools import tool

from app.utils.pdf_utils import parse_pdf

#todo if the instructor doesnt upload a document and a student asks for it, all documents will still be output. Solve this
def make_load_patient_files_tool():
    @tool("load_patient_docs")
    def load_patient_docs_tool() -> str:
        """
        Loads and returns the full text of all available medical documents
        (e.g., doctor's letters, reports, imaging findings, lab results).

        The Orchestrator Assistant should call this tool when the doctor REQUESTS for
        “Befund”, “Befunde”, "Medikamente", reports, findings, test results, imaging, or any related concept.
        """
        parts = []
        #todo call endpoint in server
        for fp in file_paths:
            text = parse_pdf(fp)
            parts.append(f"--- BEGIN DOCUMENT: {fp} ---\n{text}\n--- END DOCUMENT: {fp} ---")
        return "\n\n".join(parts)
    return load_patient_docs_tool