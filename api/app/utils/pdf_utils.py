import base64
import logging
import os
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)
container_dir = os.environ.get("ANAMNESIS_DIR", "")


def _extract_text_from_open_doc(doc: pymupdf.Document) -> str:
    all_pages_text = []

    for page_index in range(doc.page_count):
        page = doc[page_index]

        page_text = page.get_text("text")

        images = page.get_images(full=True)

        for img_index, _ in enumerate(images, start=1):
            page_text += f"\n[IMAGE {img_index} ON PAGE {page_index + 1}]\n"

        page_output = f"--- PAGE {page_index + 1} ---\n{page_text}"
        all_pages_text.append(page_output)

    return "\n\n".join(all_pages_text)


def _resolve_pdf_path(pdf_path: str) -> str:
    """
    Resolve a PDF path safely against ANAMNESIS_DIR when configured.
    Prevents path traversal outside the configured base directory.
    """
    if not container_dir:
        return pdf_path

    base_dir = Path(container_dir).resolve()
    requested_path = Path(pdf_path)

    # Keep all paths anchored under base_dir, even if pdf_path is absolute.
    if requested_path.anchor:
        requested_path = Path(*requested_path.parts[1:]) if len(requested_path.parts) > 1 else Path()

    resolved_path = (base_dir / requested_path).resolve()
    try:
        resolved_path.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"Invalid pdf path outside ANAMNESIS_DIR: {pdf_path}") from exc

    return str(resolved_path)


def parse_pdf(pdf_path: str) -> str:
    resolved_path = _resolve_pdf_path(pdf_path)
    doc = pymupdf.open(resolved_path)
    try:
        return _extract_text_from_open_doc(doc)
    finally:
        doc.close()


def parse_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return _extract_text_from_open_doc(doc)
    finally:
        doc.close()


def encode_pdf_bytes_as_base64(pdf_bytes: bytes) -> str:
    return base64.b64encode(pdf_bytes).decode("ascii")


def load_pdf_bytes_as_base64(path: str) -> dict:
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    return {
        "filename": Path(path).name,
        "content_b64": encode_pdf_bytes_as_base64(pdf_bytes),
    }


def load_pdfs_as_base64(file_paths: list[str]) -> list[dict]:
    """
    Load PDFs from disk and return a list of dicts with filename + base64 content.
    """
    docs: list[dict] = []
    for path in file_paths:
        try:
            docs.append(load_pdf_bytes_as_base64(path))
        except Exception as e:
            # optionally log and skip problematic files
            logger.error("Error loading PDF %s: %s", path, e)
    return docs
