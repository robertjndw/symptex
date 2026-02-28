import logging
import os
from pathlib import Path
from threading import Lock
from typing import TypedDict

from app.services.anamdocs_client import (
    AnamDocResponse,
    AnamDocsClient,
    AnamDocsClientError,
    DownloadAnamDocError,
    InvalidPatientFileIdError,
    ListAnamDocsError,
)
from app.utils.pdf_utils import encode_pdf_bytes_as_base64, parse_pdf_bytes

logger = logging.getLogger("uvicorn.error")


class DocumentBundle(TypedDict):
    docs_for_frontend: list[dict]
    combined_tool_text: str


class DocumentBundleCache:
    def __init__(
        self,
        client: AnamDocsClient,
        patient_file_id: int,
        max_docs: int | None = None,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
    ) -> None:
        self.client = client
        self.patient_file_id = patient_file_id
        self.max_docs = max_docs if max_docs is not None else _read_int_env("ANAMDOCS_MAX_DOCS", 10)
        self.max_file_bytes = (
            max_file_bytes
            if max_file_bytes is not None
            else _mb_to_bytes(_read_int_env("ANAMDOCS_MAX_FILE_MB", 10))
        )
        self.max_total_bytes = (
            max_total_bytes
            if max_total_bytes is not None
            else _mb_to_bytes(_read_int_env("ANAMDOCS_MAX_TOTAL_MB", 40))
        )
        self._lock = Lock()
        self._loaded = False
        self._bundle: DocumentBundle = {"docs_for_frontend": [], "combined_tool_text": ""}

    def ensure_loaded(self) -> DocumentBundle:
        if self._loaded:
            return self._bundle

        with self._lock:
            if self._loaded:
                return self._bundle

            self._bundle = self._load_bundle()
            self._loaded = True
            return self._bundle

    def get_frontend_docs(self) -> list[dict]:
        bundle = self.ensure_loaded()
        return list(bundle["docs_for_frontend"])

    def _load_bundle(self) -> DocumentBundle:
        try:
            docs = self.client.list_anamdocs(self.patient_file_id)
        except (InvalidPatientFileIdError, ListAnamDocsError, AnamDocsClientError) as exc:
            logger.error("Failed to list AnamDocs for patient_file_id=%s: %s", self.patient_file_id, exc)
            return {"docs_for_frontend": [], "combined_tool_text": ""}

        if not docs:
            logger.info("No AnamDocs available for patient_file_id=%s", self.patient_file_id)
            return {"docs_for_frontend": [], "combined_tool_text": ""}

        docs_for_frontend: list[dict] = []
        tool_parts: list[str] = []
        total_bytes = 0
        skipped_limit = 0
        skipped_failed = 0

        for index, doc_meta in enumerate(docs):
            if index >= self.max_docs:
                skipped_limit += 1
                continue

            doc_bytes = self._safe_download(doc_meta)
            if doc_bytes is None:
                skipped_failed += 1
                continue

            file_size = len(doc_bytes)
            if file_size > self.max_file_bytes:
                skipped_limit += 1
                continue
            if total_bytes + file_size > self.max_total_bytes:
                skipped_limit += 1
                continue

            text = self._safe_parse_pdf(doc_meta, doc_bytes)
            if text is None:
                skipped_failed += 1
                continue

            filename = doc_meta.original_name or Path(doc_meta.path).name
            docs_for_frontend.append(
                {
                    "filename": filename,
                    "content_b64": encode_pdf_bytes_as_base64(doc_bytes),
                }
            )
            tool_parts.append(
                f"--- BEGIN DOCUMENT: {filename} ---\n{text}\n--- END DOCUMENT: {filename} ---"
            )
            total_bytes += file_size

        if skipped_limit:
            logger.warning(
                "Skipped %s AnamDocs due to configured limits for patient_file_id=%s",
                skipped_limit,
                self.patient_file_id,
            )
        if skipped_failed:
            logger.warning(
                "Skipped %s AnamDocs due to download/parse failures for patient_file_id=%s",
                skipped_failed,
                self.patient_file_id,
            )

        return {
            "docs_for_frontend": docs_for_frontend,
            "combined_tool_text": "\n\n".join(tool_parts),
        }

    def _safe_download(self, doc_meta: AnamDocResponse) -> bytes | None:
        try:
            return self.client.download_doc_bytes(doc_meta.path)
        except DownloadAnamDocError as exc:
            logger.error("Failed to download AnamDoc id=%s path=%s: %s", doc_meta.id, doc_meta.path, exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error while downloading AnamDoc id=%s: %s", doc_meta.id, exc)
            return None

    def _safe_parse_pdf(self, doc_meta: AnamDocResponse, doc_bytes: bytes) -> str | None:
        try:
            return parse_pdf_bytes(doc_bytes)
        except Exception as exc:
            logger.error("Failed to parse PDF AnamDoc id=%s path=%s: %s", doc_meta.id, doc_meta.path, exc)
            return None


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        return default


def _mb_to_bytes(size_mb: int) -> int:
    return size_mb * 1024 * 1024
