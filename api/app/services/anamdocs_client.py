import logging
import os
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


class AnamDocsClientError(Exception):
    pass


class InvalidPatientFileIdError(AnamDocsClientError):
    pass


class ListAnamDocsError(AnamDocsClientError):
    pass


class DownloadAnamDocError(AnamDocsClientError):
    pass


@dataclass(frozen=True)
class AnamDocResponse:
    id: int
    category: str
    original_name: str
    anamnesis_id: int | None
    path: str


class AnamDocsClient:
    def __init__(
        self,
        api_base_url: str,
        file_server_route: str = "/static",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.file_server_route = "/" + file_server_route.strip("/")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()

    @classmethod
    def from_env(cls) -> "AnamDocsClient":
        api_base_url = os.getenv("ILUVI_API_BASE_URL", "").strip()
        file_server_route = os.getenv("FILE_SERVER_ROUTE", "/static")
        timeout_seconds = _read_timeout_seconds(default=10.0)
        return cls(
            api_base_url=api_base_url,
            file_server_route=file_server_route,
            timeout_seconds=timeout_seconds,
        )

    def list_anamdocs(self, patient_file_id: int) -> list[AnamDocResponse]:
        self._validate_base_url()
        url = f"{self.api_base_url}/patientFiles/{patient_file_id}/anamneses/anamDoc"
        try:
            response = self._session.get(url, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise ListAnamDocsError(f"Network error while listing AnamDocs: {exc}") from exc

        if response.status_code == 400:
            raise InvalidPatientFileIdError(
                f"Invalid patient_file_id when requesting AnamDocs: {patient_file_id}"
            )
        if response.status_code >= 500:
            raise ListAnamDocsError(
                f"Server error while listing AnamDocs: HTTP {response.status_code}"
            )
        if response.status_code != 200:
            raise ListAnamDocsError(
                f"Unexpected status while listing AnamDocs: HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ListAnamDocsError("Invalid JSON response while listing AnamDocs") from exc

        if not isinstance(payload, list):
            raise ListAnamDocsError("Unexpected AnamDocs list response format")

        docs: list[AnamDocResponse] = []
        for item in payload:
            try:
                docs.append(
                    AnamDocResponse(
                        id=int(item["id"]),
                        category=str(item["category"]),
                        original_name=str(item["originalName"]),
                        anamnesis_id=(
                            int(item["anamnesisId"])
                            if item.get("anamnesisId") is not None
                            else None
                        ),
                        path=str(item["path"]),
                    )
                )
            except Exception:
                logger.warning("Skipping malformed AnamDoc response item: %s", item)

        return docs

    def build_download_url(self, relative_path: str) -> str:
        self._validate_base_url()
        safe_relative_path = relative_path.lstrip("/")
        return f"{self.api_base_url}{self.file_server_route}/{safe_relative_path}"

    def download_doc_bytes(self, relative_path: str) -> bytes:
        url = self.build_download_url(relative_path)
        try:
            response = self._session.get(url, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise DownloadAnamDocError(f"Network error while downloading AnamDoc from {url}: {exc}") from exc
        if response.status_code >= 400:
            raise DownloadAnamDocError(
                f"Failed to download AnamDoc from {url}: HTTP {response.status_code}"
            )
        return response.content

    def _validate_base_url(self) -> None:
        if not self.api_base_url:
            raise AnamDocsClientError(
                "ILUVI_API_BASE_URL is not configured for AnamDocs integration"
            )


def _read_timeout_seconds(default: float) -> float:
    raw_value = os.getenv("ANAMDOCS_HTTP_TIMEOUT_SEC")
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
        if parsed <= 0:
            logger.warning(
                "Invalid ANAMDOCS_HTTP_TIMEOUT_SEC=%r (must be > 0). Falling back to %.1f.",
                raw_value,
                default,
            )
            return default
        return parsed
    except ValueError:
        logger.warning(
            "Invalid ANAMDOCS_HTTP_TIMEOUT_SEC=%r (not a float). Falling back to %.1f.",
            raw_value,
            default,
        )
        return default
