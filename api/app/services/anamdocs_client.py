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
        debug_login_enabled: bool = False,
        debug_login_tum_id: str = "ADMIN1234",
        debug_login_role: str = "admin",
        debug_login_first_name: str = "Symptex",
        debug_login_last_name: str = "Debug",
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.file_server_route = "/" + file_server_route.strip("/")
        self.timeout_seconds = timeout_seconds
        self.debug_login_enabled = debug_login_enabled
        self.debug_login_tum_id = debug_login_tum_id or "ADMIN1234"
        self.debug_login_role = debug_login_role or "admin"
        self.debug_login_first_name = debug_login_first_name or "Symptex"
        self.debug_login_last_name = debug_login_last_name or "Debug"
        self._session = requests.Session()

    @classmethod
    def from_env(cls) -> "AnamDocsClient":
        api_base_url = os.getenv("ILUVI_API_BASE_URL", "").strip()
        file_server_route = os.getenv("FILE_SERVER_ROUTE", "/static")
        timeout_seconds = _read_timeout_seconds(default=10.0)
        debug_login_enabled = _read_bool_env("ILUVI_DEBUG_LOGIN_ENABLED", default=False)
        debug_login_tum_id = os.getenv("ILUVI_DEBUG_LOGIN_TUM_ID", "ADMIN1234").strip()
        debug_login_role = os.getenv("ILUVI_DEBUG_LOGIN_ROLE", "admin").strip()
        debug_login_first_name = os.getenv("ILUVI_DEBUG_LOGIN_FIRST_NAME", "Symptex").strip()
        debug_login_last_name = os.getenv("ILUVI_DEBUG_LOGIN_LAST_NAME", "Debug").strip()
        return cls(
            api_base_url=api_base_url,
            file_server_route=file_server_route,
            timeout_seconds=timeout_seconds,
            debug_login_enabled=debug_login_enabled,
            debug_login_tum_id=debug_login_tum_id,
            debug_login_role=debug_login_role,
            debug_login_first_name=debug_login_first_name,
            debug_login_last_name=debug_login_last_name,
        )

    def list_anamdocs(self, patient_file_id: int) -> list[AnamDocResponse]:
        self._validate_base_url()
        url = f"{self.api_base_url}/patientFiles/{patient_file_id}/anamneses/anam-doc"
        try:
            response = self._get_with_optional_debug_reauth(url)
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
            response = self._get_with_optional_debug_reauth(url)
        except requests.RequestException as exc:
            raise DownloadAnamDocError(f"Network error while downloading AnamDoc from {url}: {exc}") from exc
        if response.status_code >= 400:
            raise DownloadAnamDocError(
                f"Failed to download AnamDoc from {url}: HTTP {response.status_code}"
            )
        return response.content

    def _get_with_optional_debug_reauth(self, url: str) -> requests.Response:
        response = self._session.get(url, timeout=self.timeout_seconds)
        if response.status_code != 401 or not self.debug_login_enabled:
            return response

        if not self._perform_debug_login():
            return response
        return self._session.get(url, timeout=self.timeout_seconds)

    def _perform_debug_login(self) -> bool:
        debug_login_url = f"{self.api_base_url}/auth/debug-login"
        params = {
            "tumID": self.debug_login_tum_id,
            "firstName": self.debug_login_first_name,
            "lastName": self.debug_login_last_name,
            "role": self.debug_login_role,
        }
        try:
            response = self._session.get(
                debug_login_url,
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("Debug login request failed: %s", exc)
            return False

        if response.status_code != 200:
            logger.warning(
                "Debug login failed with HTTP %s for tumID=%s",
                response.status_code,
                self.debug_login_tum_id,
            )
            return False
        return True

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


def _read_bool_env(key: str, default: bool) -> bool:
    raw_value = os.getenv(key)
    if raw_value is None:
        return default
    normalized = raw_value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    return normalized.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
