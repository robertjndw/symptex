from pathlib import Path

import pytest

from app.utils import pdf_utils


def test_resolve_pdf_path_without_container_dir_returns_original(monkeypatch):
    monkeypatch.setattr(pdf_utils, "container_dir", "")

    assert pdf_utils._resolve_pdf_path("reports/file.pdf") == "reports/file.pdf"


def test_resolve_pdf_path_joins_relative_path_under_container(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_utils, "container_dir", str(tmp_path))

    resolved = pdf_utils._resolve_pdf_path("reports/file.pdf")

    assert Path(resolved) == (tmp_path / "reports" / "file.pdf").resolve()


def test_resolve_pdf_path_reanchors_absolute_path_under_container(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_utils, "container_dir", str(tmp_path))

    resolved = pdf_utils._resolve_pdf_path("/reports/file.pdf")

    assert Path(resolved) == (tmp_path / "reports" / "file.pdf").resolve()


def test_resolve_pdf_path_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_utils, "container_dir", str(tmp_path))

    with pytest.raises(ValueError, match="outside ANAMNESIS_DIR"):
        pdf_utils._resolve_pdf_path("../escape.pdf")
