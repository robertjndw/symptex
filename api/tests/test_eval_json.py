import pytest

from app.utils.eval_json import (
    EVAL_CATEGORIES,
    OVERALL_CATEGORY,
    normalize_eval_result,
)


def _build_valid_payload() -> dict:
    categories = list(EVAL_CATEGORIES) + [OVERALL_CATEGORY]
    return {
        category: {
            "score": 4,
            "message": f"Bewertung fuer {category}",
            "verbesserungsvorschlag": f"Verbesserung fuer {category}",
        }
        for category in categories
    }


def test_normalize_eval_result_accepts_known_category_variant():
    payload = _build_valid_payload()
    payload["Relevante Informationen erkennen und reagiere"] = payload.pop(
        "Relevante Informationen erkennen und reagieren"
    )

    normalized = normalize_eval_result(payload)

    assert "Relevante Informationen erkennen und reagieren" in normalized
    assert "Relevante Informationen erkennen und reagiere" not in normalized


def test_normalize_eval_result_still_requires_all_categories():
    payload = _build_valid_payload()
    payload.pop("Zusammenfassung geben")

    with pytest.raises(ValueError, match="Missing or invalid category: Zusammenfassung geben"):
        normalize_eval_result(payload)


def test_normalize_eval_result_requires_verbesserungsvorschlag():
    payload = _build_valid_payload()
    payload["Gesprächsführung übernehmen"].pop("verbesserungsvorschlag")

    with pytest.raises(
        ValueError,
        match="Invalid verbesserungsvorschlag for category 'Gesprächsführung übernehmen'",
    ):
        normalize_eval_result(payload)


def test_normalize_eval_result_rejects_blank_verbesserungsvorschlag():
    payload = _build_valid_payload()
    payload["Gesamtbewertung"]["verbesserungsvorschlag"] = "   "

    with pytest.raises(
        ValueError,
        match="Invalid verbesserungsvorschlag for category 'Gesamtbewertung'",
    ):
        normalize_eval_result(payload)
