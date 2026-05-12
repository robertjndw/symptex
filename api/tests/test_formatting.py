import datetime
from types import SimpleNamespace

from chains.formatting import format_patient_details


def _build_patient_file():
    return SimpleNamespace(
        first_name="Max",
        last_name="Mustermann",
        birth_date=datetime.date(1990, 1, 1),
        ethnic_origin="Unbekannt",
        height=180,
        weight=80.0,
        gender_medical="m",
        anamneses=[],
    )


def test_format_patient_details_includes_treatment_reason_from_active_case():
    patient_file = _build_patient_file()
    medical_case = SimpleNamespace(id=42, treatment_reason="Persistierende Kopfschmerzen")

    details = format_patient_details(patient_file, medical_case)

    assert "Behandlungsgrund: Persistierende Kopfschmerzen" in details


def test_format_patient_details_defaults_treatment_reason_to_unknown():
    patient_file = _build_patient_file()

    details = format_patient_details(patient_file)

    assert "Behandlungsgrund: Unbekannt" in details
