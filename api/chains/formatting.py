def format_patient_details(patient_file):
    """
    Formats patient details from a PatientFile SQLAlchemy model instance.
    Includes all anamneses/categories belonging to the patient file.
    """
    def fmt(value, unknown="Unbekannt"):
        return value if value not in (None, "", []) else unknown

    def fmt_date(d):
        return d.strftime("%d.%m.%Y") if d else "Unbekannt"

    # Build anamnesis sections dynamically (stable order)
    anamneses = list(patient_file.anamneses or [])
    anamneses.sort(key=lambda a: (a.category or "").casefold())

    sections = []
    if anamneses:
        for anam in anamneses:
            category = fmt((anam.category or "").strip(), unknown="(Ohne Kategorie)")
            answer = (anam.answer or "").strip() or "Keine Angaben"
            sections.append(f"\n---\n{category}:\n{answer}\n")
    else:
        sections.append("\n---\nAnamnesen:\nKeine Angaben\n")

    return (
        f"Name: {fmt(patient_file.first_name, '')} {fmt(patient_file.last_name, '')}\n"
        f"Geburtsdatum: {fmt_date(patient_file.birth_date)}\n"
        f"Ethnie: {fmt(patient_file.ethnic_origin)}\n"
        f"Größe: {fmt(patient_file.height)} cm\n"
        f"Gewicht: {fmt(patient_file.weight)} kg\n"
        f"Geschlecht (medizinisch): {fmt(patient_file.gender_medical)}\n"
        + "".join(sections)
    )