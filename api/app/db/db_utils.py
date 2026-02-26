from sqlalchemy import exists
from sqlalchemy.orm import Session

from app.db.models import AnamDoc, Anamnesis


def has_anamdocs(db: Session, patient_file_id: int) -> bool:
    """
    Return whether at least one AnamDoc exists for the given patient file.
    """
    docs_available = db.query(
        exists().where(
            AnamDoc.anamnesis_id == Anamnesis.id,
            Anamnesis.patient_file_id == patient_file_id,
        )
    ).scalar()
    return bool(docs_available)
