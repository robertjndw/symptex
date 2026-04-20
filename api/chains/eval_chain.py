import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

from app.utils.eval_json import (
    EVAL_CATEGORIES,
    build_eval_response_schema,
    extract_eval_payload,
    get_eval_categories_with_overall,
    normalize_eval_result,
)
from chains.llm import get_llm

logger = logging.getLogger(__name__)


def _role_map_for_eval(messages: list) -> list:
    mapped_messages = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(msg, HumanMessage):
            mapped_messages.append(HumanMessage(content=f"Arzt: {content}"))
        elif isinstance(msg, AIMessage):
            mapped_messages.append(AIMessage(content=f"Patient: {content}"))
        else:
            mapped_messages.append(msg)
    return mapped_messages


def _build_eval_llm(model: str):
    llm = get_llm(model)
    schema = build_eval_response_schema()

    if isinstance(llm, ChatOpenAI):
        return llm.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)

    return llm.with_structured_output(schema, method="json_schema", include_raw=True)


def get_eval_prompt():
    criteria = "\n".join(f"            * {category}" for category in EVAL_CATEGORIES)
    json_keys = "\n".join(
        f"            * {category}" for category in get_eval_categories_with_overall()
    )
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                f"""
            /nothink
            Ziel: Du bist ein medizinischer Prüfer und bewertest die klinische Gesprächsführung eines Doktors während der Anamneseerhebung anhand definierter klinischer Indikatoren (CRI-HT) auf Deutsch.
            Die Bewertung erfolgt auf einer Skala von 1 bis 5 für jede Kategorie.

            Bewertungskriterien:
{criteria}

            Bewertungsskala:
            1: Kriterium nicht erfüllt
            2: Kriterium eher nicht erfüllt
            3: Teilerfüllung
            4: Kriterium weitgehend erfüllt
            5: Vollständig erfüllt

            Anweisung:
            Analysiere den vorgelegten Arzt-Patienten-Dialog und vergib für jedes der 8 Kriterien eine Punktzahl (1-5).
            Begründe jede Bewertung mit konkreten Beispielen aus dem Dialog.
            Die Bewertung soll konstruktiv sein und Verbesserungspotenziale aufzeigen.

            Gib ausschließlich ein JSON-Objekt zurück und keinen zusätzlichen Text, kein Markdown und keine Code-Fences.
            Das JSON muss exakt diese obersten Schlüssel enthalten:
{json_keys}

            Jeder dieser Schlüssel hat als Wert ein JSON-Objekt mit genau zwei Feldern:
            * score: Integer von 1 bis 5
            * message: String mit Begründung; bei den 8 Kriterien inkl. Verbesserungsvorschlag, bei Gesamtbewertung inkl. Stärken und Verbesserungspotenzial
            """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


async def eval_history(messages: list, model: str) -> str:
    try:
        prompt = get_eval_prompt()
        llm = _build_eval_llm(model)
        chain = prompt | llm

        mapped_messages = _role_map_for_eval(messages)
        logger.debug("Evaluating messages: %s", mapped_messages)
        response = await chain.ainvoke({"messages": mapped_messages})
        payload = extract_eval_payload(response)
        normalized = normalize_eval_result(payload)
        return json.dumps(normalized, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error("Error in eval_history: %s", str(e))
        return f"Entschuldigung, es ist ein Fehler aufgetreten: {str(e)}"
