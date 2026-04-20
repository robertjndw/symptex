import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

EVAL_CATEGORIES = (
    "Gesprächsführung übernehmen",
    "Relevante Informationen erkennen und reagieren",
    "Symptome präzisieren",
    "Pathophysiologisch begründete Fragen stellen",
    "Logische Fragerichtung",
    "Informationen beim Patienten rückbestätigen",
    "Zusammenfassung geben",
    "Effizienz und Datenqualität",
)
OVERALL_CATEGORY = "Gesamtbewertung"


def get_eval_categories_with_overall() -> tuple[str, ...]:
    return EVAL_CATEGORIES + (OVERALL_CATEGORY,)


def build_eval_response_schema() -> dict:
    rating_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "message": {"type": "string", "minLength": 1},
        },
        "required": ["score", "message"],
        "additionalProperties": False,
    }

    categories = list(get_eval_categories_with_overall())
    return {
        "title": "AnamneseEvaluation",
        "type": "object",
        "properties": {category: rating_schema for category in categories},
        "required": categories,
        "additionalProperties": False,
    }


def _extract_text_from_raw_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return str(content or "")


def _extract_first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_braced_region(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _try_parse_mapping_candidate(candidate: str) -> dict | None:
    stripped = candidate.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (SyntaxError, ValueError):
        pass

    return None


def _extract_mapping_from_text_relaxed(text: str) -> dict | None:
    if not text:
        return None

    direct = _try_parse_mapping_candidate(text)
    if isinstance(direct, dict):
        return direct

    braced = _extract_braced_region(text)
    if braced:
        parsed_braced = _try_parse_mapping_candidate(braced)
        if isinstance(parsed_braced, dict):
            return parsed_braced

    first_object = _extract_first_json_object(text)
    if isinstance(first_object, dict):
        return first_object

    return None


def _extract_eval_payload_by_categories(text: str) -> dict | None:
    categories = get_eval_categories_with_overall()
    positions: list[tuple[int, str]] = []

    for category in categories:
        pos = text.find(f'"{category}"')
        if pos == -1:
            return None
        positions.append((pos, category))

    positions.sort(key=lambda item: item[0])
    payload: dict[str, dict[str, Any]] = {}

    for i, (start_pos, category) in enumerate(positions):
        next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section = text[start_pos:next_pos]

        key_colon = section.find(":")
        if key_colon == -1:
            return None

        obj_start = section.find("{", key_colon)
        if obj_start == -1:
            return None

        brace_depth = 0
        obj_end = -1
        for j in range(obj_start, len(section)):
            ch = section[j]
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    obj_end = j
                    break
        if obj_end == -1:
            return None

        obj_text = section[obj_start + 1 : obj_end]

        score_match = re.search(r'"score"\s*:\s*"?([1-5])"?', obj_text)
        if not score_match:
            return None
        score = int(score_match.group(1))

        message_key = obj_text.find('"message"')
        if message_key == -1:
            return None
        message_colon = obj_text.find(":", message_key)
        if message_colon == -1:
            return None
        message_start_quote = obj_text.find('"', message_colon + 1)
        if message_start_quote == -1:
            return None

        message_end_quote = obj_text.rfind('"')
        if message_end_quote <= message_start_quote:
            return None

        message_value = obj_text[message_start_quote + 1 : message_end_quote]
        message_value = message_value.replace('\\"', '"').strip()

        payload[category] = {
            "score": score,
            "message": message_value,
        }

    return payload


def extract_eval_payload(response: Any) -> dict:

    if hasattr(response, "model_dump"):
        response = response.model_dump()

    if isinstance(response, dict) and {"raw", "parsed", "parsing_error"}.issubset(response.keys()):
        parsed = response.get("parsed")
        if isinstance(parsed, dict):
            return parsed

        parsing_error = response.get("parsing_error")

        raw = response.get("raw")
        raw_content = getattr(raw, "content", raw)
        if isinstance(raw_content, dict):
            return raw_content

        raw_text = _extract_text_from_raw_content(raw_content)
        recovered = _extract_mapping_from_text_relaxed(raw_text)
        if isinstance(recovered, dict):
            if parsing_error is not None:
                logger.warning("Recovered eval JSON from raw output after parser error: %s", parsing_error)
            return recovered

        recovered_by_categories = _extract_eval_payload_by_categories(raw_text)
        if isinstance(recovered_by_categories, dict):
            logger.warning("Recovered eval JSON by category parsing from raw output.")
            return recovered_by_categories

        recovered_from_error = _extract_mapping_from_text_relaxed(str(parsing_error or ""))
        if isinstance(recovered_from_error, dict):
            logger.warning("Recovered eval JSON from parsing_error text.")
            return recovered_from_error

        recovered_error_by_categories = _extract_eval_payload_by_categories(str(parsing_error or ""))
        if isinstance(recovered_error_by_categories, dict):
            logger.warning("Recovered eval JSON by category parsing from parsing_error text.")
            return recovered_error_by_categories

        if parsing_error is not None:
            raise ValueError(f"Invalid json output and recovery failed: {parsing_error}")
        raise ValueError("Eval model did not return a structured JSON payload.")

    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        recovered = _extract_mapping_from_text_relaxed(response)
        if isinstance(recovered, dict):
            return recovered

        recovered_by_categories = _extract_eval_payload_by_categories(response)
        if isinstance(recovered_by_categories, dict):
            return recovered_by_categories

    raise ValueError("Eval model response was not parseable as JSON.")


def normalize_eval_result(payload: dict) -> dict:
    categories = get_eval_categories_with_overall()
    normalized: dict[str, dict[str, Any]] = {}

    for category in categories:
        item = payload.get(category)
        if not isinstance(item, dict):
            raise ValueError(f"Missing or invalid category: {category}")

        score = item.get("score")
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score.strip())
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError(f"Invalid score for category '{category}': {score!r}")

        message = item.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"Invalid message for category '{category}'")

        normalized[category] = {
            "score": score,
            "message": message.strip(),
        }

    return normalized
