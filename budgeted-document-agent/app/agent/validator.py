import json
import re
from typing import Any, Dict, List, Set


ALLOWED_QUESTION_TYPES = {
    "factual",
    "comparison",
    "procedural",
    "lookup",
    "other",
}


class PlanValidationError(ValueError):
    """Controlled exception raised when LLM output violates query plan validation rules."""
    pass


def extract_json_from_text(text: str) -> str:
    """
    Extract a valid JSON substring from raw text, removing markdown fences if present.
    """
    text = text.strip()

    # Match ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Fallback: Extract from the first '{' to the last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text


def validate_query_plan(
    raw_plan: Any,
    trim_excess: bool = True,
    strict_max: bool = False,
) -> Dict[str, Any]:
    """
    Deterministically validate and sanitize the structured query plan.

    Validation rules:
    - Must be a dictionary or a valid JSON string representing a dictionary.
    - 'question_type' must be in ALLOWED_QUESTION_TYPES.
    - 'keywords' must be a list.
    - Every keyword must be a non-empty string when trimmed.
    - Duplicate keywords are removed (case-insensitive, preserving first occurrence order).
    - Keywords list must contain at least 1 item.
    - Keywords list must contain at most 3 items:
        - If trim_excess=True and not strict_max: trimmed to the first 3 items.
        - If strict_max=True or trim_excess=False: raises PlanValidationError if > 3 items.
    - Any failure raises PlanValidationError as a controlled error.

    Returns:
        Dict[str, Any]: Sanitized dictionary containing 'question_type' and 'keywords'.
    """
    # 1. Parse JSON if string input
    if isinstance(raw_plan, str):
        cleaned_str = extract_json_from_text(raw_plan)
        try:
            data = json.loads(cleaned_str)
        except Exception as e:
            raise PlanValidationError(f"Malformed LLM output: not valid JSON ({e})") from e
    elif isinstance(raw_plan, dict):
        data = raw_plan
    else:
        raise PlanValidationError(
            f"Expected dictionary or JSON string, received {type(raw_plan).__name__}"
        )

    if not isinstance(data, dict):
        raise PlanValidationError(
            f"Parsed output must be a JSON object, received {type(data).__name__}"
        )

    # 2. Validate 'question_type'
    if "question_type" not in data:
        raise PlanValidationError("Missing required field 'question_type'")

    raw_type = data["question_type"]
    if not isinstance(raw_type, str):
        raise PlanValidationError(
            f"'question_type' must be a string, got {type(raw_type).__name__}"
        )

    clean_type = raw_type.strip().lower()
    if clean_type not in ALLOWED_QUESTION_TYPES:
        raise PlanValidationError(
            f"Invalid question_type '{raw_type}'. Allowed types: {sorted(ALLOWED_QUESTION_TYPES)}"
        )

    # 3. Validate 'keywords'
    if "keywords" not in data:
        raise PlanValidationError("Missing required field 'keywords'")

    raw_keywords = data["keywords"]
    if not isinstance(raw_keywords, list):
        raise PlanValidationError(
            f"'keywords' must be a list, got {type(raw_keywords).__name__}"
        )

    # 4. Clean, trim whitespace, and deduplicate keywords
    seen: Set[str] = set()
    cleaned_keywords: List[str] = []

    for kw in raw_keywords:
        if not isinstance(kw, str):
            raise PlanValidationError(
                f"Each keyword must be a string, got {type(kw).__name__}"
            )
        trimmed = kw.strip()
        if not trimmed:
            continue
        lower_kw = trimmed.lower()
        if lower_kw not in seen:
            seen.add(lower_kw)
            cleaned_keywords.append(trimmed)

    # 5. Check item count constraints
    if len(cleaned_keywords) < 1:
        raise PlanValidationError(
            "Keywords list must contain at least 1 non-empty keyword."
        )

    if strict_max and len(cleaned_keywords) > 3:
        raise PlanValidationError(
            f"Keywords list cannot exceed 3 items, found {len(cleaned_keywords)}."
        )

    if trim_excess and len(cleaned_keywords) > 3:
        cleaned_keywords = cleaned_keywords[:3]

    if not trim_excess and len(cleaned_keywords) > 3:
        raise PlanValidationError(
            f"Keywords list cannot exceed 3 items, found {len(cleaned_keywords)}."
        )

    return {
        "question_type": clean_type,
        "keywords": cleaned_keywords,
    }
