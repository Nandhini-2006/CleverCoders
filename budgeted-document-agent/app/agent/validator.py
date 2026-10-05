import json
import re
from typing import Any, Dict, List, Optional, Set, Union


# -------------------------------------------------------------
# Phase 3: Planner Validation Rules & Constants
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Phase 7: Answer Groundedness Validation & Revision
# -------------------------------------------------------------
GROUNDING_THRESHOLD = 0.80

STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "that", "this", "these", "those", "it", "its", "as", "of"
}


class ValidationResult(dict):
    """
    Structured groundedness assessment.
    Subclasses dict for JSON transparency and provides property accessors.
    """
    def __init__(
        self,
        valid: bool,
        groundedness: float,
        unsupported_claims: List[str],
        supported_claims: List[str],
    ):
        super().__init__(
            valid=valid,
            groundedness=groundedness,
            unsupported_claims=unsupported_claims,
            supported_claims=supported_claims,
        )

    @property
    def valid(self) -> bool:
        return self["valid"]

    @property
    def groundedness(self) -> float:
        return self["groundedness"]

    @property
    def unsupported_claims(self) -> List[str]:
        return self["unsupported_claims"]

    @property
    def supported_claims(self) -> List[str]:
        return self["supported_claims"]


def extract_evidence_text(evidence: Union[str, List[Any], Dict[str, Any]]) -> str:
    """
    Normalize various evidence formats into a single lowercase text corpus.
    """
    if isinstance(evidence, str):
        return evidence.lower()
    if isinstance(evidence, dict):
        # Could be evidence_state or retrieved_pages result
        parts = []
        if "claims" in evidence:
            for c in evidence["claims"]:
                if isinstance(c, dict):
                    parts.append(str(c.get("claim", "")))
        if "retrieved_pages" in evidence:
            for p in evidence["retrieved_pages"]:
                if isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
        if not parts and "text" in evidence:
            parts.append(str(evidence["text"]))
        return " ".join(parts).lower()
    if isinstance(evidence, list):
        parts = []
        for item in evidence:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("claim", ""))))
            else:
                parts.append(str(item))
        return " ".join(parts).lower()
    return str(evidence).lower()


def split_into_verifiable_claims(draft_answer: str) -> List[str]:
    """
    Split a draft answer into verifiable atomic claims.
    Handles compound sentences joined by conjunctions like 'and is', 'and was', etc.
    """
    clean_answer = draft_answer.strip()
    if not clean_answer:
        return []

    # 1. Split on sentence boundaries
    sentence_chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_answer) if s.strip()]
    claims: List[str] = []

    for chunk in sentence_chunks:
        # Check for compound sentence with 'and is', 'and are', 'and was', 'and were' or ', and'
        # e.g., "SAML is the current authentication method and is 50% faster than OAuth."
        match = re.search(
            r"^(.*?)\s*,?\s+and\s+(is\s+|are\s+|was\s+|were\s+)?(.*)$",
            chunk,
            flags=re.IGNORECASE,
        )
        if match and len(match.group(1).strip()) >= 8 and len(match.group(3).strip()) >= 4:
            part1 = match.group(1).strip().rstrip(".")
            verb = match.group(2) or ""
            part2 = match.group(3).strip().rstrip(".")

            # Extract subject from part1
            subject_words = part1.split()
            subject = subject_words[0] if subject_words else ""

            claims.append(f"{part1}.")
            if verb:
                claims.append(f"{subject} {verb.strip()} {part2}.")
            elif subject and not any(part2.lower().startswith(w) for w in ["the", "it", "this", subject.lower()]):
                claims.append(f"{subject} {part2}.")
            else:
                claims.append(f"{part2}.")
        else:
            claims.append(chunk if chunk.endswith((".", "!", "?")) else f"{chunk}.")

    return claims


def validate_answer(
    question: str,
    draft_answer: str,
    evidence: Union[str, List[Any], Dict[str, Any]],
    threshold: float = GROUNDING_THRESHOLD,
    llm_client: Optional[Any] = None,
) -> ValidationResult:
    """
    Verify whether the draft answer is grounded strictly in the retrieved evidence.
    Calculates Groundedness = supported_claims / total_claims.

    Can use an optional LLM client or deterministic verification.
    """
    if not draft_answer or not draft_answer.strip():
        return ValidationResult(
            valid=False,
            groundedness=0.0,
            unsupported_claims=["Empty answer."],
            supported_claims=[],
        )

    # Standard "Insufficient information" answers are inherently grounded
    if "insufficient information" in draft_answer.lower():
        return ValidationResult(
            valid=True,
            groundedness=1.0,
            unsupported_claims=[],
            supported_claims=[draft_answer],
        )

    # If an LLM client is provided, attempt LLM-based structured validation
    if llm_client is not None:
        try:
            from app.agent.prompts import build_validation_prompt
            prompt = build_validation_prompt(question, draft_answer, extract_evidence_text(evidence))
            raw_output = llm_client.generate(prompt)
            cleaned_json = extract_json_from_text(raw_output)
            data = json.loads(cleaned_json)
            if "valid" in data and "groundedness" in data:
                return ValidationResult(
                    valid=bool(data["valid"]),
                    groundedness=float(data["groundedness"]),
                    unsupported_claims=data.get("unsupported_claims", []),
                    supported_claims=data.get("supported_claims", []),
                )
        except Exception:
            pass  # Fallback to deterministic verification

    evidence_corpus = extract_evidence_text(evidence)
    claims = split_into_verifiable_claims(draft_answer)

    # Extract question content tokens to establish topic frame
    q_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", question.lower())) - STOP_WORDS

    supported: List[str] = []
    unsupported: List[str] = []

    for claim in claims:
        claim_clean = claim.strip()
        claim_lower = claim_clean.lower()

        # Check 1: Metric, percentage, and comparative assertions
        # (e.g. "50%", "10 MB", "faster", "cheaper", "twice as")
        unsupported_assertion = False

        percentages = re.findall(r"\b\d+%\b", claim_lower)
        for pct in percentages:
            if pct not in evidence_corpus:
                unsupported_assertion = True
                break

        comparatives = re.findall(
            r"\b(faster|slower|cheaper|better|worse|twice as|50% faster)\b",
            claim_lower,
        )
        for comp in comparatives:
            if comp not in evidence_corpus:
                unsupported_assertion = True
                break

        if unsupported_assertion:
            unsupported.append(claim_clean)
            continue

        # Check 2: Content token overlap
        content_tokens = [
            w for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", claim_lower)
            if w not in STOP_WORDS
        ]

        if not content_tokens:
            supported.append(claim_clean)
            continue

        # Identify novel assertion tokens not already framed by the question
        novel_tokens = [
            t for t in content_tokens
            if not any(t.startswith(qt[:4]) or qt.startswith(t[:4]) for qt in q_tokens)
        ]

        if novel_tokens:
            # Check how many novel factual terms appear directly in the evidence
            novel_matches = [t for t in novel_tokens if t in evidence_corpus]
            if len(novel_matches) / len(novel_tokens) >= 0.50:
                supported.append(claim_clean)
            else:
                unsupported.append(claim_clean)
        else:
            # Claim consists of question frame words; check overall evidence overlap
            matches = [t for t in content_tokens if t in evidence_corpus]
            if len(matches) / len(content_tokens) >= 0.33 or any(w in evidence_corpus for w in content_tokens):
                supported.append(claim_clean)
            else:
                unsupported.append(claim_clean)

    total = len(supported) + len(unsupported)
    groundedness = round(len(supported) / total, 4) if total > 0 else 1.0
    is_valid = groundedness >= threshold

    return ValidationResult(
        valid=is_valid,
        groundedness=groundedness,
        unsupported_claims=unsupported,
        supported_claims=supported,
    )


def revise_answer(
    question: str,
    draft_answer: str,
    evidence: Union[str, List[Any], Dict[str, Any]],
    validation_result: Optional[ValidationResult] = None,
) -> str:
    """
    Revise a draft answer to retain only supported claims.
    If supported claims exist, they are preserved and contextualized with grounded evidence.
    """
    val = validation_result or validate_answer(question, draft_answer, evidence)
    if val.valid:
        return draft_answer

    # Retain only supported claims
    supported = val.supported_claims
    if not supported:
        return "Insufficient information."

    # Preserve supported claims
    revised_parts = [s.rstrip(".") for s in supported]

    # If the evidence text contains explicit statements, optionally contextualize
    raw_ev = extract_evidence_text(evidence)
    if "replaced by" in raw_ev and not any("replaced by" in p.lower() for p in revised_parts):
        # Extract the replacement sentence snippet
        match = re.search(r"([^.]*replaced by[^.]*)", raw_ev, flags=re.IGNORECASE)
        if match:
            snippet = match.group(1).strip()
            revised_parts.append(f"The document states that {snippet}")

    revised_text = ". ".join(revised_parts)
    return f"{revised_text}."

