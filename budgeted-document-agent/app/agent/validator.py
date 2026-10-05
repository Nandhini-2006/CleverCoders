import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# -------------------------------------------------------------
# Phase 3: Planner Validation Rules, Stopwords & Constants
# -------------------------------------------------------------
ALLOWED_QUESTION_TYPES = {
    "definition",
    "explanation",
    "comparison",
    "procedure",
    "numerical",
    "formula",
    "multi_hop",
    "example",
    "list",
    "other",
    # Legacy/compatibility question types
    "factual",
    "procedural",
    "why/causal",
    "how",
    "summary",
    "timeline",
    "lookup",
    "multi-part",
}

ALLOWED_RETRIEVAL_STRATEGIES: Set[str] = {
    "exact_phrase",
    "aho_corasick",
    "stemming",
    "lemmatization",
    "ngram",
    "levenshtein",
    "bm25",
    "semantic_similarity",
    "regex",
}

DEFAULT_RETRIEVAL_STRATEGY: List[str] = [
    "exact_phrase",
    "aho_corasick",
    "stemming",
    "lemmatization",
    "ngram",
    "levenshtein",
    "bm25",
]

DEFAULT_PLANNER_STOPWORDS: Set[str] = {
    "when", "what", "where", "why", "how", "is", "was", "were", "are",
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "for",
    "with", "from", "which", "who", "whom", "whose", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "their", "our", "its", "this", "that",
    "these", "those", "there", "here", "be", "been", "being", "have", "has",
    "had", "by", "as", "about", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "work", "between", "against"
}

DEFAULT_GENERIC_WORDS: Set[str] = {
    "used", "use", "using", "term", "terms", "thing", "things", "information",
    "document", "documents", "question", "answer", "mention", "mentions", "mentioned",
    "according", "describe", "describes", "tell", "explain", "detail", "details", "overview",
    "pdf", "file", "page", "pages", "section", "difference", "differences",
    "algorithm", "algorithms", "method", "methods", "system", "systems",
    "milestone", "milestones", "timeline", "event", "events", "aspect", "aspects",
    "step", "steps", "summary", "list", "history"
}

SENTENCE_STARTERS: Set[str] = {
    "compare", "tell", "does", "do", "explain", "describe", "find", "show",
    "is", "are", "what", "when", "where", "why", "how", "who", "which"
}

KNOWN_TECHNICAL_PATTERNS: List[str] = [
    r"\bdegrees\s+of\s+freedom\b",
    r"\bconfiguration\s+space\b",
    r"\buniform-cost\s+search\b",
    r"\buniform\s+cost\s+search\b",
    r"\b[a-zA-Z0-9_-]+\*+\s+[a-zA-Z0-9_-]+\b",
    r"\b[a-zA-Z0-9_-]+\*+\b",
    r"\b(?:artificial\s+intelligence|machine\s+learning|neural\s+networks|bayesian\s+networks)\b",
    r"\b(?:programming\s+language|priority\s+function|probabilistic\s+roadmap|logic\s+theorist)\b",
    r"\b(?:bfs|dfs|ucs|prm|svm)\b",
]


class PlanValidationError(ValueError):
    """Controlled exception raised when LLM output violates query plan validation rules."""
    pass


def is_keyword_in_question(kw: str, question: str) -> bool:
    """
    Check if a keyword or phrase is grounded directly in the user question.
    Preserves acronyms and technical phrases (e.g. A*, degrees of freedom).
    Prevents hallucinating external facts (e.g. Dartmouth).
    """
    if not question or not question.strip():
        return True

    q_lower = question.strip().lower()
    kw_clean = kw.strip(" ,.?!;:()[]{}'\"")
    if not kw_clean:
        return False
    kw_lower = kw_clean.lower()

    # Direct substring match
    if kw_lower in q_lower:
        return True

    # Multi-word phrase: verify all non-stopword tokens appear in the question
    tokens = [t.strip(" ,.?!;:()[]{}'\"") for t in kw_lower.split() if t.strip(" ,.?!;:()[]{}'\"")]
    if not tokens:
        return False

    content_tokens = [t for t in tokens if t not in DEFAULT_PLANNER_STOPWORDS]
    if not content_tokens:
        content_tokens = tokens

    return all(t in q_lower for t in content_tokens)


def extract_json_from_text(text: str) -> str:
    """
    Extract a valid JSON substring from raw text, removing markdown fences or reasoning tags.
    """
    text = text.strip()

    # Strip thinking/reasoning tags if present e.g. <think>...</think>
    text_clean = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if not text_clean:
        text_clean = text

    # Match ```json ... ``` or ``` ... ``` (preferring the last block)
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean)
    if fences:
        for f in reversed(fences):
            candidate = f.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
        return fences[-1].strip()

    # Search from the last '}' backwards for valid JSON object
    last_brace = text_clean.rfind("}")
    if last_brace != -1:
        brace_indices = [m.start() for m in re.finditer(r"\{", text_clean[:last_brace])]
        for start_idx in reversed(brace_indices):
            candidate_str = text_clean[start_idx:last_brace + 1].strip()
            try:
                parsed = json.loads(candidate_str)
                if isinstance(parsed, dict):
                    return candidate_str
            except Exception:
                continue

    return text_clean



def extract_question_keywords(
    question: str,
    stopwords: Optional[Set[str]] = None,
    generic_words: Optional[Set[str]] = None,
    max_keywords: int = 3,
) -> List[str]:
    """
    Deterministically extract distinctive technical phrases and content keywords
    directly from a user question without hallucination or outside information.
    """
    sw = stopwords if stopwords is not None else DEFAULT_PLANNER_STOPWORDS
    gw = generic_words if generic_words is not None else DEFAULT_GENERIC_WORDS

    q = question.strip()
    extracted: List[str] = []
    seen: Set[str] = set()
    used_spans: List[tuple] = []

    def span_overlaps(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in used_spans)

    def add_keyword(kw: str, start: int, end: int):
        clean = kw.strip(" ,.?!;:()[]{}'\"")
        if not clean:
            return
        lower = clean.lower()
        if lower in sw or lower in gw:
            return
        if lower not in seen:
            seen.add(lower)
            extracted.append(clean)
            used_spans.append((start, end))

    # 1. Known technical patterns & idiomatic phrases (e.g. "degrees of freedom", "A* search", "machine learning")
    for pat in KNOWN_TECHNICAL_PATTERNS:
        for m in re.finditer(pat, q, re.IGNORECASE):
            if not span_overlaps(m.start(), m.end()):
                add_keyword(m.group(0), m.start(), m.end())

    # 2. Capitalized multi-word sequences (Named Entities / Proper Nouns)
    for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)+\b", q):
        if span_overlaps(m.start(), m.end()):
            continue
        phrase = m.group(0)
        p_words = [w.lower() for w in phrase.split()]
        if m.start() == 0 and p_words[0] in SENTENCE_STARTERS:
            remaining = phrase.split()[1:]
            if len(remaining) >= 2:
                phrase = " ".join(remaining)
                add_keyword(phrase, m.start() + len(p_words[0]) + 1, m.end())
            continue
        if not all(w in sw or w in gw for w in p_words):
            add_keyword(phrase, m.start(), m.end())

    # 3. Compound technical phrases (2 adjacent non-stopword content words)
    for m in re.finditer(r"\b([a-zA-Z0-9_*#-]+)\s+([a-zA-Z0-9_*#-]+)\b", q):
        if span_overlaps(m.start(), m.end()):
            continue
        w1, w2 = m.group(1), m.group(2)
        l1, l2 = w1.lower(), w2.lower()
        if l1 not in sw and l1 not in gw and len(l1) > 2:
            if l2 not in sw and l2 not in gw and len(l2) > 2:
                add_keyword(f"{w1} {w2}", m.start(), m.end())

    # 4. Single content words (nouns, distinctive verbs, technical terms)
    for m in re.finditer(r"\b[a-zA-Z0-9_*#-]+\b", q):
        if span_overlaps(m.start(), m.end()):
            continue
        token = m.group(0)
        l = token.lower()
        if l not in sw and l not in gw and len(token) > 1:
            add_keyword(token, m.start(), m.end())

    return extracted[:max_keywords]


def clean_and_filter_keywords(
    raw_keywords: List[str],
    question: Optional[str] = None,
    stopwords: Optional[Set[str]] = None,
    generic_words: Optional[Set[str]] = None,
    max_keywords: int = 3,
) -> List[str]:
    """
    Clean, validate, and ground keywords:
    - If raw_keywords is empty, returns empty list (does not fabricate keywords).
    - Filters out stopwords from configurable STOPWORDS set.
    - Filters out generic query words ("term", "thing", "information", "document").
    - Preserves meaningful multi-word phrases.
    - Merges fragments that appear adjacent in the user question.
    - If LLM returned only stopwords/generic words, extracts meaningful phrases from question.
    """
    if not raw_keywords:
        return []

    sw = stopwords if stopwords is not None else DEFAULT_PLANNER_STOPWORDS
    gw = generic_words if generic_words is not None else DEFAULT_GENERIC_WORDS

    cleaned: List[str] = []
    seen: Set[str] = set()

    for item in raw_keywords:
        if not isinstance(item, str):
            continue
        trimmed = item.strip(" ,.?!;:()[]{}'\"")
        if not trimmed:
            continue

        lower = trimmed.lower()
        if lower in sw or lower in gw:
            continue

        words = trimmed.split()
        if len(words) > 1:
            while words and words[0].lower() in sw:
                words.pop(0)
            while words and words[-1].lower() in sw:
                words.pop()
            if not words:
                continue
            trimmed = " ".join(words)
            lower = trimmed.lower()
            if lower in sw or lower in gw:
                continue

        if lower not in seen:
            seen.add(lower)
            cleaned.append(trimmed)

    # Phrase preservation & question-based grounding
    if question and question.strip():
        q_text = question.strip()

        # Merge adjacent split single-word keywords if they form an intact phrase in the question
        merged: List[str] = []
        i = 0
        while i < len(cleaned):
            if (
                i + 1 < len(cleaned)
                and len(cleaned[i].split()) == 1
                and len(cleaned[i + 1].split()) == 1
            ):
                candidate_pair = f"{cleaned[i]} {cleaned[i+1]}"
                if candidate_pair.lower() in q_text.lower():
                    merged.append(candidate_pair)
                    i += 2
                    continue
            merged.append(cleaned[i])
            i += 1
        cleaned = merged

        # If LLM returned only stopwords/generic words, fall back to non-stopword question phrases
        if not cleaned and raw_keywords:
            cleaned = extract_question_keywords(q_text, stopwords=sw, generic_words=gw, max_keywords=max_keywords)

    # Deduplicate preserving order
    final_keywords: List[str] = []
    final_seen: Set[str] = set()
    for kw in cleaned:
        l = kw.lower()
        if l not in final_seen:
            final_seen.add(l)
            final_keywords.append(kw)

    return final_keywords



def extract_technical_phrases(question: str, max_phrases: int = 5) -> List[str]:
    """
    Extract important multi-word technical phrases from the question.
    Examples: 'heuristic function', 'configuration space', 'uniform cost search', 'probabilistic roadmap'.
    """
    if not question or not question.strip():
        return []

    phrases: List[str] = []
    seen: Set[str] = set()

    # 1. Match known technical patterns
    for pat in KNOWN_TECHNICAL_PATTERNS:
        for m in re.finditer(pat, question, re.IGNORECASE):
            ph = m.group(0).strip()
            if ph.lower() not in seen:
                seen.add(ph.lower())
                phrases.append(ph)
                if len(phrases) >= max_phrases:
                    return phrases

    # 2. Match capitalized technical noun phrases (e.g. "Artificial Intelligence", "Deep Blue")
    matches = re.findall(r"\b([A-Z][a-zA-Z0-9*#-]+(?:\s+[A-Z][a-zA-Z0-9*#-]+)+)\b", question)
    for m in matches:
        if m.lower() not in seen:
            seen.add(m.lower())
            phrases.append(m)
            if len(phrases) >= max_phrases:
                return phrases

    # 3. Match adjacent content word pairs (excluding stopwords and generic words)
    tokens = [
        w for w in re.findall(r"\b[a-zA-Z0-9*#-]+\b", question)
        if w.lower() not in DEFAULT_PLANNER_STOPWORDS and w.lower() not in DEFAULT_GENERIC_WORDS and len(w) > 2
    ]
    for i in range(len(tokens) - 1):
        pair = f"{tokens[i]} {tokens[i+1]}"
        if pair.lower() in question.lower() and pair.lower() not in seen:
            seen.add(pair.lower())
            phrases.append(pair)
            if len(phrases) >= max_phrases:
                return phrases

    return phrases[:max_phrases]


def normalize_term(term: str) -> List[str]:
    """
    Normalize important terms using:
    1. lowercase
    2. stemming
    3. lightweight lemmatization
    Preserves original technical term as well.
    """
    if not term or not term.strip():
        return []

    results: List[str] = [term.strip()]
    lower = term.lower().strip()
    if lower not in results:
        results.append(lower)

    words = lower.split()
    stemmed_words = []
    lemmatized_words = []

    for w in words:
        # Lightweight lemmatization heuristics (plurals, irregulars)
        lem = w
        if lem.endswith("ies") and len(lem) > 4:
            lem = lem[:-3] + "y"
        elif lem.endswith("es") and len(lem) > 3 and lem[-3] in ("s", "x", "z", "c", "h"):
            lem = lem[:-2]
        elif lem.endswith("s") and not lem.endswith("ss") and len(lem) > 3:
            lem = lem[:-1]
        lemmatized_words.append(lem)

        # Lightweight stemming heuristics (common suffixes)
        stem = lem
        for suffix in ("ing", "ed", "tional", "tion", "ator", "ment", "able", "ible", "ity"):
            if stem.endswith(suffix) and len(stem) > len(suffix) + 3:
                stem = stem[:-len(suffix)]
                break
        stemmed_words.append(stem)

    stemmed_term = " ".join(stemmed_words)
    lemmatized_term = " ".join(lemmatized_words)

    if lemmatized_term not in results:
        results.append(lemmatized_term)
    if stemmed_term not in results:
        results.append(stemmed_term)

    return results


def normalize_terms(terms: List[str]) -> List[str]:
    """
    Normalize a list of terms, preserving original and adding normalized variants.
    """
    normalized: List[str] = []
    seen: Set[str] = set()
    for t in terms:
        for v in normalize_term(t):
            if v and v.lower() not in seen:
                seen.add(v.lower())
                normalized.append(v)
    return normalized


def build_fallback_retrieval_plan(question: str) -> Dict[str, Any]:
    """
    Deterministic robustness fallback:
    Never allow planner failure to result in zero retrieval attempts.
    """
    q_clean = question.strip() if question else ""
    extracted_keywords = extract_question_keywords(q_clean, max_keywords=3)
    if not extracted_keywords and q_clean:
        tokens = [
            w for w in re.findall(r"\b[a-zA-Z0-9*#-]+\b", q_clean)
            if w.lower() not in DEFAULT_PLANNER_STOPWORDS and len(w) > 1
        ]
        extracted_keywords = tokens[:3] if tokens else ["document"]

    extracted_phrases = extract_technical_phrases(q_clean, max_phrases=5)
    normalized_keywords = normalize_terms(extracted_keywords + extracted_phrases)

    retrieval_strategy = [
        "exact_phrase",
        "aho_corasick",
        "stemming",
        "lemmatization",
        "ngram",
        "levenshtein",
        "bm25",
    ]
    if q_clean and any(c in q_clean for c in "*+-/=<>#0123456789"):
        retrieval_strategy.insert(2, "regex")

    return {
        "question_type": "other",
        "search_query": q_clean,
        "query": q_clean,
        "keywords": extracted_keywords,
        "synonyms": [],
        "phrases": extracted_phrases,
        "normalized_terms": normalized_keywords,
        "retrieval_strategy": retrieval_strategy,
        "important_concepts": extracted_keywords,
        "requested_entities": extracted_keywords[:1],
        "requested_attributes": [],
        "expected_answer_structure": "direct answer",
    }


def parse_query_plan(
    response: Any,
    question: Optional[str] = None,
    stopwords: Optional[Set[str]] = None,
    generic_words: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Robust JSON parser and deterministic validator for Query Planning & Retrieval Planning.

    Features:
    - Extracts JSON safely handling thinking/reasoning tags or markdown fences.
    - Handles malformed LLM outputs with deterministic fallback (never zero retrieval).
    - Enforces schema: question_type, search_query, keywords, synonyms, phrases,
      normalized_terms, retrieval_strategy.
    - Preserves exact phrases and normalization.
    """
    q_str = question.strip() if question else ""

    if response is None:
        return build_fallback_retrieval_plan(q_str)

    data = None
    if isinstance(response, dict):
        data = response
    elif isinstance(response, str):
        extracted_str = extract_json_from_text(response)
        try:
            candidate = json.loads(extracted_str)
            if isinstance(candidate, dict):
                data = candidate
        except Exception:
            data = None

        if data is None:
            return {
                "question_type": "other",
                "search_query": q_str,
                "query": q_str,
                "keywords": [],
                "synonyms": [],
                "phrases": extract_technical_phrases(q_str),
                "normalized_terms": normalize_terms(extract_question_keywords(q_str)),
                "retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
                "important_concepts": [],
                "requested_entities": [],
                "requested_attributes": [],
                "expected_answer_structure": "direct answer",
            }
    else:
        return {
            "question_type": "other",
            "search_query": q_str,
            "query": q_str,
            "keywords": [],
            "synonyms": [],
            "phrases": extract_technical_phrases(q_str),
            "normalized_terms": normalize_terms(extract_question_keywords(q_str)),
            "retrieval_strategy": DEFAULT_RETRIEVAL_STRATEGY,
            "important_concepts": [],
            "requested_entities": [],
            "requested_attributes": [],
            "expected_answer_structure": "direct answer",
        }

    # 1. Validate question_type
    raw_type = data.get("question_type")
    if isinstance(raw_type, str):
        clean_type = raw_type.strip().lower()
        if clean_type not in ALLOWED_QUESTION_TYPES:
            clean_type = "other"
    else:
        clean_type = "other"

    # 2. Validate & clean keywords
    raw_keywords = data.get("keywords")
    if isinstance(raw_keywords, list):
        if len(raw_keywords) > 0:
            cleaned_keywords = clean_and_filter_keywords(
                raw_keywords=raw_keywords,
                question=question,
                stopwords=stopwords,
                generic_words=generic_words,
                max_keywords=3,
            )
            final_keywords = cleaned_keywords[:3]
        else:
            final_keywords = []
    else:
        final_keywords = []

    # 3. Preserve search_query
    raw_query = data.get("search_query") or data.get("query")
    if isinstance(raw_query, str) and raw_query.strip():
        search_query = raw_query.strip()
    elif q_str:
        search_query = q_str
    else:
        search_query = " ".join(final_keywords)

    # 4. Extract phrases
    raw_phrases = data.get("phrases")
    if isinstance(raw_phrases, list) and raw_phrases:
        phrases = [p.strip() for p in raw_phrases if isinstance(p, str) and p.strip()][:5]
    else:
        phrases = extract_technical_phrases(q_str, max_phrases=5)

    # 5. Extract synonyms (max 5 per keyword, max 15 total)
    raw_synonyms = data.get("synonyms")
    synonyms: List[str] = []
    if isinstance(raw_synonyms, list):
        for s in raw_synonyms:
            if isinstance(s, str) and s.strip() and s.strip() not in synonyms:
                synonyms.append(s.strip())
                if len(synonyms) >= 15:
                    break

    # 6. Extract normalized_terms
    raw_normalized = data.get("normalized_terms")
    if isinstance(raw_normalized, list) and raw_normalized:
        normalized_terms = [n.strip() for n in raw_normalized if isinstance(n, str) and n.strip()]
    else:
        normalized_terms = normalize_terms(final_keywords + phrases)

    # 7. Extract retrieval_strategy
    raw_strategy = data.get("retrieval_strategy")
    if isinstance(raw_strategy, list) and raw_strategy:
        retrieval_strategy = [s.strip() for s in raw_strategy if isinstance(s, str) and s.strip() in ALLOWED_RETRIEVAL_STRATEGIES]
    else:
        retrieval_strategy = [
            "exact_phrase",
            "aho_corasick",
            "stemming",
            "lemmatization",
            "ngram",
            "levenshtein",
            "bm25",
        ]
        if q_str and (any(c in q_str for c in "*+-/=<>#0123456789") or clean_type == "formula"):
            retrieval_strategy.insert(2, "regex")

    # 8. Populate semantic query plan fields
    structure_map = {
        "timeline": "chronological list/summary",
        "comparison": "explicit comparison",
        "procedure": "steps/process",
        "procedural": "steps/process",
        "how": "steps/process",
        "definition": "definition",
        "formula": "direct formula/lookup",
        "lookup": "direct formula/lookup",
        "numerical": "numerical value/limit",
        "list": "bulleted list",
        "why/causal": "causal explanation",
        "summary": "summary",
        "multi-part": "multi-component breakdown",
        "multi_hop": "multi-hop lookup/synthesis",
        "example": "concrete example/illustration",
        "factual": "direct answer",
        "explanation": "explanation",
        "other": "direct answer",
    }

    raw_concepts = data.get("important_concepts")
    important_concepts = raw_concepts if isinstance(raw_concepts, list) and raw_concepts else final_keywords

    raw_entities = data.get("requested_entities")
    requested_entities = raw_entities if isinstance(raw_entities, list) and raw_entities else (final_keywords[:1] if final_keywords else [])

    raw_attrs = data.get("requested_attributes")
    requested_attributes = raw_attrs if isinstance(raw_attrs, list) else []

    raw_structure = data.get("expected_answer_structure")
    expected_structure = raw_structure if isinstance(raw_structure, str) and raw_structure.strip() else structure_map.get(clean_type, "direct answer")

    return {
        "question_type": clean_type,
        "search_query": search_query,
        "query": search_query,
        "keywords": final_keywords,
        "synonyms": synonyms,
        "phrases": phrases,
        "normalized_terms": normalized_terms,
        "retrieval_strategy": retrieval_strategy,
        "important_concepts": important_concepts,
        "requested_entities": requested_entities,
        "requested_attributes": requested_attributes,
        "expected_answer_structure": expected_structure,
    }




def validate_query_plan(
    raw_plan: Any,
    trim_excess: bool = True,
    strict_max: bool = False,
    question: Optional[str] = None,
    stopwords: Optional[Set[str]] = None,
    generic_words: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Deterministically validate, sanitize, and ground the structured query plan.

    Validation & Cleaning pipeline:
    - Must be a dictionary or a valid JSON string representing a dictionary.
    - 'question_type' must be in ALLOWED_QUESTION_TYPES.
    - 'keywords' must be a list containing at least 1 keyword.
    - Stopword removal (using configurable STOPWORDS set).
    - Generic-word filtering ("term", "thing", "information", "document").
    - Multi-word phrase preservation ("Artificial Intelligence", "A* search").
    - Maximum 3 keywords.
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

    if len(raw_keywords) == 0:
        raise PlanValidationError(
            "Keywords list must contain at least 1 non-empty keyword."
        )

    for kw in raw_keywords:
        if not isinstance(kw, str):
            raise PlanValidationError(
                f"Each keyword must be a string, got {type(kw).__name__}"
            )

    # 4. Clean, remove stopwords, filter generic words, and preserve phrases
    cleaned_keywords = clean_and_filter_keywords(
        raw_keywords=raw_keywords,
        question=question,
        stopwords=stopwords,
        generic_words=generic_words,
        max_keywords=3,
    )

    # 5. Check item count constraints
    if len(cleaned_keywords) < 1:
        if question and question.strip():
            cleaned_keywords = extract_question_keywords(
                question,
                stopwords=stopwords,
                generic_words=generic_words,
                max_keywords=3,
            )
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

    raw_concepts = data.get("important_concepts")
    important_concepts = raw_concepts if isinstance(raw_concepts, list) and raw_concepts else cleaned_keywords

    raw_entities = data.get("requested_entities")
    requested_entities = raw_entities if isinstance(raw_entities, list) and raw_entities else (cleaned_keywords[:1] if cleaned_keywords else [])

    raw_attrs = data.get("requested_attributes")
    requested_attributes = raw_attrs if isinstance(raw_attrs, list) else []

    structure_map = {
        "timeline": "chronological list/summary",
        "comparison": "explicit comparison",
        "procedural": "steps/process",
        "how": "steps/process",
        "definition": "definition",
        "lookup": "direct formula/lookup",
        "list": "bulleted list",
        "why/causal": "causal explanation",
        "summary": "summary",
        "multi-part": "multi-component breakdown",
        "factual": "direct answer",
        "explanation": "explanation",
        "other": "direct answer",
    }
    raw_structure = data.get("expected_answer_structure")
    expected_structure = raw_structure if isinstance(raw_structure, str) and raw_structure.strip() else structure_map.get(clean_type, "direct answer")

    return {
        "question_type": clean_type,
        "keywords": cleaned_keywords,
        "important_concepts": important_concepts,
        "requested_entities": requested_entities,
        "requested_attributes": requested_attributes,
        "expected_answer_structure": expected_structure,
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


COMMON_PREDICATE_VERBS: Set[str] = {
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "means", "refers", "defines", "defined", "consists",
    "involves", "includes", "included", "provides", "provided", "uses", "used",
    "requires", "required", "allows", "operates", "connects", "selects",
    "implements", "implemented", "states", "stated", "represents", "causes",
    "caused", "computes", "calculated", "born", "created", "adopted", "demonstrated",
    "run", "runs", "running", "ran", "executes", "executed", "stores", "stored",
    "supports", "supported", "contains", "contained", "occurs", "occurred",
    "generates", "generated", "produces", "produced", "manages", "managed",
    "starts", "started", "stops", "stopped", "moves", "moved", "collides"
}

HEADING_NUMBER_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*|[A-Z]\.|\bsection\s+\d+|\bchapter\s+\d+)\s+([A-Za-z].*)$",
    re.IGNORECASE
)

META_REFERENCE_PATTERNS = [
    r"\b(?:is|are)\s+discussed\s+in\s+(?:section|chapter|page|part)\b",
    r"\bsee\s+(?:section|chapter|page|part|figure|table)\s+\d+\b",
    r"\brefer\s+to\s+(?:section|chapter|page|part)\s+\d+\b",
    r"\b(?:detailed|described|explained)\s+in\s+(?:section|chapter|page)\b",
    r"\bmentioned\s+in\s+section\b",
]


class ValidationResult(dict):
    """
    Structured answer quality and groundedness assessment.
    Subclasses dict for JSON transparency and provides property accessors.
    """
    def __init__(
        self,
        valid: bool,
        groundedness: float,
        unsupported_claims: List[str],
        supported_claims: List[str],
        relevance: float = 1.0,
        coverage: float = 1.0,
        completeness: float = 1.0,
        status: str = "answered",
        rejection_reason: Optional[str] = None,
    ):
        super().__init__(
            valid=valid,
            groundedness=groundedness,
            relevance=relevance,
            coverage=coverage,
            completeness=completeness,
            status=status,
            rejection_reason=rejection_reason,
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
    def relevance(self) -> float:
        return self["relevance"]

    @property
    def coverage(self) -> float:
        return self["coverage"]

    @property
    def completeness(self) -> float:
        return self["completeness"]

    @property
    def status(self) -> str:
        return self["status"]

    @property
    def rejection_reason(self) -> Optional[str]:
        return self.get("rejection_reason")

    @property
    def unsupported_claims(self) -> List[str]:
        return self["unsupported_claims"]

    @property
    def supported_claims(self) -> List[str]:
        return self["supported_claims"]


def is_heading_or_fragment_answer(
    answer_text: str,
    question: str,
    question_type: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Detect whether a draft answer merely repeats headings, table-of-contents entries,
    section titles, meta-references, or isolated fragments without answering the question.
    """
    clean_ans = str(answer_text or "").strip()
    if not clean_ans:
        return True, "Empty answer."

    # Standard "Insufficient information" answers are valid and not heading stubs
    if clean_ans.lower().startswith("insufficient information"):
        return False, None

    # 1. Meta-reference detection (e.g. "TCP is discussed in Section 4.")
    for pat in META_REFERENCE_PATTERNS:
        if re.search(pat, clean_ans, re.IGNORECASE):
            return True, f"Answer is a meta-reference without substantive content: '{clean_ans}'"

    # 2. Section number prefix (e.g. "3.2 TCP", "Networking. 3.2 TCP.")
    if HEADING_NUMBER_PATTERN.match(clean_ans) and len(clean_ans.split()) <= 8:
        return True, f"Answer is a section number heading: '{clean_ans}'"

    # 3. Dot-separated title phrases where parts lack predicate verbs (e.g. "Introduction and History. Brief history of AI.")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_ans) if s.strip()]
    if len(sentences) >= 2:
        is_all_headings = True
        for s in sentences:
            tokens = [w.lower().strip(" ,:;()[]\"'") for w in s.split() if w.strip(" ,:;()[]\"'")]
            has_verb = any(w in COMMON_PREDICATE_VERBS for w in tokens)
            if has_verb or len(tokens) > 10:
                is_all_headings = False
                break
        if is_all_headings and len(sentences) <= 3 and len(clean_ans.split()) <= 14:
            return True, f"Answer consists of section heading titles without explanatory verbs: '{clean_ans}'"

    # 4. Short noun phrase without verbs (e.g. "Reset procedure.", "Failure analysis.")
    tokens = [w.lower().strip(" ,.?!;:()[]{}'\"") for w in clean_ans.split()]
    tokens = [t for t in tokens if t]
    has_verb = any(t in COMMON_PREDICATE_VERBS for t in tokens)

    # If it contains numbers, times, percentages, or units, it's an assertion, not a heading
    has_metric_or_time = bool(re.search(r"\b(?:\d+(?:\.\d+)?\s*(?:am|pm|mb|gb|kb|seconds|minutes|hours|%|km/h)?|\d{4})\b", clean_ans, re.IGNORECASE))

    q_lower = question.lower()
    needs_substantive = any(term in q_lower for term in [
        "explain", "how", "what is", "what are", "compare", "describe", "why", "procedure", "milestones", "history", "causes"
    ])
    if needs_substantive and not has_verb and not has_metric_or_time and len(tokens) <= 7:
        return True, f"Answer is a short noun phrase/heading without a verb or predicate: '{clean_ans}'"

    # 5. Heading that merely echoes question words without new information
    q_words = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", q_lower)) - STOP_WORDS
    ans_words = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", clean_ans.lower())) - STOP_WORDS
    if ans_words and ans_words.issubset(q_words) and not has_verb and not has_metric_or_time and len(tokens) <= 6:
        return True, f"Answer merely echoes question words without answering: '{clean_ans}'"

    return False, None


def compute_answer_relevance(
    question: str,
    answer_text: str,
    query_plan: Optional[Dict[str, Any]] = None
) -> float:
    """
    Evaluate whether the answer directly addresses what was requested in the question.
    """
    clean_ans = str(answer_text or "").strip()
    if not clean_ans:
        return 0.0

    if clean_ans.lower().startswith("insufficient information"):
        return 1.0

    # Reject headings and fragments with very low relevance score
    is_heading, _ = is_heading_or_fragment_answer(clean_ans, question)
    if is_heading:
        return 0.20

    q_lower = question.lower()
    ans_lower = clean_ans.lower()

    # Extract target concepts from query plan or question
    target_concepts: List[str] = []
    if query_plan and isinstance(query_plan, dict):
        target_concepts.extend(query_plan.get("important_concepts", []))
        target_concepts.extend(query_plan.get("keywords", []))
    if not target_concepts:
        target_concepts = extract_question_keywords(question, max_keywords=4)

    # Check comparison question relevance
    q_type = query_plan.get("question_type") if (query_plan and isinstance(query_plan, dict)) else None
    is_comparison = q_type == "comparison" or any(w in q_lower for w in ["compare", "difference", "versus", " vs "])
    if is_comparison and len(target_concepts) >= 2:
        c_cleaned = []
        for tc in target_concepts[:2]:
            tc_words = [w for w in re.findall(r"\b[A-Za-z0-9_-]+\b", tc.lower()) if w not in SENTENCE_STARTERS and w not in DEFAULT_PLANNER_STOPWORDS]
            c_cleaned.append(tc_words or [tc.lower()])
        c1_words, c2_words = c_cleaned[0], c_cleaned[1]
        has_c1 = any(w in ans_lower for w in c1_words)
        has_c2 = any(w in ans_lower for w in c2_words)
        if has_c1 and has_c2:
            return 0.95
        elif has_c1 or has_c2:
            return 0.40

    if target_concepts:
        matched = 0
        for tc in target_concepts:
            tc_words = [w for w in re.findall(r"\b[A-Za-z0-9_-]+\b", tc.lower()) if w not in SENTENCE_STARTERS and w not in DEFAULT_PLANNER_STOPWORDS]
            if not tc_words:
                tc_words = tc.lower().split()
            if any(w in ans_lower for w in tc_words):
                matched += 1
        concept_ratio = matched / len(target_concepts)
        if matched > 0:
            return round(max(0.75, min(1.0, 0.70 + 0.30 * concept_ratio)), 2)
        else:
            return 0.30

    return 0.85


def compute_answer_coverage(
    question: str,
    evidence: Any,
    query_plan: Optional[Dict[str, Any]] = None,
    answer_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generic concept coverage estimator.
    Determines whether the retrieved evidence covers the requested information.
    """
    evidence_corpus = extract_evidence_text(evidence)

    required: List[str] = []
    candidates: List[str] = []
    if query_plan and isinstance(query_plan, dict):
        candidates.extend(query_plan.get("important_concepts", []))
        candidates.extend(query_plan.get("requested_entities", []))
        candidates.extend(query_plan.get("requested_attributes", []))

    for c in candidates:
        if not c or not isinstance(c, str):
            continue
        c_clean = c.strip()
        c_tokens = [
            t for t in re.findall(r"\b[a-zA-Z0-9*#-]+\b", c_clean.lower())
            if t not in DEFAULT_PLANNER_STOPWORDS and t not in DEFAULT_GENERIC_WORDS and len(t) >= 2
        ]
        # Only require this component if it contains non-generic substantive tokens
        if c_tokens and c_clean.lower() not in [r.lower() for r in required]:
            required.append(c_clean)

    if not required:
        # Check comparison or list conjoined entities e.g. "A, B, and C"
        match_conjoined = re.findall(r"\b([a-zA-Z0-9*#-]+(?:\s+[a-zA-Z0-9*#-]+)?)\s*(?:,|\band\b)", question, re.IGNORECASE)
        filtered_conjoined = [
            m.strip() for m in match_conjoined
            if m.lower().strip() not in DEFAULT_PLANNER_STOPWORDS and m.lower().strip() not in DEFAULT_GENERIC_WORDS and len(m.strip()) >= 2
        ]
        if len(filtered_conjoined) >= 2:
            required = filtered_conjoined
        else:
            required = extract_question_keywords(question, max_keywords=3)

    if not required:
        required = ["main topic"]

    covered: List[str] = []
    missing: List[str] = []

    for req in required:
        req_clean = req.lower().strip()
        tokens = [
            t for t in re.findall(r"\b[a-zA-Z0-9*#-]+\b", req_clean)
            if t not in DEFAULT_PLANNER_STOPWORDS and t not in DEFAULT_GENERIC_WORDS and len(t) >= 2
        ]
        if not tokens:
            tokens = [t for t in req_clean.split() if t not in DEFAULT_PLANNER_STOPWORDS]
        if not tokens:
            tokens = req_clean.split()

        in_evidence = any(t in evidence_corpus for t in tokens)
        if in_evidence:
            covered.append(req)
        else:
            missing.append(req)

    coverage_ratio = len(covered) / len(required) if required else 1.0
    is_complete = (len(missing) == 0) or (coverage_ratio >= 0.85)

    return {
        "coverage_ratio": round(coverage_ratio, 4),
        "is_complete": is_complete,
        "required_components": required,
        "covered_components": covered,
        "missing_components": missing,
        "details": f"Covered: {covered}, Missing: {missing}",
    }


def compute_answer_completeness(
    question_type: str,
    answer_text: str,
    query_plan: Optional[Dict[str, Any]] = None
) -> float:
    """
    Evaluate structural completeness of an answer for its question type.
    """
    clean_ans = str(answer_text or "").strip()
    if not clean_ans:
        return 0.0

    if clean_ans.lower().startswith("insufficient information"):
        return 1.0

    words = clean_ans.split()
    total_words = len(words)

    # 1. Timeline questions: expect chronological markers or multiple milestones
    if question_type == "timeline":
        years = re.findall(r"\b(19\d\d|20\d\d)\b", clean_ans)
        time_markers = re.findall(r"\b(in \d{4}|first|then|later|subsequently|began|adopted|developed|milestone|events)\b", clean_ans, re.IGNORECASE)
        has_items = "\n-" in clean_ans or "\n*" in clean_ans or bool(re.search(r"\d+\.\s+", clean_ans))
        if years or len(time_markers) >= 2 or has_items:
            return 0.95
        if total_words < 15:
            return 0.30
        return 0.65

    # 2. Procedural questions: expect steps or sequential flow
    if question_type in ("procedural", "how"):
        has_steps = bool(re.search(r"(?:step\s+\d+|1\.|2\.|first|second|then|finally|next)", clean_ans, re.IGNORECASE))
        if has_steps or total_words >= 15:
            return 0.90
        return 0.40

    # 3. Comparison questions: expect substantive contrast
    if question_type == "comparison":
        has_contrast = bool(re.search(r"(?:whereas|while|in contrast|differs|unlike|compared to|on the other hand|both|table|\|)", clean_ans, re.IGNORECASE))
        if (has_contrast and total_words >= 12) or "|" in clean_ans:
            return 0.95
        if total_words < 12:
            return 0.40
        return 0.70

    # 4. Multi-part questions: expect multi-sentence answer
    if question_type == "multi-part":
        sentences = [s for s in re.split(r"[.!?]\s+", clean_ans) if len(s.strip()) > 5]
        if len(sentences) >= 2:
            return 0.95
        return 0.50

    # Complete sentence with verb predicate
    has_verb = any(w.lower().strip(" ,:;.!?()[]\"'") in COMMON_PREDICATE_VERBS for w in words)
    if has_verb and total_words >= 4:
        return 0.90

    if total_words >= 8:
        return 0.90
    return 0.40


def extract_evidence_text(evidence: Union[str, List[Any], Dict[str, Any]]) -> str:
    """
    Normalize various evidence formats into a single lowercase text corpus.
    """
    if isinstance(evidence, str):
        return evidence.lower()
    if isinstance(evidence, dict):
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
    clean_answer = str(draft_answer or "").strip()
    if not clean_answer:
        return []

    sentence_chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_answer) if s.strip()]
    claims: List[str] = []

    for chunk in sentence_chunks:
        match = re.search(
            r"^(.*?)\s*,?\s+and\s+(is\s+|are\s+|was\s+|were\s+)?(.*)$",
            chunk,
            flags=re.IGNORECASE,
        )
        if match and len(match.group(1).strip()) >= 8 and len(match.group(3).strip()) >= 4:
            part1 = match.group(1).strip().rstrip(".")
            verb = match.group(2) or ""
            part2 = match.group(3).strip().rstrip(".")

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
    draft_answer: Any,
    evidence: Any,
    query_plan: Optional[Any] = None,
    threshold: float = GROUNDING_THRESHOLD,
    llm_client: Optional[Any] = None,
) -> ValidationResult:
    """
    Verify whether the draft answer is grounded, relevant, complete, and covers
    the requested information.
    """
    # Normalize flexible signature:
    # Mode A: validate_answer(question, draft_answer, evidence, query_plan=None, threshold=0.80, llm_client=None)
    # Mode B: validate_answer(question, query_plan, evidence, answer)
    actual_plan = None
    actual_draft_answer = draft_answer
    actual_evidence = evidence
    actual_threshold = threshold

    if isinstance(draft_answer, (dict, list)) and not isinstance(draft_answer, str):
        actual_plan = draft_answer
        actual_evidence = evidence
        actual_draft_answer = query_plan if isinstance(query_plan, str) else str(query_plan or "")
    else:
        if isinstance(query_plan, (dict, list)):
            actual_plan = query_plan
        elif isinstance(query_plan, (int, float)):
            actual_threshold = float(query_plan)

    clean_ans = str(actual_draft_answer or "").strip()

    if not clean_ans:
        return ValidationResult(
            valid=False,
            groundedness=0.0,
            unsupported_claims=["Empty answer."],
            supported_claims=[],
            relevance=0.0,
            coverage=0.0,
            completeness=0.0,
            status="validation_failed",
            rejection_reason="Empty answer.",
        )

    # Standard "Insufficient information" answers are inherently grounded and valid
    if clean_ans.lower().startswith("insufficient information"):
        return ValidationResult(
            valid=True,
            groundedness=1.0,
            unsupported_claims=[],
            supported_claims=[clean_ans],
            relevance=1.0,
            coverage=1.0,
            completeness=1.0,
            status="insufficient_information",
            rejection_reason=None,
        )

    q_type = actual_plan.get("question_type", "factual") if isinstance(actual_plan, dict) else "factual"

    # Step 1: Check if answer merely repeats headings, TOC entries, or meta-references
    is_heading, heading_reason = is_heading_or_fragment_answer(clean_ans, question, q_type)
    if is_heading:
        claims = split_into_verifiable_claims(clean_ans)
        evidence_corpus = extract_evidence_text(actual_evidence)
        supported = [c for c in claims if any(w in evidence_corpus for w in c.lower().split() if len(w) > 3)]
        groundedness = round(len(supported) / len(claims), 4) if claims else 1.0

        return ValidationResult(
            valid=False,
            groundedness=groundedness,
            unsupported_claims=[],
            supported_claims=supported,
            relevance=0.20,
            coverage=0.25,
            completeness=0.20,
            status="validation_failed",
            rejection_reason=heading_reason,
        )

    # Step 2: If an LLM client is provided, attempt LLM-based structured validation
    if llm_client is not None:
        try:
            from app.agent.prompts import build_validation_prompt
            prompt = build_validation_prompt(question, clean_ans, extract_evidence_text(actual_evidence))
            raw_output = llm_client.generate(prompt)
            cleaned_json = extract_json_from_text(raw_output)
            data = json.loads(cleaned_json)
            if "valid" in data and "groundedness" in data:
                return ValidationResult(
                    valid=bool(data["valid"]),
                    groundedness=float(data["groundedness"]),
                    unsupported_claims=data.get("unsupported_claims", []),
                    supported_claims=data.get("supported_claims", []),
                    relevance=float(data.get("relevance", 0.90)),
                    coverage=float(data.get("coverage", 0.90)),
                    completeness=float(data.get("completeness", 0.90)),
                    status="answered" if data.get("valid") else "validation_failed",
                    rejection_reason=data.get("rejection_reason"),
                )
        except Exception:
            pass  # Fallback to deterministic verification

    # Step 3: Deterministic Groundedness Calculation
    evidence_corpus = extract_evidence_text(actual_evidence)
    claims = split_into_verifiable_claims(clean_ans)
    q_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", question.lower())) - STOP_WORDS

    supported = []
    unsupported = []

    for claim in claims:
        claim_clean = claim.strip()
        claim_lower = claim_clean.lower()

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

        content_tokens = [
            w for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", claim_lower)
            if w not in STOP_WORDS
        ]
        if not content_tokens:
            supported.append(claim_clean)
            continue

        novel_tokens = [
            t for t in content_tokens
            if not any(t.startswith(qt[:4]) or qt.startswith(t[:4]) for qt in q_tokens)
        ]
        if novel_tokens:
            novel_matches = [t for t in novel_tokens if t in evidence_corpus]
            if len(novel_matches) / len(novel_tokens) >= 0.50:
                supported.append(claim_clean)
            else:
                unsupported.append(claim_clean)
        else:
            matches = [t for t in content_tokens if t in evidence_corpus]
            if len(matches) / len(content_tokens) >= 0.33 or any(w in evidence_corpus for w in content_tokens):
                supported.append(claim_clean)
            else:
                unsupported.append(claim_clean)

    total = len(supported) + len(unsupported)
    groundedness = round(len(supported) / total, 4) if total > 0 else 1.0

    # Step 4: Evaluate Relevance, Coverage, Completeness
    relevance = compute_answer_relevance(question, clean_ans, actual_plan)
    cov_dict = compute_answer_coverage(question, actual_evidence, actual_plan, clean_ans)
    coverage = cov_dict["coverage_ratio"]
    completeness = compute_answer_completeness(q_type, clean_ans, actual_plan)

    # Step 5: Multi-dimensional validity threshold
    is_valid = (groundedness >= actual_threshold) and (relevance >= 0.70) and (completeness >= 0.60)

    rejection_reason = None
    if not is_valid:
        reasons = []
        if groundedness < actual_threshold:
            reasons.append(f"Groundedness {groundedness:.2f} < {actual_threshold:.2f}")
        if relevance < 0.70:
            reasons.append(f"Relevance {relevance:.2f} < 0.70")
        if completeness < 0.60:
            reasons.append(f"Completeness {completeness:.2f} < 0.60")
        rejection_reason = "; ".join(reasons)

    status = "answered" if is_valid else "validation_failed"

    return ValidationResult(
        valid=is_valid,
        groundedness=groundedness,
        unsupported_claims=unsupported,
        supported_claims=supported,
        relevance=relevance,
        coverage=coverage,
        completeness=completeness,
        status=status,
        rejection_reason=rejection_reason,
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

