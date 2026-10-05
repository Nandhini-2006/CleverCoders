"""
Evidence Management and Contradiction Handling (Phase 5).

Transforms retrieved page text from Phase 4 into structured, verified evidence
with deterministic contradiction detection, superseding statement resolution,
and confidence calculation.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# Explicit superseding markers required to resolve a contradiction
SUPERSEDING_PATTERNS = [
    r"\breplaced\b",
    r"\breplaced\s+by\b",
    r"\bsupersedes\b",
    r"\bsuperseded\b",
    r"\bdeprecated\b",
    r"\bupdated\b",
    r"\brevised\b",
    r"\bprevious\s+version\b",
    r"\beffective\s+from\b",
    r"\beffective\s+date\b",
    r"\beffective\s+[A-Za-z]+(?:\s+\d+)?,?\s+\d{4}\b",
    r"\bnew\s+policy\b",
    r"\bversion\s+\d+\b",
    r"\bas\s+of\s+[A-Za-z]+\s+\d+\b",
]

# Topic clustering vocabulary for deterministic matching
KNOWN_TOPICS = {
    "authentication": ["oauth", "saml", "sso", "authentication", "login", "identity", "jwt", "password", "ldap"],
    "file_size": ["file size", "upload size", "maximum file", "max file", "file limit", "attachment size", "10 mb", "20 mb"],
    "timeout": ["timeout", "expire", "expiration", "session timeout", "duration", "seconds", "minutes"],
    "encryption": ["encryption", "aes", "rsa", "tls", "ssl", "cipher", "crypto"],
    "rate_limit": ["rate limit", "requests per", "throttle", "requests/sec", "max requests"],
}


def is_superseding_statement(text: str) -> bool:
    """
    Check if a statement contains explicit revision or superseding language.
    Page number alone must NEVER be used to determine if a statement supersedes another.
    """
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in SUPERSEDING_PATTERNS)


def detect_topic_and_value(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a normalized topic and asserted value from a factual statement.
    """
    text_clean = text.strip()
    text_lower = text_clean.lower()

    # 1. Metric / numerical limit pattern (e.g., "Maximum file size is 10 MB", "Timeout is 30 seconds")
    metric_match = re.search(
        r"(?:maximum|max|minimum|min|default|total)?\s*([a-zA-Z\s]{3,25}?)\s*(?:is|of|set to|:|=)\s*(\d+(?:\.\d+)?\s*(?:mb|gb|kb|seconds|minutes|hours|days|%|users|px|ms)?\b)",
        text_clean,
        re.IGNORECASE,
    )
    if metric_match:
        raw_topic = metric_match.group(1).strip().lower()
        val = metric_match.group(2).strip().lower()
        # Normalize topic
        for topic_key, keywords in KNOWN_TOPICS.items():
            if any(k in raw_topic for k in keywords):
                return topic_key, val
        return raw_topic, val

    # 2. Replaced by pattern (e.g., "OAuth is replaced by SAML")
    replace_match = re.search(
        r"([A-Za-z0-9_-]+)\s+(?:is\s+)?replaced by\s+([A-Za-z0-9_-]+)",
        text_clean,
        re.IGNORECASE,
    )
    if replace_match:
        old_val = replace_match.group(1).strip().lower()
        new_val = replace_match.group(2).strip().lower()
        for topic_key, keywords in KNOWN_TOPICS.items():
            if any(k in [old_val, new_val] for k in keywords):
                return topic_key, new_val
        return f"{old_val}_mechanism", new_val

    # 3. Known domain topics (authentication, encryption, file_size, timeout)
    for topic_key, keywords in KNOWN_TOPICS.items():
        matched = [k for k in keywords if k in text_lower]
        if matched:
            return topic_key, matched[0]

    # 4. System / technology assertion fallback (e.g., "The system uses CustomTech")
    system_match = re.search(
        r"(?:the\s+)?(system|application|service|platform)?\s*(?:uses|supports|implements|requires|is based on)\s+([A-Za-z0-9_-]+)",
        text_clean,
        re.IGNORECASE,
    )
    if system_match:
        subject = (system_match.group(1) or "system").lower()
        tech = system_match.group(2).strip().lower()
        for topic_key, keywords in KNOWN_TOPICS.items():
            if any(k in tech for k in keywords):
                return topic_key, tech
        return f"{subject}_mechanism", tech

    # 5. Generic predicate assertion (e.g., "Project deadline is set to October 15, 2026")
    generic_assert = re.search(
        r"^(?:the\s+)?([A-Za-z0-9_\s]{3,30}?)\s+(?:is\s+set\s+to|is\s+scheduled\s+for|is|was|are|were|set\s+to|deadline\s+is)\s+(.+)$",
        text_clean,
        re.IGNORECASE,
    )
    if generic_assert:
        subj = generic_assert.group(1).strip().lower()
        val = generic_assert.group(2).strip().rstrip(".").lower()
        words = subj.split()
        if len(words) <= 4 and words[0] not in ["it", "this", "that", "there", "what", "which", "who", "how"]:
            return subj, val

    return None, None


class EvidenceManager:
    """
    Phase 5 Evidence Manager:
    Extracts claims, groups supporting evidence, detects contradictions,
    resolves superseding statements, and computes confidence.
    
    Security: All document text is treated as passive, untrusted data.
    Budget: Performs zero tool calls.
    """
    def __init__(
        self,
        w1: float = 0.4,
        w2: float = 0.2,
        w3: float = 0.3,
        w4: float = 0.5,
        tau: float = 0.7,
    ):
        self.w1 = w1  # Evidence support strength weight
        self.w2 = w2  # Verification / agreement strength weight
        self.w3 = w3  # Source relevance strength weight
        self.w4 = w4  # Contradiction penalty weight
        self.tau = tau  # Confidence threshold

    def process(
        self,
        retrieval_data: Union[List[Dict[str, Any]], Dict[str, Any]],
        query_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process retrieved pages into a structured evidence state.

        Args:
            retrieval_data: Either a list of retrieved page dicts or the full Phase 4 result dict.
            query_plan: Optional query plan from Phase 3.

        Returns:
            Dict[str, Any]: Structured Evidence State.
        """
        # 1. Normalize input pages
        pages: List[Dict[str, Any]] = []
        if isinstance(retrieval_data, dict):
            pages = retrieval_data.get("retrieved_pages", [])
        elif isinstance(retrieval_data, list):
            pages = retrieval_data

        if not pages:
            return {
                "claims": [],
                "contradictions": [],
                "unresolved_contradictions": [],
                "confidence": 0.0,
                "status": "insufficient_information",
                "evidence_strength": 0.0,
            }

        # 2. Extract raw statements with source page tracking
        raw_items: List[Dict[str, Any]] = []
        for p in pages:
            page_num = p.get("page_number", p.get("page", 1))
            text = str(p.get("text", "")).strip()
            relevance = float(p.get("relevance", p.get("score", 1.0)))

            if not text:
                continue

            # Split sentences deterministically
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 8]
            for s in sentences:
                topic, val = detect_topic_and_value(s)
                raw_items.append({
                    "text": s,
                    "page_number": page_num,
                    "relevance": relevance,
                    "topic": topic,
                    "value": val,
                    "is_superseding": is_superseding_statement(s),
                })

        if not raw_items:
            return {
                "claims": [],
                "contradictions": [],
                "unresolved_contradictions": [],
                "confidence": 0.0,
                "status": "insufficient_information",
                "evidence_strength": 0.0,
            }

        # 3. Group by topic to analyze agreement and contradictions
        topic_groups: Dict[str, List[Dict[str, Any]]] = {}
        ungrouped_items: List[Dict[str, Any]] = []

        for item in raw_items:
            if item["topic"]:
                topic_groups.setdefault(item["topic"], []).append(item)
            else:
                ungrouped_items.append(item)

        claims_list: List[Dict[str, Any]] = []
        contradictions_list: List[Dict[str, Any]] = []
        unresolved_list: List[Dict[str, Any]] = []

        for topic, group in topic_groups.items():
            # Check unique values within topic
            values_map: Dict[str, List[Dict[str, Any]]] = {}
            for item in group:
                val_key = item["value"] or item["text"]
                values_map.setdefault(val_key, []).append(item)

            if len(values_map) <= 1:
                # All statements agree or support the same value
                all_pages = sorted(list({item["page_number"] for item in group}))
                rep_text = group[0]["text"]
                claims_list.append({
                    "claim": rep_text,
                    "topic": topic,
                    "supporting_pages": all_pages,
                    "confidence": 0.85 if len(all_pages) > 1 else 0.70,
                })
            else:
                # Contradiction detected across different values!
                c_claims = [
                    {"text": item["text"], "page": item["page_number"]}
                    for item in group
                ]

                # Check for explicit superseding statement
                superseding_candidates = [item for item in group if item["is_superseding"]]

                if len(superseding_candidates) == 1:
                    # Exactly one statement has explicit superseding marker
                    winner = superseding_candidates[0]
                    contra_entry = {
                        "topic": topic,
                        "claims": c_claims,
                        "status": "superseded",
                        "winning_page": winner["page_number"],
                    }
                    contradictions_list.append(contra_entry)

                    # Add the winning superseding statement to valid claims
                    claims_list.append({
                        "claim": winner["text"],
                        "topic": topic,
                        "supporting_pages": [winner["page_number"]],
                        "confidence": 0.80,
                    })
                else:
                    # Either no superseding markers OR multiple ambiguous markers:
                    # Page number alone must NOT determine which claim wins!
                    contra_entry = {
                        "topic": topic,
                        "claims": c_claims,
                        "status": "unresolved",
                        "winning_page": None,
                    }
                    contradictions_list.append(contra_entry)
                    unresolved_list.append(contra_entry)

        # 4. Handle ungrouped items as individual claims
        for item in ungrouped_items:
            claims_list.append({
                "claim": item["text"],
                "topic": "general",
                "supporting_pages": [item["page_number"]],
                "confidence": 0.65,
            })

        # 5. Compute Confidence Formulation: Conf = w1*E + w2*V + w3*S - w4*X
        # Keyword alignment with query plan
        keywords = query_plan.get("keywords", []) if query_plan else []
        if keywords and claims_list:
            matched_kws = sum(
                1 for kw in keywords
                if any(kw.lower() in cl["claim"].lower() for cl in claims_list)
            )
            kw_ratio = matched_kws / len(keywords)
        else:
            kw_ratio = 0.5 if not keywords else 0.0

        # E: Evidence support strength [0, 1]
        if not claims_list:
            E = 0.0
        else:
            base_e = 0.4 + 0.3 * len(claims_list)
            if query_plan and kw_ratio > 0:
                base_e += 0.3 * kw_ratio
            E = min(1.0, base_e)

        # V: Verification / agreement strength [0, 1]
        multi_page_count = sum(1 for c in claims_list if len(c.get("supporting_pages", [])) > 1)
        if len(pages) > 1:
            base_v = (multi_page_count / len(claims_list)) if claims_list else 0.0
            if not unresolved_list and kw_ratio >= 0.5:
                base_v = max(base_v, 0.5 * kw_ratio)
            V = min(1.0, base_v)
        else:
            # Single authoritative page without contradictions
            if query_plan:
                V = 0.6 * kw_ratio
            else:
                V = 0.4

        # S: Source relevance strength [0, 1]
        relevance_vals = [float(p.get("relevance", 1.0)) for p in pages]
        avg_rel = sum(relevance_vals) / len(relevance_vals) if relevance_vals else 1.0
        S = min(1.0, max(0.0, avg_rel))



        # X: Contradiction penalty [0, 1]
        if unresolved_list:
            X = 1.0  # Full penalty for unresolved contradictions
        elif contradictions_list:
            X = 0.2  # Minor penalty if all contradictions were superseded
        else:
            X = 0.0  # No contradictions

        # Calculate final confidence score
        raw_conf = (self.w1 * E) + (self.w2 * V) + (self.w3 * S) - (self.w4 * X)
        confidence = round(max(0.0, min(1.0, raw_conf)), 4)
        evidence_strength = round(E, 4)

        # Status determination
        if confidence >= self.tau and not unresolved_list and claims_list:
            status = "sufficient"
        else:
            status = "insufficient_information"

        # 6. Compute Answer Coverage across requested entities/concepts
        from app.agent.validator import compute_answer_coverage
        coverage_data = compute_answer_coverage(
            question=query_plan.get("question", "") if (query_plan and isinstance(query_plan, dict)) else "",
            evidence=pages,
            query_plan=query_plan
        )

        return {
            "claims": claims_list,
            "contradictions": contradictions_list,
            "unresolved_contradictions": unresolved_list,
            "confidence": confidence,
            "status": status,
            "evidence_strength": evidence_strength,
            "answer_coverage": coverage_data,
        }


def process_evidence(
    retrieval_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    query_plan: Optional[Dict[str, Any]] = None,
    w1: float = 0.4,
    w2: float = 0.2,
    w3: float = 0.3,
    w4: float = 0.5,
    tau: float = 0.7,
) -> Dict[str, Any]:
    """
    Functional interface to process retrieved evidence.
    """
    manager = EvidenceManager(w1=w1, w2=w2, w3=w3, w4=w4, tau=tau)
    return manager.process(retrieval_data, query_plan=query_plan)
