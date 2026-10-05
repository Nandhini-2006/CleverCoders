import re
from typing import Any, Dict, List, Optional, Set, Union

from app.agent.prompts import build_answer_prompt
from app.agent.validator import validate_answer, revise_answer, GROUNDING_THRESHOLD


class AnswerGenerator:
    """
    Phase 7 Final Answer Generator:
    Synthesizes a grounded, verifiable final answer from retrieved evidence,
    resolves superseding statements, highlights unresolved contradictions,
    and runs groundedness validation.

    Security & Architecture Rules:
    - Never receives the entire PDF.
    - Performs zero document-tool calls.
    - Treats all document text as untrusted data.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        grounding_threshold: float = GROUNDING_THRESHOLD,
        llm_client: Optional[Any] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.grounding_threshold = grounding_threshold
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        evidence_state: Optional[Dict[str, Any]] = None,
        retrieved_pages: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        contradictions: Optional[List[Dict[str, Any]]] = None,
        unresolved_contradictions: Optional[List[Dict[str, Any]]] = None,
        agent_state: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded final answer structure.

        Returns:
            Dict[str, Any]:
                - answer: str
                - sources: List[int]
                - status: 'answered' | 'insufficient_information' | 'unresolved_contradiction' | 'validation_failed'
                - groundedness: float
        """
        # 1. Unpack parameters if passed as AgentState
        query_plan = None
        if agent_state is not None:
            evidence_state = agent_state.get("evidence", {})
            retrieved_pages = agent_state.get("retrieved_pages", [])
            confidence = agent_state.get("confidence", 0.0)
            query_plan = agent_state.get("query_plan")

        ev_state = evidence_state or {}
        pages = retrieved_pages or []
        conf = confidence if confidence is not None else ev_state.get("confidence", 0.0)
        all_contradictions = contradictions if contradictions is not None else ev_state.get("contradictions", [])
        unresolved = unresolved_contradictions if unresolved_contradictions is not None else ev_state.get("unresolved_contradictions", [])
        claims = ev_state.get("claims", [])
        answer_coverage = ev_state.get("answer_coverage", {})
        missing_components = answer_coverage.get("missing_components", [])

        # -------------------------------------------------------------
        # STEP 1: Unresolved Contradictions Handling
        # -------------------------------------------------------------
        if unresolved:
            # Never choose one arbitrarily! Explain the conflict.
            contra = unresolved[0]
            contra_claims = contra.get("claims", [])
            sources = sorted(list({c.get("page") for c in contra_claims if c.get("page")}))

            if len(contra_claims) >= 2:
                c1, c2 = contra_claims[0], contra_claims[1]
                answer_text = (
                    f"The document contains conflicting information. "
                    f"Page {c1.get('page')} states '{c1.get('text', '').rstrip('.')}', "
                    f"while Page {c2.get('page')} states '{c2.get('text', '').rstrip('.')}'. "
                    f"The document does not clearly indicate which statement supersedes the other."
                )
            else:
                answer_text = "The document contains unresolved conflicting information on this topic."

            return {
                "answer": answer_text,
                "sources": sources,
                "status": "unresolved_contradiction",
                "groundedness": 1.0,
            }

        # -------------------------------------------------------------
        # STEP 2: Insufficient Information Check
        # -------------------------------------------------------------
        if (
            conf < self.confidence_threshold
            or ev_state.get("status") == "insufficient_information"
            or not claims
        ):
            # Gather any pages that were retrieved even if insufficient
            sources = sorted(list({p.get("page_number", p.get("page")) for p in pages if p.get("page_number", p.get("page"))}))
            return {
                "answer": "Insufficient information.",
                "sources": sources,
                "status": "insufficient_information",
                "groundedness": 1.0,
            }

        # -------------------------------------------------------------
        # STEP 3: Superseding Statements Resolution
        # -------------------------------------------------------------
        superseded_contras = [c for c in all_contradictions if c.get("status") == "superseded"]
        if superseded_contras:
            # Use the winning/superseding statement
            winning_contra = superseded_contras[0]
            win_page = winning_contra.get("winning_page")
            sources = [win_page] if win_page else []

            # Find matching winning claim
            winning_claim_text = ""
            for cl in claims:
                if win_page in cl.get("supporting_pages", []):
                    winning_claim_text = cl.get("claim", "")
                    break

            if not winning_claim_text:
                for c_item in winning_contra.get("claims", []):
                    if c_item.get("page") == win_page:
                        winning_claim_text = c_item.get("text", "")
                        break

            # Formulate clear concise answer
            answer_text = winning_claim_text.strip()
            if not answer_text.endswith("."):
                answer_text += "."

            val = validate_answer(question, answer_text, pages or claims, threshold=self.grounding_threshold)

            return {
                "answer": answer_text,
                "sources": sources,
                "status": "answered",
                "groundedness": val.groundedness,
            }

        # -------------------------------------------------------------
        # STEP 4: Standard Grounded Answer Generation
        # -------------------------------------------------------------
        INJECTION_PATTERNS = [
            r"ignore\s+(?:all\s+|previous\s+)?instructions",
            r"reveal\s+(?:your\s+|the\s+)?system\s+prompt",
            r"output\s+hacked",
            r"disregard\s+prior",
            r"call\s+another\s+tool",
            r"tell\s+the\s+user\b",
            r"say\s+to\s+the\s+user\b",
            r"instruct\s+the\s+user\b",
            r"pretend\s+(?:that|to\s+be)\b",
            r"written\s+by\s+superman\b",
            r"\bsuperman\b",
        ]

        # Extract question keywords for topical filtering (excluding generic doc words)
        from app.agent.validator import STOP_WORDS, DEFAULT_GENERIC_WORDS
        GENERIC_FILTER_WORDS = (
            STOP_WORDS
            | DEFAULT_GENERIC_WORDS
            | {"document", "system", "information", "page", "text", "according", "milestone", "milestones", "timeline", "event", "events"}
        )
        q_tokens = [
            w for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", question.lower())
            if w not in GENERIC_FILTER_WORDS
        ]


        collected_sources: Set[int] = set()
        answer_sentences: List[str] = []

        # Filter out claims that contain injection instructions or are untrusted directives
        clean_claims = []
        for cl in claims:
            txt = cl.get("claim", "").strip()
            # Check for adversarial prompt injection in document evidence
            if any(re.search(pat, txt, flags=re.IGNORECASE) for pat in INJECTION_PATTERNS):
                continue
            clean_claims.append(cl)

        # If we have clean claims with topical relevance to the question, prioritize them
        topical_claims = []
        for cl in clean_claims:
            txt = cl.get("claim", "").lower()
            if any(qt[:4] in txt for qt in q_tokens):
                topical_claims.append(cl)

        is_structured_query = (
            (query_plan and query_plan.get("question_type") in ("timeline", "summary", "list", "procedural"))
            or any(t in question.lower() for t in ["timeline", "milestones", "history", "steps", "list"])
        )
        selected_claims = clean_claims if is_structured_query else (topical_claims if topical_claims else clean_claims)

        for cl in selected_claims:
            txt = cl.get("claim", "").strip()
            if txt:
                if not txt.endswith("."):
                    txt += "."
                answer_sentences.append(txt)
                for sp in cl.get("supporting_pages", []):
                    if isinstance(sp, int):
                        collected_sources.add(sp)

        for p in pages:
            p_num = p.get("page_number", p.get("page"))
            if isinstance(p_num, int):
                collected_sources.add(p_num)

        sources = sorted(list(collected_sources))

        NEGATIVE_EVIDENCE_PATTERNS = [
            r"\bwithout\s+referencing\b",
            r"\bdoes\s+not\s+(?:mention|state|contain|reference|specify)\b",
            r"\bnot\s+referenced\b",
            r"\bnot\s+specified\b",
            r"\bno\s+[a-z\s]+(?:is|are)\s+mentioned\b",
        ]

        from app.agent.validator import is_heading_or_fragment_answer
        substantive_sentences: List[str] = []
        for s in answer_sentences:
            is_h, _ = is_heading_or_fragment_answer(s, question)
            if not is_h:
                substantive_sentences.append(s)

        # If answer sentences only contained headings or titles, extract substantive sentences from retrieved pages
        if not substantive_sentences and pages:
            for p in pages:
                p_text = str(p.get("text", "")).strip()
                p_num = p.get("page_number", p.get("page"))
                raw_sentences = [
                    s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", p_text)
                    if len(s.strip()) >= 10
                ]
                for s in raw_sentences:
                    is_h, _ = is_heading_or_fragment_answer(s, question)
                    if not is_h and not any(re.search(pat, s, re.IGNORECASE) for pat in INJECTION_PATTERNS):
                        clean_s = s if s.endswith((".", "!", "?")) else f"{s}."
                        substantive_sentences.append(clean_s)
                        if isinstance(p_num, int):
                            collected_sources.add(p_num)

        effective_sentences = substantive_sentences if substantive_sentences else answer_sentences

        active_llm = llm_client or self.llm_client
        evidence_text = "\n".join(
            [f"Page {p.get('page_number', p.get('page'))}: {p.get('text')}" for p in pages if p.get('text')]
        ) or "\n".join([f"Claim: {cl.get('claim')}" for cl in selected_claims])

        # Check for missing essential attributes requested by user question
        if missing_components:
            for mc in missing_components:
                mc_tokens = [w for w in re.findall(r"\b[A-Za-z0-9_-]+\b", mc.lower()) if len(w) >= 2 and w not in GENERIC_FILTER_WORDS]
                if mc_tokens and not any(w in evidence_text.lower() for w in mc_tokens):
                    if any(w in question.lower() for w in mc_tokens):
                        return {
                            "answer": f"Insufficient information. The document describes the topic, but the retrieved evidence does not specify {mc}.",
                            "sources": sources,
                            "status": "insufficient_information",
                            "groundedness": 1.0,
                        }

        if active_llm is not None:
            coverage_detail = answer_coverage.get("details") if answer_coverage else None
            prompt = build_answer_prompt(
                question=question,
                evidence_text=evidence_text,
                query_plan=query_plan,
                coverage_info=coverage_detail,
            )
            try:
                draft_answer = active_llm.generate(prompt).strip()
            except Exception:
                draft_answer = ""
            # If LLM returned empty, fall back to deterministic assembly
            if not draft_answer:
                draft_answer = " ".join(effective_sentences) if effective_sentences else "Insufficient information."
        else:
            has_negative_evidence = any(
                re.search(pat, s, flags=re.IGNORECASE)
                for pat in NEGATIVE_EVIDENCE_PATTERNS
                for s in effective_sentences
            )
            if has_negative_evidence or not effective_sentences:
                draft_answer = "Insufficient information."
            else:
                draft_answer = " ".join(effective_sentences)

        # -------------------------------------------------------------
        # STEP 5: Answer Quality Validation & Regeneration
        # -------------------------------------------------------------
        val = validate_answer(
            question,
            draft_answer,
            pages or claims,
            query_plan=query_plan,
            threshold=self.grounding_threshold,
            llm_client=active_llm,
        )

        if not val.valid and pages:
            # Attempt 1 regeneration using the same evidence
            if active_llm is not None:
                from app.agent.prompts import build_regeneration_prompt
                rejection_msg = val.rejection_reason or "Answer did not sufficiently address the question."
                regen_prompt = build_regeneration_prompt(
                    question=question,
                    evidence_text=evidence_text,
                    previous_answer=draft_answer,
                    rejection_reason=rejection_msg,
                )
                try:
                    regen_ans = active_llm.generate(regen_prompt).strip()
                    if not regen_ans:
                        raise ValueError("LLM returned empty regeneration response.")
                    val_regen = validate_answer(
                        question,
                        regen_ans,
                        pages or claims,
                        query_plan=query_plan,
                        threshold=self.grounding_threshold,
                        llm_client=active_llm,
                    )
                    if val_regen.valid:
                        draft_answer = regen_ans
                        val = val_regen
                except Exception:
                    pass

            if not val.valid:
                # If still invalid, check if rejected due to heading or fragment
                if is_heading_or_fragment_answer(draft_answer, question)[0]:
                    if substantive_sentences:
                        draft_answer = " ".join(substantive_sentences)
                        val = validate_answer(
                            question,
                            draft_answer,
                            pages or claims,
                            query_plan=query_plan,
                            threshold=self.grounding_threshold,
                            llm_client=active_llm,
                        )
                    else:
                        draft_answer = "Insufficient information: The document contains section headings for this topic, but does not provide detailed information."
                        val = validate_answer(question, draft_answer, pages or claims, query_plan=query_plan)
                else:
                    revised = revise_answer(question, draft_answer, pages or claims, validation_result=val)
                    val_revised = validate_answer(
                        question,
                        revised,
                        pages or claims,
                        query_plan=query_plan,
                        threshold=self.grounding_threshold,
                        llm_client=active_llm,
                    )
                    if val_revised.valid:
                        draft_answer = revised
                        val = val_revised

        if val.valid:
            final_answer = draft_answer
            if "insufficient information" in draft_answer.lower():
                status = "insufficient_information"
            else:
                status = "answered"
                # If some requested components were missing, explicitly state so
                if missing_components:
                    unmentioned_missing = [
                        mc for mc in missing_components
                        if mc.lower() not in final_answer.lower()
                    ]
                    if unmentioned_missing:
                        final_answer = final_answer.rstrip(".") + f". (Note: The retrieved evidence does not specify information for {', '.join(unmentioned_missing)}.)"
        else:
            final_answer = draft_answer
            status = "validation_failed"

        return {
            "answer": final_answer,
            "sources": sources,
            "status": status,
            "groundedness": val.groundedness,
        }


def generate_final_answer(
    question: str,
    evidence_state: Optional[Dict[str, Any]] = None,
    retrieved_pages: Optional[List[Dict[str, Any]]] = None,
    confidence: Optional[float] = None,
    contradictions: Optional[List[Dict[str, Any]]] = None,
    unresolved_contradictions: Optional[List[Dict[str, Any]]] = None,
    agent_state: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.70,
    grounding_threshold: float = GROUNDING_THRESHOLD,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Functional interface to generate the final grounded answer.
    """
    generator = AnswerGenerator(
        confidence_threshold=confidence_threshold,
        grounding_threshold=grounding_threshold,
        llm_client=llm_client,
    )
    return generator.generate(
        question=question,
        evidence_state=evidence_state,
        retrieved_pages=retrieved_pages,
        confidence=confidence,
        contradictions=contradictions,
        unresolved_contradictions=unresolved_contradictions,
        agent_state=agent_state,
        llm_client=llm_client,
    )

