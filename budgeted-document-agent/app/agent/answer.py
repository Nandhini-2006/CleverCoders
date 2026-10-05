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
        if agent_state is not None:
            evidence_state = agent_state.get("evidence", {})
            retrieved_pages = agent_state.get("retrieved_pages", [])
            confidence = agent_state.get("confidence", 0.0)

        ev_state = evidence_state or {}
        pages = retrieved_pages or []
        conf = confidence if confidence is not None else ev_state.get("confidence", 0.0)
        all_contradictions = contradictions if contradictions is not None else ev_state.get("contradictions", [])
        unresolved = unresolved_contradictions if unresolved_contradictions is not None else ev_state.get("unresolved_contradictions", [])
        claims = ev_state.get("claims", [])

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
            r"tell\s+the\s+user\s+confidential",
        ]

        # Extract question keywords for topical filtering
        from app.agent.validator import STOP_WORDS
        q_tokens = [
            w for w in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", question.lower())
            if w not in STOP_WORDS
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

        selected_claims = topical_claims if topical_claims else clean_claims

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

        active_llm = llm_client or self.llm_client
        if active_llm is not None:
            evidence_text = "\n".join(
                [f"Page {p.get('page_number', p.get('page'))}: {p.get('text')}" for p in pages if p.get('text')]
            ) or "\n".join([f"Claim: {cl.get('claim')}" for cl in selected_claims])
            prompt = build_answer_prompt(question, evidence_text)
            draft_answer = active_llm.generate(prompt).strip()
        else:
            draft_answer = " ".join(answer_sentences) if answer_sentences else "Insufficient information."

        sources = sorted(list(collected_sources))

        # -------------------------------------------------------------
        # STEP 5: Answer Groundedness Validation & Revision
        # -------------------------------------------------------------
        val = validate_answer(
            question,
            draft_answer,
            pages or claims,
            threshold=self.grounding_threshold,
            llm_client=active_llm,
        )

        if val.valid:
            final_answer = draft_answer
            status = "answered"
        else:
            # Attempt revision using only supported claims
            revised = revise_answer(question, draft_answer, pages or claims, validation_result=val)
            val_revised = validate_answer(
                question,
                revised,
                pages or claims,
                threshold=self.grounding_threshold,
                llm_client=active_llm,
            )
            if val_revised.valid:
                final_answer = revised
                if "insufficient information" in revised.lower():
                    status = "insufficient_information"
                else:
                    status = "answered"
                val = val_revised
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

