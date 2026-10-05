import math
from typing import Any, Callable, Dict, List, Optional, Union

from app.config import MAX_TOOL_CALLS
from app.harness.budget import BudgetTracker
from app.tools.search_keyword import search_keyword
from app.tools.get_page import get_page


class PageRetriever:
    """
    Phase 4 Page Selection & Retrieval engine.
    
    Discovers candidate pages using search_keyword(), ranks them deterministically
    by keyword match score and page number, and retrieves page contents using get_page()
    while strictly enforcing a maximum tool-call budget.
    """
    def __init__(
        self,
        max_budget: int = MAX_TOOL_CALLS,
        search_fn: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
        get_page_fn: Optional[Callable[[str, int], Any]] = None,
    ):
        self.max_budget = max_budget
        self.search_fn = search_fn or search_keyword
        self.get_page_fn = get_page_fn or get_page

    def retrieve(
        self,
        doc_id: str,
        query_plan: Union[Dict[str, Any], Any],
        max_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute candidate page discovery, scoring, ranking, and budget-constrained retrieval.

        Args:
            doc_id: Unique document identifier.
            query_plan: Query plan dictionary containing at most 3 validated 'keywords'.
            max_budget: Optional override for the maximum tool-call budget.

        Returns:
            Dict[str, Any]: Structured result containing:
                - query_plan
                - candidate_pages: list of ranked candidate dicts (page_number, matched_keywords, score)
                - retrieved_pages: list of fetched pages (page_number, text)
                - tool_calls_used: total tool calls consumed
        """
        budget_limit = max_budget if max_budget is not None else self.max_budget
        budget = BudgetTracker(max_calls=budget_limit)

        # Extract keywords safely from query plan
        keywords: List[str] = []
        if isinstance(query_plan, dict):
            raw_kws = query_plan.get("keywords", [])
            if isinstance(raw_kws, list):
                keywords = [str(k).strip() for k in raw_kws if str(k).strip()]

        # -------------------------------------------------------------
        # STEP 1: Candidate Page Discovery (Consumes search_keyword calls)
        # -------------------------------------------------------------
        candidate_map: Dict[int, Dict[str, Any]] = {}

        for kw in keywords:
            if budget.is_exhausted:
                break

            # Consume 1 budget call per keyword search
            budget.consume(1)

            try:
                matches = self.search_fn(doc_id, kw)
            except Exception:
                # Tool failure handled safely - proceed with remaining searches
                matches = []

            if not matches:
                continue

            for match in matches:
                if isinstance(match, dict):
                    page_num = match.get("page_number", match.get("page"))
                else:
                    page_num = getattr(match, "page_number", getattr(match, "page", None))

                if not isinstance(page_num, int) or page_num <= 0:
                    continue

                if page_num not in candidate_map:
                    candidate_map[page_num] = {
                        "page_number": page_num,
                        "matched_keywords": [kw],
                        "score": 1,
                    }
                else:
                    # Deduplicate keywords for this page and update score
                    if kw not in candidate_map[page_num]["matched_keywords"]:
                        candidate_map[page_num]["matched_keywords"].append(kw)
                        candidate_map[page_num]["score"] += 1

        # -------------------------------------------------------------
        # STEP 2: Candidate Relevance R(p,q), Probability P(p|q), and Entropy
        # -------------------------------------------------------------
        candidate_pages = list(candidate_map.values())
        total_relevance = sum(c["score"] for c in candidate_pages)

        for c in candidate_pages:
            c["relevance"] = c["score"]  # R(p, q)
            # Probability P(p | q) = R(p, q) / sum(R(p', q))
            c["probability"] = (
                round(c["score"] / total_relevance, 4) if total_relevance > 0 else 0.0
            )

        # Shannon Entropy H(P) = -sum(P(p|q) * log2(P(p|q)))
        if total_relevance > 0 and len(candidate_pages) > 0:
            entropy = -sum(
                c["probability"] * math.log2(c["probability"])
                for c in candidate_pages
                if c["probability"] > 0
            )
            entropy = round(entropy, 4)
        else:
            entropy = 0.0

        # Sort rule:
        #   1. Higher relevance/probability first (-score)
        #   2. Deterministic tie-breaking by page_number (ascending)
        candidate_pages.sort(key=lambda c: (-c["score"], c["page_number"]))

        # -------------------------------------------------------------
        # STEP 3: Page Retrieval (Consumes get_page calls within budget)
        # -------------------------------------------------------------
        retrieved_pages: List[Dict[str, Any]] = []

        for candidate in candidate_pages:
            if budget.is_exhausted:
                break

            page_num = candidate["page_number"]
            budget.consume(1)

            try:
                page_data = self.get_page_fn(doc_id, page_num)
                if isinstance(page_data, dict):
                    page_text = page_data.get("text", "")
                else:
                    page_text = str(page_data)

                retrieved_pages.append({
                    "page_number": page_num,
                    "text": page_text,
                })
            except Exception:
                # Tool failure handled safely
                continue

        # -------------------------------------------------------------
        # STEP 4: Return Structured Result
        # -------------------------------------------------------------
        return {
            "query_plan": query_plan,
            "candidate_pages": candidate_pages,
            "retrieved_pages": retrieved_pages,
            "tool_calls_used": budget.calls_used,
            "entropy": entropy,
        }


def retrieve_pages(
    doc_id: str,
    query_plan: Union[Dict[str, Any], Any],
    max_budget: int = MAX_TOOL_CALLS,
    search_fn: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
    get_page_fn: Optional[Callable[[str, int], Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience functional interface for Phase 4 Page Selection & Retrieval.
    """
    retriever = PageRetriever(
        max_budget=max_budget,
        search_fn=search_fn,
        get_page_fn=get_page_fn,
    )
    return retriever.retrieve(doc_id=doc_id, query_plan=query_plan)
