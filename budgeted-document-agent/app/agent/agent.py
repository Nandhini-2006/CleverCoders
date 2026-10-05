import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from app.config import MAX_TOOL_CALLS
from app.harness.budget import BudgetController, BudgetExhaustedError
from app.harness.state import AgentState
from app.agent.planner import plan_query
from app.agent.evidence import process_evidence
from app.tools.search_keyword import search_keyword
from app.tools.get_page import get_page


def compute_entropy(candidate_pages: List[Dict[str, Any]]) -> float:
    """
    Calculate Shannon entropy H(P) over candidate pages based on probability distribution.
    """
    if not candidate_pages:
        return 0.0

    total_score = sum(c.get("score", 1) for c in candidate_pages)
    if total_score <= 0:
        return 0.0

    entropy = 0.0
    for c in candidate_pages:
        prob = c.get("score", 1) / total_score
        if prob > 0:
            entropy -= prob * math.log2(prob)

    return round(entropy, 4)


class DocumentAgent:
    """
    Phase 6 Agent Loop with Information-Gain Based Action Selection.
    Enforces the deterministic constraint: sum(Cost(a_t)) <= 6.
    Every tool call passes through the BudgetController.
    """
    def __init__(
        self,
        max_budget: int = MAX_TOOL_CALLS,
        tau: float = 0.7,
        search_fn: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
        get_page_fn: Optional[Callable[[str, int], Any]] = None,
        planner_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        evidence_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.max_budget = max_budget
        self.tau = tau
        self.search_fn = search_fn or search_keyword
        self.get_page_fn = get_page_fn or get_page
        self.planner_fn = planner_fn or plan_query
        self.evidence_fn = evidence_fn or process_evidence

    def run(self, doc_id: str, question: str) -> AgentState:
        """
        Run the budgeted agent loop to collect evidence for a user question.

        Hard constraint: Total document-tool calls <= 6.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # 1. Initialize Budget Controller & State
        budget = BudgetController(max_calls=self.max_budget)

        # 2. Run Phase 3 Query Planner (does not consume document tools)
        plan = self.planner_fn(question.strip())
        keywords: List[str] = plan.get("keywords", [])

        state = AgentState(
            question=question.strip(),
            query_plan=plan,
            candidate_pages=[],
            retrieved_pages=[],
            evidence={},
            entropy=1.5 if keywords else 0.0,
            confidence=0.0,
            tool_calls_used=0,
            remaining_budget=budget.remaining_calls(),
            tool_trace=[],
            status="running",
        )

        searched_keywords: Set[str] = set()
        retrieved_page_numbers: Set[int] = set()
        candidate_map: Dict[int, Dict[str, Any]] = {}

        # 3. Iterative Information-Gain Agent Loop
        while budget.remaining_calls() > 0:
            # Check stopping condition: sufficient confidence and no unresolved contradictions
            if (
                state["confidence"] >= self.tau
                and state["evidence"]
                and not state["evidence"].get("unresolved_contradictions")
            ):
                state["status"] = "completed"
                break

            # --- Action Generation & Expected Information Gain Scoring ---
            candidate_actions: List[Tuple[str, Any, float]] = []

            # Option A: Search keyword action (if not searched yet)
            for kw in keywords:
                if kw not in searched_keywords:
                    # High expected value if no candidates yet, or moderate if adding coverage
                    expected_ig = 1.0 if not candidate_map else 0.65
                    candidate_actions.append(("search_keyword", kw, expected_ig))

            # Option B: Get page action (if discovered and not retrieved yet)
            for page_num, cand in candidate_map.items():
                if page_num not in retrieved_page_numbers:
                    prob = cand.get("probability", 0.5)
                    # Bonus if this page is involved in an unresolved contradiction
                    contra_bonus = 0.0
                    unresolved = state["evidence"].get("unresolved_contradictions", [])
                    for contra in unresolved:
                        if any(c["page"] == page_num for c in contra.get("claims", [])):
                            contra_bonus = 0.4
                            break

                    expected_ig = prob + contra_bonus
                    candidate_actions.append(("get_page", page_num, expected_ig))

            # Stop condition: No candidate actions available
            if not candidate_actions:
                break

            # Pick action with highest expected information score
            candidate_actions.sort(key=lambda a: -a[2])
            best_action_type, best_action_param, _ = candidate_actions[0]

            # Stop condition: Expected gain too low
            if candidate_actions[0][2] <= 0.0:
                break

            entropy_before = state["entropy"]

            # --- Action Execution via BudgetController ---
            if best_action_type == "search_keyword":
                kw = best_action_param
                searched_keywords.add(kw)
                call_num = budget.calls_used() + 1

                try:
                    matches = budget.execute(self.search_fn, doc_id, kw)
                    status_str = "success"
                except Exception as e:
                    matches = []
                    status_str = f"failed: {e}"

                # Update candidate pages
                for m in (matches or []):
                    p_num = m.get("page_number", m.get("page")) if isinstance(m, dict) else getattr(m, "page_number", getattr(m, "page", None))
                    if isinstance(p_num, int) and p_num > 0:
                        if p_num not in candidate_map:
                            candidate_map[p_num] = {
                                "page_number": p_num,
                                "matched_keywords": [kw],
                                "score": 1,
                            }
                        else:
                            if kw not in candidate_map[p_num]["matched_keywords"]:
                                candidate_map[p_num]["matched_keywords"].append(kw)
                                candidate_map[p_num]["score"] += 1

                # Recompute probabilities and candidate list
                candidates_list = list(candidate_map.values())
                total_relevance = sum(c["score"] for c in candidates_list)
                for c in candidates_list:
                    c["probability"] = round(c["score"] / total_relevance, 4) if total_relevance > 0 else 0.0

                candidates_list.sort(key=lambda c: (-c["score"], c["page_number"]))
                state["candidate_pages"] = candidates_list

                # Update entropy
                raw_cand_entropy = compute_entropy(candidates_list)
                entropy_after = round((1.0 - state["confidence"]) * raw_cand_entropy, 4)
                actual_ig = round(entropy_before - entropy_after, 4)

                state["entropy"] = entropy_after
                state["tool_calls_used"] = budget.calls_used()
                state["remaining_budget"] = budget.remaining_calls()

                # Record tool trace
                result_pages = [m.get("page_number", m.get("page")) for m in (matches or [])]
                state["tool_trace"].append({
                    "call_number": call_num,
                    "tool": "search_keyword",
                    "input": {"keyword": kw},
                    "result_summary": f"Found on page(s): {result_pages}",
                    "entropy_before": entropy_before,
                    "entropy_after": entropy_after,
                    "information_gain": actual_ig,
                    "calls_used": budget.calls_used(),
                    "remaining_budget": budget.remaining_calls(),
                    "status": status_str,
                })

            elif best_action_type == "get_page":
                page_num = best_action_param
                retrieved_page_numbers.add(page_num)
                call_num = budget.calls_used() + 1

                try:
                    page_res = budget.execute(self.get_page_fn, doc_id, page_num)
                    text = page_res.get("text", "") if isinstance(page_res, dict) else str(page_res)
                    status_str = "success"
                except Exception as e:
                    text = ""
                    status_str = f"failed: {e}"

                # Append retrieved page
                state["retrieved_pages"].append({
                    "page_number": page_num,
                    "text": text,
                    "relevance": candidate_map.get(page_num, {}).get("score", 1.0),
                })

                # Run Phase 5 Evidence Manager
                evidence_res = self.evidence_fn(state["retrieved_pages"], query_plan=plan)
                state["evidence"] = evidence_res
                state["confidence"] = evidence_res.get("confidence", 0.0)

                # Recompute entropy (as confidence increases, residual entropy decreases)
                raw_cand_entropy = compute_entropy(state["candidate_pages"])
                entropy_after = round((1.0 - state["confidence"]) * raw_cand_entropy, 4)
                actual_ig = round(entropy_before - entropy_after, 4)

                state["entropy"] = entropy_after
                state["tool_calls_used"] = budget.calls_used()
                state["remaining_budget"] = budget.remaining_calls()

                # Record tool trace
                state["tool_trace"].append({
                    "call_number": call_num,
                    "tool": "get_page",
                    "input": {"page_number": page_num},
                    "result_summary": f"Retrieved {len(text)} chars",
                    "entropy_before": entropy_before,
                    "entropy_after": entropy_after,
                    "information_gain": actual_ig,
                    "calls_used": budget.calls_used(),
                    "remaining_budget": budget.remaining_calls(),
                    "status": status_str,
                })

        # --- Final State & Status Evaluation ---
        state["tool_calls_used"] = budget.calls_used()
        state["remaining_budget"] = budget.remaining_calls()

        # If evidence was collected, check if sufficient or insufficient
        if not state["evidence"] and state["retrieved_pages"]:
            state["evidence"] = self.evidence_fn(state["retrieved_pages"], query_plan=plan)
            state["confidence"] = state["evidence"].get("confidence", 0.0)

        unresolved = state["evidence"].get("unresolved_contradictions", [])
        if state["confidence"] >= self.tau and not unresolved and state["retrieved_pages"]:
            state["status"] = "completed"
        else:
            state["status"] = "insufficient_information"

        return state


def run_agent(
    doc_id: str,
    question: str,
    max_budget: int = MAX_TOOL_CALLS,
    tau: float = 0.7,
    search_fn: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
    get_page_fn: Optional[Callable[[str, int], Any]] = None,
    planner_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    evidence_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> AgentState:
    """
    Functional interface to run the DocumentAgent loop.
    """
    agent = DocumentAgent(
        max_budget=max_budget,
        tau=tau,
        search_fn=search_fn,
        get_page_fn=get_page_fn,
        planner_fn=planner_fn,
        evidence_fn=evidence_fn,
    )
    return agent.run(doc_id=doc_id, question=question)
