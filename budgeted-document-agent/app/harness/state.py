from typing import Any, Dict, List, Optional


class AgentState(dict):
    """
    Per-question temporary runtime state for the Budgeted Document Agent.
    Subclasses dict for JSON-serializability and transparency.
    """
    def __init__(
        self,
        question: str = "",
        query_plan: Optional[Dict[str, Any]] = None,
        candidate_pages: Optional[List[Dict[str, Any]]] = None,
        retrieved_pages: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        entropy: float = 0.0,
        confidence: float = 0.0,
        tool_calls_used: int = 0,
        remaining_budget: int = 6,
        tool_trace: Optional[List[Dict[str, Any]]] = None,
        status: str = "running",
    ):
        super().__init__(
            question=question,
            query_plan=query_plan or {},
            candidate_pages=candidate_pages or [],
            retrieved_pages=retrieved_pages or [],
            evidence=evidence or {},
            entropy=entropy,
            confidence=confidence,
            tool_calls_used=tool_calls_used,
            remaining_budget=remaining_budget,
            tool_trace=tool_trace or [],
            status=status,
        )

    # Property accessors
    @property
    def question(self) -> str:
        return self["question"]

    @property
    def query_plan(self) -> Dict[str, Any]:
        return self["query_plan"]

    @property
    def candidate_pages(self) -> List[Dict[str, Any]]:
        return self["candidate_pages"]

    @property
    def retrieved_pages(self) -> List[Dict[str, Any]]:
        return self["retrieved_pages"]

    @property
    def evidence(self) -> Dict[str, Any]:
        return self["evidence"]

    @property
    def entropy(self) -> float:
        return self["entropy"]

    @property
    def confidence(self) -> float:
        return self["confidence"]

    @property
    def tool_calls_used(self) -> int:
        return self["tool_calls_used"]

    @property
    def remaining_budget(self) -> int:
        return self["remaining_budget"]

    @property
    def tool_trace(self) -> List[Dict[str, Any]]:
        return self["tool_trace"]

    @property
    def status(self) -> str:
        return self["status"]
