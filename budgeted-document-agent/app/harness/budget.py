from typing import Any, Callable, Optional
from app.config import MAX_TOOL_CALLS


class BudgetExhaustedError(RuntimeError):
    """Raised when an attempt is made to execute a tool beyond the maximum budget."""
    pass


class CallableInt(int):
    """
    An integer subclass that is also callable.
    Allows calls_used and remaining_calls to be used either as an attribute or as a method.
    Example: controller.calls_used() or controller.calls_used
    """
    def __call__(self) -> int:
        return int(self)


class BudgetController:
    """
    Deterministic tool-call budget controller enforcing MAX_TOOL_CALLS = 6.
    Every document tool execution must pass through this controller.
    """
    def __init__(self, max_calls: int = MAX_TOOL_CALLS):
        if max_calls < 0:
            raise ValueError("max_calls cannot be negative")
        self.max_calls = max_calls
        self._calls_used = 0

    @property
    def remaining_calls(self) -> CallableInt:
        """Returns the number of remaining tool calls. Never negative."""
        return CallableInt(max(0, self.max_calls - self._calls_used))

    @property
    def calls_used(self) -> CallableInt:
        """Returns the number of tool calls consumed so far."""
        return CallableInt(self._calls_used)

    def can_call(self, count: int = 1) -> bool:
        """Checks if 'count' calls can be made within the remaining budget."""
        return int(self.remaining_calls) >= count

    def consume(self, count: int = 1) -> None:
        """
        Record consumption of 'count' tool calls.
        Raises BudgetExhaustedError if the budget is exceeded.
        """
        if count <= 0:
            return
        if not self.can_call(count):
            raise BudgetExhaustedError(
                f"Budget exhausted. Attempted {count} call(s), but only {int(self.remaining_calls)} remaining."
            )
        self._calls_used += count

    def reset(self) -> None:
        """Reset the tool calls counter to 0."""
        self._calls_used = 0

    def execute(self, tool_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a document tool function while enforcing the budget constraint.
        Consumes exactly 1 call before executing.
        Raises BudgetExhaustedError on call number 7 (or whenever budget is 0).
        """
        self.consume(1)
        return tool_fn(*args, **kwargs)

    # --- Compatibility properties with earlier Phase 4 BudgetTracker ---
    @property
    def remaining(self) -> int:
        return int(self.remaining_calls)

    @property
    def is_exhausted(self) -> bool:
        return int(self.remaining_calls) <= 0


# Backwards compatibility alias for Phase 4 retriever
BudgetTracker = BudgetController
