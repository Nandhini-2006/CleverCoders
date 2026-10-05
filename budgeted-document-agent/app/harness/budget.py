from typing import Optional


class BudgetTracker:
    """
    Deterministic tool-call budget controller.
    Tracks and enforces the maximum number of allowed tool calls per operation.
    """
    def __init__(self, max_calls: int = 6):
        if max_calls < 0:
            raise ValueError("max_calls cannot be negative")
        self.max_calls = max_calls
        self.calls_used = 0

    @property
    def remaining(self) -> int:
        """Number of tool calls remaining in the budget."""
        return max(0, self.max_calls - self.calls_used)

    @property
    def is_exhausted(self) -> bool:
        """True if the budget has been reached or exceeded."""
        return self.remaining <= 0

    def can_consume(self, count: int = 1) -> bool:
        """Check if 'count' tool calls can be made within budget."""
        return self.remaining >= count

    def consume(self, count: int = 1) -> bool:
        """
        Record the consumption of 'count' tool calls.
        Returns True if consumed within budget, False if budget would be exceeded.
        """
        if count <= 0:
            return True
        if self.can_consume(count):
            self.calls_used += count
            return True
        return False
