import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.harness.budget import BudgetController, BudgetExhaustedError


class TestBudgetController(unittest.TestCase):
    # --- Test 1: Budget starts at 6 ---
    def test_budget_starts_at_six(self):
        controller = BudgetController(max_calls=6)
        self.assertEqual(controller.remaining_calls(), 6)
        self.assertEqual(controller.calls_used(), 0)
        self.assertTrue(controller.can_call(1))

    # --- Test 2: One tool call decreases remaining budget to 5 ---
    def test_one_call_decreases_remaining_to_five(self):
        controller = BudgetController(max_calls=6)
        controller.consume(1)
        self.assertEqual(controller.calls_used(), 1)
        self.assertEqual(controller.remaining_calls(), 5)

    # --- Test 3: Six calls are allowed ---
    def test_six_calls_allowed(self):
        controller = BudgetController(max_calls=6)
        for i in range(1, 7):
            controller.consume(1)
            self.assertEqual(controller.calls_used(), i)
            self.assertEqual(controller.remaining_calls(), 6 - i)

        self.assertEqual(controller.calls_used(), 6)
        self.assertEqual(controller.remaining_calls(), 0)

    # --- Test 4: Seventh call is rejected ---
    def test_seventh_call_rejected(self):
        controller = BudgetController(max_calls=6)
        for _ in range(6):
            controller.consume(1)

        self.assertFalse(controller.can_call(1))
        with self.assertRaises(BudgetExhaustedError):
            controller.consume(1)

    # --- Test 5: Budget cannot become negative ---
    def test_budget_cannot_become_negative(self):
        controller = BudgetController(max_calls=6)
        for _ in range(6):
            controller.consume(1)

        with self.assertRaises(BudgetExhaustedError):
            controller.consume(1)

        self.assertEqual(controller.remaining_calls(), 0)
        self.assertGreaterEqual(controller.remaining_calls(), 0)

    # --- Test: Execution wrapper enforces budget ---
    def test_execute_wrapper(self):
        controller = BudgetController(max_calls=2)

        def dummy_tool(x, y):
            return x + y

        res1 = controller.execute(dummy_tool, 1, 2)
        self.assertEqual(res1, 3)
        self.assertEqual(controller.calls_used(), 1)

        res2 = controller.execute(dummy_tool, 3, 4)
        self.assertEqual(res2, 7)
        self.assertEqual(controller.calls_used(), 2)

        with self.assertRaises(BudgetExhaustedError):
            controller.execute(dummy_tool, 5, 6)

    # --- Test: Reset functionality ---
    def test_reset(self):
        controller = BudgetController(max_calls=6)
        controller.consume(4)
        self.assertEqual(controller.calls_used(), 4)
        controller.reset()
        self.assertEqual(controller.calls_used(), 0)
        self.assertEqual(controller.remaining_calls(), 6)


if __name__ == "__main__":
    unittest.main()
