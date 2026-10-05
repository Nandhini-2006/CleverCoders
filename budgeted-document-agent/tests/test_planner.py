import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.planner import QueryPlanner, plan_query, MockLLMClient, QueryPlan
from app.agent.validator import validate_query_plan, PlanValidationError


class TestQueryPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = QueryPlanner()

    # --- Test 1: "What authentication methods are supported?" ---
    def test_question_1_authentication_methods(self):
        question = "What authentication methods are supported?"
        plan = self.planner.plan(question)

        self.assertIsInstance(plan, QueryPlan)
        self.assertEqual(plan["question_type"], "factual")
        self.assertLessEqual(len(plan["keywords"]), 3)
        self.assertGreaterEqual(len(plan["keywords"]), 1)
        self.assertIn("authentication", [k.lower() for k in plan["keywords"]])
        # Verify property access
        self.assertEqual(plan.question_type, "factual")
        self.assertEqual(plan.keywords, plan["keywords"])

    # --- Test 2: "How do I reset my password?" ---
    def test_question_2_reset_password(self):
        question = "How do I reset my password?"
        plan = self.planner.plan(question)

        self.assertEqual(plan["question_type"], "procedural")
        self.assertLessEqual(len(plan["keywords"]), 3)
        self.assertGreaterEqual(len(plan["keywords"]), 1)
        self.assertTrue(any("password" in k.lower() for k in plan["keywords"]))

    # --- Test 3: "Does the document mention OAuth?" ---
    def test_question_3_mention_oauth(self):
        question = "Does the document mention OAuth?"
        plan = self.planner.plan(question)

        self.assertEqual(plan["question_type"], "lookup")
        self.assertLessEqual(len(plan["keywords"]), 3)
        self.assertGreaterEqual(len(plan["keywords"]), 1)
        self.assertTrue(any("oauth" in k.lower() for k in plan["keywords"]))

    # --- Test 4: "Compare OAuth and password authentication." ---
    def test_question_4_compare_oauth(self):
        question = "Compare OAuth and password authentication."
        plan = self.planner.plan(question)

        self.assertEqual(plan["question_type"], "comparison")
        self.assertLessEqual(len(plan["keywords"]), 3)
        self.assertGreaterEqual(len(plan["keywords"]), 1)
        self.assertTrue(any("oauth" in k.lower() for k in plan["keywords"]))

    # --- Test 5: Verify that more than 3 keywords are rejected/trimmed ---
    def test_more_than_three_keywords_trimmed_and_rejected(self):
        # 5a. Trimming behavior (default)
        raw_output_4_items = {
            "question_type": "factual",
            "keywords": ["security", "encryption", "SSL", "TLS", "certificates"]
        }
        mock_client = MockLLMClient(default_response=raw_output_4_items)
        planner = QueryPlanner(llm_client=mock_client)
        plan = planner.plan("Tell me about security")
        self.assertEqual(len(plan.keywords), 3)
        self.assertEqual(plan.keywords, ["security", "encryption", "SSL"])

        # 5b. Strict rejection mode in validator
        with self.assertRaises(PlanValidationError):
            validate_query_plan(raw_output_4_items, strict_max=True)

    # --- Test 6: Verify duplicate keywords are removed ---
    def test_duplicate_keywords_removed(self):
        raw_output_duplicates = {
            "question_type": "factual",
            "keywords": ["OAuth", "oauth", "OAUTH", "token"]
        }
        mock_client = MockLLMClient(default_response=raw_output_duplicates)
        planner = QueryPlanner(llm_client=mock_client)
        plan = planner.plan("Does it use OAuth?")
        self.assertEqual(plan.keywords, ["OAuth", "token"])

    # --- Test 7: Verify malformed LLM output is handled safely ---
    def test_malformed_llm_output_handled_safely(self):
        # Case A: Not valid JSON
        broken_client = MockLLMClient(default_response="This is plain text without any JSON")
        planner = QueryPlanner(llm_client=broken_client)
        with self.assertRaises(PlanValidationError):
            planner.plan("Any question")

        # Case B: Missing required fields
        missing_fields_client = MockLLMClient(default_response={"only_type": "factual"})
        planner = QueryPlanner(llm_client=missing_fields_client)
        with self.assertRaises(PlanValidationError):
            planner.plan("Any question")

        # Case C: Invalid question type
        invalid_type_client = MockLLMClient(default_response={
            "question_type": "unsupported_type",
            "keywords": ["valid"]
        })
        planner = QueryPlanner(llm_client=invalid_type_client)
        with self.assertRaises(PlanValidationError):
            planner.plan("Any question")

        # Case D: Empty keywords list
        empty_keywords_client = MockLLMClient(default_response={
            "question_type": "factual",
            "keywords": []
        })
        planner = QueryPlanner(llm_client=empty_keywords_client)
        with self.assertRaises(PlanValidationError):
            planner.plan("Any question")

    # --- Additional Test: Markdown code fences in LLM output ---
    def test_markdown_code_fence_json(self):
        fenced_json = """```json
        {
            "question_type": "lookup",
            "keywords": ["database", "postgresql"]
        }
        ```"""
        mock_client = MockLLMClient(default_response=fenced_json)
        planner = QueryPlanner(llm_client=mock_client)
        plan = planner.plan("Where is the database defined?")
        self.assertEqual(plan.question_type, "lookup")
        self.assertEqual(plan.keywords, ["database", "postgresql"])

    # --- Additional Test: Empty question input validation ---
    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            self.planner.plan("")

        with self.assertRaises(ValueError):
            self.planner.plan("   ")


if __name__ == "__main__":
    unittest.main()
