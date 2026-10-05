import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.planner import (
    QueryPlanner,
    plan_query,
    MockLLMClient,
    NvidiaNemotronClient,
    QueryPlan,
)
from app.agent.validator import (
    validate_query_plan,
    parse_query_plan,
    PlanValidationError,
)


class TestQueryPlanner(unittest.TestCase):
    def setUp(self):
        self.mock_client = MockLLMClient()
        self.planner = QueryPlanner(llm_client=self.mock_client)

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

    # --- Test 5: Verify that more than 3 keywords are trimmed/capped ---
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

    # --- Test 7: Verify malformed LLM output does NOT crash the application ---
    def test_malformed_llm_output_handled_safely(self):
        # Case A: Plain text output without any JSON
        broken_client = MockLLMClient(default_response="This is plain text without any JSON")
        planner = QueryPlanner(llm_client=broken_client)
        plan = planner.plan("Any question")
        self.assertEqual(plan.question_type, "other")
        self.assertEqual(plan.keywords, [])
        self.assertIsNotNone(plan.error)

        # Case B: Missing required fields
        missing_fields_client = MockLLMClient(default_response={"only_type": "factual"})
        planner = QueryPlanner(llm_client=missing_fields_client)
        plan = planner.plan("Any question")
        self.assertEqual(plan.question_type, "other")
        self.assertEqual(plan.keywords, [])

        # Case C: Invalid question type normalized safely to 'other'
        invalid_type_client = MockLLMClient(default_response={
            "question_type": "unsupported_type",
            "keywords": ["database"]
        })
        planner = QueryPlanner(llm_client=invalid_type_client)
        plan = planner.plan("Where is the database?")
        self.assertEqual(plan.question_type, "other")
        self.assertEqual(plan.keywords, ["database"])

        # Case D: Empty keywords list handled safely
        empty_keywords_client = MockLLMClient(default_response={
            "question_type": "factual",
            "keywords": []
        })
        planner = QueryPlanner(llm_client=empty_keywords_client)
        plan = planner.plan("Any question")
        self.assertEqual(plan.question_type, "factual")
        self.assertEqual(plan.keywords, [])

    # --- Test 8: Missing NVIDIA_API_KEY handled safely ---
    def test_missing_nvidia_api_key_handled_safely(self):
        nemotron_client = NvidiaNemotronClient(api_key="")
        planner = QueryPlanner(llm_client=nemotron_client)

        plan = planner.plan("What are the degrees of freedom of a robot?")
        self.assertEqual(plan.question_type, "other")
        self.assertEqual(plan.keywords, [])
        self.assertIsNotNone(plan.error)
        self.assertIn("NVIDIA_API_KEY", plan.error)

    # --- Test 9: Markdown code fences in LLM output ---
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

    # --- Test 10: Empty question input validation ---
    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            self.planner.plan("")

        with self.assertRaises(ValueError):
            self.planner.plan("   ")

    # --- Requirement 1: Artificial Intelligence question ---
    def test_ai_question(self):
        question = "When was the term Artificial Intelligence adopted, and at which meeting?"
        plan = self.planner.plan(question)

        self.assertIn("Artificial Intelligence", plan.keywords)
        self.assertIn("adopted", plan.keywords)
        self.assertIn("meeting", plan.keywords)
        self.assertNotIn("When", plan.keywords)
        self.assertNotIn("was", plan.keywords)
        self.assertNotIn("term", plan.keywords)
        self.assertLessEqual(len(plan.keywords), 3)

        # Test deterministic cleaning even if LLM generated bad generic stopwords
        bad_llm_output = {
            "question_type": "factual",
            "keywords": ["When", "was", "term"]
        }
        mock_client = MockLLMClient(default_response=bad_llm_output)
        planner = QueryPlanner(llm_client=mock_client)
        plan_cleaned = planner.plan(question)
        self.assertIn("Artificial Intelligence", plan_cleaned.keywords)
        self.assertNotIn("When", plan_cleaned.keywords)
        self.assertNotIn("was", plan_cleaned.keywords)
        self.assertNotIn("term", plan_cleaned.keywords)

    # --- Requirement 2: A* question ---
    def test_a_star_programming_language_question(self):
        question = "What programming language was used to implement the A* algorithm in this document?"
        plan = self.planner.plan(question)

        keywords_lower = [k.lower() for k in plan.keywords]
        self.assertTrue(any("programming language" in k for k in keywords_lower))
        self.assertTrue(any("implement" in k for k in keywords_lower))
        self.assertTrue(any("a*" in k for k in keywords_lower))
        self.assertNotIn("what", keywords_lower)
        self.assertNotIn("was", keywords_lower)
        self.assertNotIn("used", keywords_lower)
        self.assertNotIn("the", keywords_lower)
        self.assertLessEqual(len(plan.keywords), 3)

        # Test with LLM output containing the generic word 'used'
        mock_output = {
            "question_type": "factual",
            "keywords": ["programming language", "used"]
        }
        mock_client = MockLLMClient(default_response=mock_output)
        planner = QueryPlanner(llm_client=mock_client)
        plan_cleaned = planner.plan(question)
        cleaned_lower = [k.lower() for k in plan_cleaned.keywords]
        self.assertNotIn("used", cleaned_lower)
        self.assertTrue(any("programming language" in k for k in cleaned_lower))

    # --- Requirement 3: PRM question ---
    def test_prm_question(self):
        question = "How does the Probabilistic Roadmap algorithm select landmarks and connect them?"
        plan = self.planner.plan(question)

        keywords_lower = [k.lower() for k in plan.keywords]
        self.assertTrue(any("probabilistic roadmap" in k for k in keywords_lower))
        self.assertTrue(any("landmarks" in k for k in keywords_lower))
        self.assertTrue(any("connect" in k for k in keywords_lower))
        self.assertNotIn("how", keywords_lower)
        self.assertNotIn("does", keywords_lower)
        self.assertNotIn("the", keywords_lower)
        self.assertLessEqual(len(plan.keywords), 3)

    # --- Requirement 4: Degrees of freedom ---
    def test_degrees_of_freedom_robot(self):
        question = "What are the degrees of freedom of a robot?"
        plan = self.planner.plan(question)

        keywords_lower = [k.lower() for k in plan.keywords]
        self.assertTrue(any("degrees of freedom" in k for k in keywords_lower))
        self.assertTrue(any("robot" in k for k in keywords_lower))
        self.assertNotIn("what", keywords_lower)
        self.assertNotIn("are", keywords_lower)
        self.assertNotIn("the", keywords_lower)
        self.assertLessEqual(len(plan.keywords), 3)

    # --- Requirement 5: Generic words not returned as standalone keywords ---
    def test_generic_words_filtered(self):
        mock_output = {
            "question_type": "factual",
            "keywords": ["what", "when", "was", "the", "used"]
        }
        client = MockLLMClient(default_response=mock_output)
        planner = QueryPlanner(llm_client=client)
        plan = planner.plan("What was used when the system ran?")
        for bad_kw in ["what", "when", "was", "the", "used"]:
            self.assertNotIn(bad_kw, [k.lower() for k in plan.keywords])

    # --- Requirement 6: Maximum 3 keywords ---
    def test_maximum_three_keywords_enforced(self):
        mock_output = {
            "question_type": "factual",
            "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"]
        }
        client = MockLLMClient(default_response=mock_output)
        planner = QueryPlanner(llm_client=client)
        plan = planner.plan("Tell me about kw1, kw2, kw3, kw4, kw5")
        self.assertLessEqual(len(plan.keywords), 3)

    # --- Test 15: Phrase preservation: "What is the A* search priority function?" ---
    def test_phrase_preservation_a_star_search(self):
        question = "What is the A* search priority function?"
        plan = self.planner.plan(question)

        keywords_lower = [k.lower() for k in plan.keywords]
        self.assertTrue(
            any("a* search" in k for k in keywords_lower) or any("priority function" in k for k in keywords_lower)
        )
        self.assertNotIn("what", keywords_lower)
        self.assertNotIn("is", keywords_lower)
        self.assertNotIn("the", keywords_lower)
        self.assertLessEqual(len(plan.keywords), 3)

    # --- Test 16: Phrase preservation: "What are the degrees of freedom of a helicopter?" ---
    def test_phrase_preservation_degrees_of_freedom_helicopter(self):
        question = "What are the degrees of freedom of a helicopter?"
        plan = self.planner.plan(question)

        keywords_lower = [k.lower() for k in plan.keywords]
        self.assertTrue(any("degrees of freedom" in k for k in keywords_lower))
        self.assertTrue(any("helicopter" in k for k in keywords_lower))
        self.assertNotIn("what", keywords_lower)
        self.assertNotIn("are", keywords_lower)
        self.assertNotIn("the", keywords_lower)
        self.assertLessEqual(len(plan.keywords), 3)

    # --- Test 17: Phrase preservation: "What is machine learning?" ---
    def test_phrase_preservation_machine_learning(self):
        question = "What is machine learning?"
        plan = self.planner.plan(question)

        keywords_lower = [k.lower() for k in plan.keywords]
        self.assertIn("machine learning", keywords_lower)
        self.assertNotIn("machine", keywords_lower)
        self.assertNotIn("learning", keywords_lower)
        self.assertNotIn("what", keywords_lower)
        self.assertNotIn("is", keywords_lower)

        # Test deterministic phrase merging when LLM splits the phrase
        split_llm_output = {
            "question_type": "factual",
            "keywords": ["machine", "learning"]
        }
        mock_client = MockLLMClient(default_response=split_llm_output)
        planner = QueryPlanner(llm_client=mock_client)
        plan_merged = planner.plan(question)
        self.assertEqual(plan_merged.keywords, ["machine learning"])

    # --- Test 18: Outside information (Part 4 constraint: Dartmouth example) ---
    def test_outside_information_filtered(self):
        plan = self.planner.plan("When was Artificial Intelligence adopted?")
        self.assertIn("Artificial Intelligence", plan.keywords)
        self.assertIn("adopted", plan.keywords)
        self.assertNotIn("Dartmouth", plan.keywords)


    # --- Test 19: Nemotron reasoning and thinking extraction ---
    def test_nemotron_reasoning_and_thinking_extraction(self):
        nemotron_response_with_thinking = """<think>
        The user is asking: When was the term Artificial Intelligence adopted, and at which meeting?
        Key concepts: Artificial Intelligence, adopted, meeting.
        Exclude stopwords: when, was, at, which.
        </think>
        ```json
        {
            "question_type": "factual",
            "keywords": ["Artificial Intelligence", "adopted", "meeting"]
        }
        ```"""
        client = MockLLMClient(default_response=nemotron_response_with_thinking)
        planner = QueryPlanner(llm_client=client)
        plan = planner.plan("When was the term Artificial Intelligence adopted, and at which meeting?")
        self.assertEqual(plan.question_type, "factual")
        self.assertEqual(plan.keywords, ["Artificial Intelligence", "adopted", "meeting"])

    # --- Test 20: NvidiaNemotronClient configuration ---
    def test_nvidia_nemotron_client_configuration(self):
        client = NvidiaNemotronClient(api_key="nvapi-test-key")
        self.assertEqual(client.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertEqual(client.model, "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(client.temperature, 0.1)
        self.assertEqual(client.top_p, 0.95)
        self.assertEqual(client.max_tokens, 512)
        self.assertEqual(client.reasoning_budget, 256)

    # --- Test 21: Configurable STOPWORDS set ---
    def test_configurable_stopwords_set(self):
        from app.agent.validator import DEFAULT_PLANNER_STOPWORDS
        custom_stopwords = DEFAULT_PLANNER_STOPWORDS | {"helicopter"}
        mock_client = MockLLMClient(stopwords=custom_stopwords)
        planner = QueryPlanner(llm_client=mock_client, stopwords=custom_stopwords)
        plan = planner.plan("What are the degrees of freedom of a helicopter?")

        keywords_lower = [k.lower() for k in plan.keywords]
        self.assertNotIn("helicopter", keywords_lower)
    # --- Test 22: Query plan with query field and numerical type ---
    def test_query_plan_with_query_field_and_numerical_type(self):
        raw_json = '{"question_type": "numerical", "keywords": ["file size", "maximum limit"], "query": "What is the maximum file size?"}'
        client = MockLLMClient(default_response=raw_json)
        planner = QueryPlanner(llm_client=client)
        plan = planner.plan("What is the maximum file size?")
        self.assertEqual(plan.question_type, "numerical")
        self.assertEqual(plan.keywords, ["file size", "maximum limit"])
        self.assertEqual(plan.query, "What is the maximum file size?")

    # --- Test 24: Full retrieval plan schema with phrases and normalized terms ---
    def test_search_query_phrases_and_normalized_terms(self):
        question = "What is the configuration space of the robot?"
        planner = QueryPlanner(llm_client=MockLLMClient())
        plan = planner.plan(question)

        self.assertEqual(plan.search_query, question)
        self.assertLessEqual(len(plan.keywords), 3)
        self.assertTrue(any("configuration space" in p.lower() for p in plan.phrases))
        self.assertTrue(any("configuration" in n.lower() for n in plan.normalized_terms))
        self.assertIn("exact_phrase", plan.retrieval_strategy)
        self.assertIn("aho_corasick", plan.retrieval_strategy)
        self.assertIn("bm25", plan.retrieval_strategy)

    # --- Test 25: Formula question includes regex strategy ---
    def test_formula_question_includes_regex_strategy(self):
        question = "What is the priority formula f(n) = g(n) + h(n) in A*?"
        planner = QueryPlanner(llm_client=MockLLMClient())
        plan = planner.plan(question)

        self.assertIn("regex", plan.retrieval_strategy)
        self.assertTrue(any("a*" in k.lower() or "formula" in k.lower() for k in plan.keywords))

    # --- Test 26: Fallback retrieval plan never returns zero retrieval attempts ---
    def test_fallback_retrieval_plan_structure(self):
        from app.agent.validator import build_fallback_retrieval_plan
        question = "How does uniform cost search select nodes?"
        fallback = build_fallback_retrieval_plan(question)

        self.assertEqual(fallback["question_type"], "other")
        self.assertEqual(fallback["search_query"], question)
        self.assertGreater(len(fallback["keywords"]), 0)
        self.assertLessEqual(len(fallback["keywords"]), 3)
        self.assertTrue(any("uniform cost search" in p.lower() for p in fallback["phrases"]))
        self.assertIn("exact_phrase", fallback["retrieval_strategy"])
        self.assertIn("aho_corasick", fallback["retrieval_strategy"])
        self.assertIn("bm25", fallback["retrieval_strategy"])

    # --- Test 27: Full Retrieval Plan JSON compliance ---
    def test_full_retrieval_plan_json_schema(self):
        raw_json = (
            '{\n'
            '  "question_type": "definition",\n'
            '  "search_query": "What is configuration space?",\n'
            '  "keywords": ["configuration space"],\n'
            '  "synonyms": ["c-space", "state space"],\n'
            '  "phrases": ["configuration space"],\n'
            '  "normalized_terms": ["configuration space", "configur", "space"],\n'
            '  "retrieval_strategy": ["exact_phrase", "aho_corasick", "bm25"]\n'
            '}'
        )
        client = MockLLMClient(default_response=raw_json)
        planner = QueryPlanner(llm_client=client)
        plan = planner.plan("What is configuration space?")

        self.assertEqual(plan.question_type, "definition")
        self.assertEqual(plan.search_query, "What is configuration space?")
        self.assertEqual(plan.keywords, ["configuration space"])
        self.assertEqual(plan.synonyms, ["c-space", "state space"])
        self.assertEqual(plan.phrases, ["configuration space"])
        self.assertEqual(plan.retrieval_strategy, ["exact_phrase", "aho_corasick", "bm25"])


if __name__ == "__main__":
    unittest.main()


