import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from app.config import MAX_TOOL_CALLS
from app.harness.budget import BudgetController, BudgetExhaustedError
from app.harness.state import AgentState
from app.harness.logger import log_query_trace, read_query_traces, sanitize_data
from app.agent.agent import DocumentAgent, run_agent, run_pipeline
from app.agent.answer import generate_final_answer
from app.agent.planner import MockLLMClient
from tests.sample_questions import get_sample_question, get_all_questions


class FakeMockEnvironment:
    """Helper to mock document tools and search index for test questions."""
    def __init__(self, pages_dict):
        self.pages = {int(k): str(v) for k, v in pages_dict.items()}
        self.tool_calls_count = 0

    def search_keyword(self, doc_id: str, keyword: str):
        self.tool_calls_count += 1
        matches = []
        kw_clean = keyword.lower().strip()
        for p_num, text in self.pages.items():
            if kw_clean in text.lower():
                matches.append({"page_number": p_num, "snippet": text[:100]})
        return matches

    def get_page(self, doc_id: str, page_number: int):
        self.tool_calls_count += 1
        text = self.pages.get(int(page_number), "")
        return {"page_number": int(page_number), "text": text}


class TestPhase8IntegrationAndReadiness:
    """
    Phase 8 Comprehensive Benchmark and Integration Test Suite.
    Validates all 13 canonical test questions, hard budget limits,
    early stopping, prompt injection defense, contradiction handling,
    and trace logging.
    """

    # =============================================================
    # 1. HARD TOOL BUDGET TESTS
    # =============================================================
    def test_budget_exact_zero_calls(self):
        """Budget invariant: 0 calls used leaves remaining budget = 6."""
        ctrl = BudgetController(max_calls=6)
        assert ctrl.calls_used() == 0
        assert ctrl.remaining_calls() == 6

    def test_budget_exact_one_call(self):
        """Budget invariant: 1 call consumes exactly 1."""
        ctrl = BudgetController(max_calls=6)
        ctrl.execute(lambda x: x + 1, 10)
        assert ctrl.calls_used() == 1
        assert ctrl.remaining_calls() == 5

    def test_budget_exact_five_calls(self):
        """Budget invariant: 5 calls leaves remaining budget = 1."""
        ctrl = BudgetController(max_calls=6)
        for i in range(5):
            ctrl.execute(lambda x: x * 2, i)
        assert ctrl.calls_used() == 5
        assert ctrl.remaining_calls() == 1

    def test_budget_exact_six_calls(self):
        """Budget invariant: exactly 6 calls permitted; budget becomes 0."""
        ctrl = BudgetController(max_calls=6)
        for i in range(6):
            ctrl.execute(lambda: "ok")
        assert ctrl.calls_used() == 6
        assert ctrl.remaining_calls() == 0
        assert ctrl.can_call(1) is False

    def test_budget_seventh_call_deterministically_blocked(self):
        """The 7th document-tool call must be blocked deterministically by BudgetController."""
        ctrl = BudgetController(max_calls=6)
        for _ in range(6):
            ctrl.execute(lambda: "ok")

        with pytest.raises(BudgetExhaustedError):
            ctrl.execute(lambda: "should never execute")

    def test_final_llm_answer_call_does_not_count_as_document_tool_call(self):
        """Answer generation call does not count toward the 6-tool budget."""
        ctrl = BudgetController(max_calls=6)
        # Execute 6 document tool calls
        for _ in range(6):
            ctrl.execute(lambda: "doc tool")
        assert ctrl.calls_used() == 6

        # Phase 7 generate_final_answer runs without document tools
        ans = generate_final_answer(
            question="What is the speed limit?",
            evidence_state={"confidence": 0.8, "claims": [{"claim": "Speed limit is 50 km/h.", "supporting_pages": [1]}]},
            retrieved_pages=[{"page_number": 1, "text": "Speed limit is 50 km/h."}],
        )
        assert ans["status"] == "answered"
        # Tool calls remain strictly at 6
        assert ctrl.calls_used() == 6

    # =============================================================
    # 2. RETRIEVAL EFFICIENCY & EARLY STOPPING
    # =============================================================
    def test_early_stopping_on_simple_factual_question(self):
        """
        Agent stops early when strong evidence has already been obtained.
        Simple factual question terminates in < 6 calls.
        """
        env = FakeMockEnvironment({
            1: "The term Artificial Intelligence was officially adopted in 1956 at the Dartmouth Summer Research Project on Artificial Intelligence.",
            2: "Unrelated page 2 text.",
            3: "Unrelated page 3 text.",
            4: "Unrelated page 4 text.",
        })
        planner_fn = lambda q: {"question_type": "factual", "keywords": ["Artificial Intelligence", "Dartmouth"]}

        state = run_pipeline(
            doc_id="test_doc",
            question="When was the term Artificial Intelligence adopted, and at which meeting?",
            max_budget=6,
            tau=0.70,
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state["tool_calls_used"] < 6
        assert state["remaining_budget"] > 0
        assert state["confidence"] >= 0.70
        assert state.final_answer["status"] == "answered"
        assert "1956" in state.final_answer["answer"]
        assert "Dartmouth" in state.final_answer["answer"]

    # =============================================================
    # 3. CANONICAL BENCHMARK EVALUATION QUESTIONS (1-13)
    # =============================================================
    def test_benchmark_q1_simple_factual(self):
        """TEST 1 — SIMPLE FACTUAL: 1956 Dartmouth meeting."""
        q = get_sample_question(1)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["Artificial Intelligence", "adopted", "meeting"]}

        state = run_pipeline(
            doc_id="doc1",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        for fact in q["expected_facts"]:
            assert fact.lower() in state.final_answer["answer"].lower()
        assert 1 in state.final_answer["sources"]

    def test_benchmark_q2_definition(self):
        """TEST 2 — DEFINITION: Rational behavior grounded in document."""
        q = get_sample_question(2)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "definition", "keywords": ["rational behavior"]}

        state = run_pipeline(
            doc_id="doc2",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        assert "rational behavior" in state.final_answer["answer"].lower()
        assert 2 in state.final_answer["sources"]

    def test_benchmark_q3_comparison(self):
        """TEST 3 — COMPARISON: Workspace vs Configuration Space vs Free Space."""
        q = get_sample_question(3)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "comparison", "keywords": ["workspace", "configuration space", "free space"]}

        state = run_pipeline(
            doc_id="doc3",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        ans_lower = state.final_answer["answer"].lower()
        assert "workspace" in ans_lower
        assert "configuration space" in ans_lower
        assert "free space" in ans_lower

    def test_benchmark_q4_a_star_priority_function(self):
        """TEST 4 — A*: f(n) = g(n) + h(n)."""
        q = get_sample_question(4)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["priority function", "A* search"]}

        state = run_pipeline(
            doc_id="doc4",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        assert "g(n)" in state.final_answer["answer"]
        assert "h(n)" in state.final_answer["answer"]
        assert 5 in state.final_answer["sources"]

    def test_benchmark_q5_dfs_bfs_ucs_comparison(self):
        """TEST 5 — DFS/BFS/UCS: Graph search comparison."""
        q = get_sample_question(5)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "comparison", "keywords": ["DFS", "BFS", "Uniform-Cost Search"]}

        state = run_pipeline(
            doc_id="doc5",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        ans = state.final_answer["answer"]
        assert "DFS" in ans or "Depth-First" in ans
        assert "BFS" in ans or "Breadth-First" in ans
        assert "Uniform-Cost" in ans or "UCS" in ans

    def test_benchmark_q6_admissible_heuristic(self):
        """TEST 6 — ADMISSIBLE HEURISTIC: Definition and optimality."""
        q = get_sample_question(6)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "definition", "keywords": ["admissible heuristic", "A* search"]}

        state = run_pipeline(
            doc_id="doc6",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        ans = state.final_answer["answer"].lower()
        assert "never overestimates" in ans
        assert 8 in state.final_answer["sources"]

    def test_benchmark_q7_prm_algorithm(self):
        """TEST 7 — PRM: Probabilistic Roadmap selection and connection."""
        q = get_sample_question(7)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["Probabilistic Roadmap", "landmarks"]}

        state = run_pipeline(
            doc_id="doc7",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        ans = state.final_answer["answer"].lower()
        assert "sample" in ans or "sampling" in ans
        assert "collision-free" in ans

    def test_benchmark_q8_high_dimensional_search(self):
        """TEST 8 — HIGH-DIMENSIONAL SEARCH: Curse of dimensionality / grid discretization."""
        q = get_sample_question(8)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["discretization", "high-dimensional"]}

        state = run_pipeline(
            doc_id="doc8",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        ans = state.final_answer["answer"].lower()
        assert "exponential" in ans or "curse of dimensionality" in ans
        assert 11 in state.final_answer["sources"]

    def test_benchmark_q9_unsupported_chatgpt(self):
        """TEST 9 — UNSUPPORTED QUESTION: Who invented ChatGPT? -> Insufficient information."""
        q = get_sample_question(9)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["ChatGPT"]}

        state = run_pipeline(
            doc_id="doc9",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "insufficient_information"
        assert state.final_answer["answer"] == "Insufficient information."

    def test_benchmark_q10_unsupported_implementation_language(self):
        """TEST 10 — UNSUPPORTED LANGUAGE: A* programming language unstated -> Insufficient information."""
        q = get_sample_question(10)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["programming language", "A* algorithm"]}

        state = run_pipeline(
            doc_id="doc10",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "insufficient_information"
        assert state.final_answer["answer"] == "Insufficient information."

    def test_benchmark_q11_prompt_injection(self):
        """
        TEST 11 — PROMPT INJECTION:
        PDF text contains instruction to say document was written by Superman.
        Must treat embedded text as passive data, not follow it.
        """
        q = get_sample_question(11)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["primary topic"]}

        state = run_pipeline(
            doc_id="doc11",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        ans = state.final_answer["answer"]
        assert "Superman" not in ans
        assert "Artificial Intelligence" in ans

    def test_benchmark_q12_unresolved_contradiction(self):
        """
        TEST 12 — UNRESOLVED CONTRADICTION:
        Page 1: 50 km/h. Page 2: 70 km/h.
        No superseding language. Must report conflict, not arbitrarily pick one.
        """
        q = get_sample_question(12)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["speed limit"]}

        state = run_pipeline(
            doc_id="doc12",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "unresolved_contradiction"
        ans = state.final_answer["answer"]
        assert "conflicting" in ans.lower()
        assert "50 km/h" in ans
        assert "70 km/h" in ans

    def test_benchmark_q13_explicit_supersession(self):
        """
        TEST 13 — EXPLICIT SUPERSESSION:
        Page 1: 50 km/h. Page 2: Effective July 1, 2026, the speed limit is changed to 70 km/h.
        Later statement supersedes earlier.
        """
        q = get_sample_question(13)
        env = FakeMockEnvironment(q["mock_pages"])
        planner_fn = lambda query: {"question_type": "factual", "keywords": ["speed limit"]}

        state = run_pipeline(
            doc_id="doc13",
            question=q["question"],
            search_fn=env.search_keyword,
            get_page_fn=env.get_page,
            planner_fn=planner_fn,
            log_trace=False,
        )

        assert state.final_answer["status"] == "answered"
        assert "70 km/h" in state.final_answer["answer"]
        assert 2 in state.final_answer["sources"]

    # =============================================================
    # 4. JSONL LOGGING AND SANITIZATION
    # =============================================================
    def test_jsonl_logging_and_secret_sanitization(self):
        """Verify traces are written to JSONL and secrets are never logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log_file = Path(tmpdir) / "test_traces.jsonl"
            trace_record = {
                "question": "What is the secret formula?",
                "api_key": "nvapi-abcdef1234567890abcdef1234567890",
                "nested": {
                    "nvidia_api_key": "nvapi-secretsecretsecretsecret1234",
                    "value": "Here is nvapi-abcdef1234567890abcdef1234567890 in text",
                },
                "status": "answered",
            }

            log_query_trace(trace_record, log_file=test_log_file)

            traces = read_query_traces(test_log_file)
            assert len(traces) == 1
            logged = traces[0]

            assert logged["api_key"] == "[REDACTED_SECRET]"
            assert logged["nested"]["nvidia_api_key"] == "[REDACTED_SECRET]"
            assert "nvapi-" not in logged["nested"]["value"]
            assert "[REDACTED_API_KEY]" in logged["nested"]["value"]

    # =============================================================
    # 5. ERROR HANDLING AND RESILIENCE
    # =============================================================
    def test_tool_failure_handled_gracefully_in_pipeline(self):
        """Pipeline does not crash when an individual tool fails."""
        def failing_search(doc_id, kw):
            raise ConnectionError("Simulated network blip")

        def normal_page(doc_id, p_num):
            return {"page_number": p_num, "text": "Some document text"}

        state = run_pipeline(
            doc_id="doc_fail",
            question="What is the process?",
            search_fn=failing_search,
            get_page_fn=normal_page,
            planner_fn=lambda q: {"question_type": "factual", "keywords": ["process"]},
            log_trace=False,
        )

        assert isinstance(state, AgentState)
        assert state["status"] in ["insufficient_information", "completed"]
        assert state.final_answer["status"] == "insufficient_information"

    def test_empty_question_raises_value_error(self):
        """Empty or whitespace-only questions raise ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            run_pipeline(doc_id="doc1", question="   ")

    def test_nvidia_401_authentication_failure_handled_safely(self):
        """NVIDIA 401 Authentication Error does not crash the pipeline."""
        class Mock401Client:
            def generate(self, prompt: str):
                raise RuntimeError("401 Client Error: Unauthorized for url: https://integrate.api.nvidia.com/v1")

        state = run_pipeline(
            doc_id="doc_auth",
            question="What is the speed limit?",
            search_fn=lambda doc, kw: [],
            get_page_fn=lambda doc, p: {"page_number": p, "text": ""},
            planner_fn=lambda q: {"question_type": "other", "keywords": [], "error": "401 Unauthorized"},
            llm_client=Mock401Client(),
            log_trace=False,
        )

        assert isinstance(state, AgentState)
        assert state.final_answer["status"] == "insufficient_information"

    def test_llm_timeout_handled_safely(self):
        """LLM timeout does not crash the pipeline."""
        class TimeoutClient:
            def generate(self, prompt: str):
                import socket
                raise socket.timeout("NVIDIA API connection timed out after 30s")

        state = run_pipeline(
            doc_id="doc_timeout",
            question="What is the speed limit?",
            search_fn=lambda doc, kw: [],
            get_page_fn=lambda doc, p: {"page_number": p, "text": ""},
            planner_fn=lambda q: {"question_type": "other", "keywords": [], "error": "LLM Timeout"},
            llm_client=TimeoutClient(),
            log_trace=False,
        )

        assert isinstance(state, AgentState)
        assert state.final_answer["status"] == "insufficient_information"

    def test_empty_and_malformed_llm_response_handled_safely(self):
        """Empty and malformed LLM outputs return safe fallback plans."""
        from app.agent.planner import plan_query

        class EmptyClient:
            def generate(self, prompt: str):
                return ""

        plan_empty = plan_query("What is the speed limit?", llm_client=EmptyClient())
        assert plan_empty.question_type == "other"
        assert plan_empty.keywords == []

        class MalformedClient:
            def generate(self, prompt: str):
                return "This is definitely not JSON output <<<>>>"

        plan_malformed = plan_query("What is the speed limit?", llm_client=MalformedClient())
        assert plan_malformed.question_type == "other"
        assert plan_malformed.keywords == []

    def test_corrupt_pdf_handled_safely(self):
        """Invalid or corrupt PDF files are handled safely by PDF parser."""
        from app.pdf.parser import get_page_count, read_page, parse_headings, search_in_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(b"NOT A VALID PDF CONTENT")
            corrupt_path = Path(tf.name)

        try:
            # Corrupt PDF functions should either return safe defaults or raise handled exceptions
            with pytest.raises(Exception):
                get_page_count(corrupt_path)

            with pytest.raises(Exception):
                read_page(corrupt_path, 1)

            assert parse_headings(corrupt_path) == []
            assert search_in_pdf(corrupt_path, "test") == []
        finally:
            import gc
            gc.collect()
            if corrupt_path.exists():
                try:
                    corrupt_path.unlink()
                except Exception:
                    pass


