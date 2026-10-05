import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.agent import DocumentAgent, run_agent
from app.harness.state import AgentState


class FakeTools:
    def __init__(self, search_map=None, pages=None):
        self.search_map = {k.lower(): v for k, v in (search_map or {}).items()}
        self.pages = pages or {}
        self.search_calls = []
        self.get_page_calls = []
        self.fail_keywords = set()
        self.fail_pages = set()

    def search_keyword(self, doc_id: str, keyword: str):
        self.search_calls.append((doc_id, keyword))
        if keyword in self.fail_keywords:
            raise RuntimeError(f"Simulated search error for '{keyword}'")
        page_nums = self.search_map.get(keyword.lower(), [])
        return [{"page_number": p, "snippet": f"snippet on page {p}"} for p in page_nums]

    def get_page(self, doc_id: str, page_number: int):
        self.get_page_calls.append((doc_id, page_number))
        if page_number in self.fail_pages:
            raise RuntimeError(f"Simulated page error for page {page_number}")
        text = self.pages.get(page_number, f"Default content on page {page_number}")
        return {"page_number": page_number, "text": text}


class TestDocumentAgent(unittest.TestCase):
    # --- Test 6: Agent never exceeds six calls ---
    def test_agent_never_exceeds_six_calls(self):
        # Setup lots of keywords and pages
        fake = FakeTools(
            search_map={"k1": [1, 2], "k2": [3, 4], "k3": [5, 6]},
            pages={i: f"Page {i} content text." for i in range(1, 10)},
        )
        planner = lambda q: {"question_type": "factual", "keywords": ["k1", "k2", "k3"]}
        agent = DocumentAgent(
            max_budget=6,
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "What are the rules?")
        self.assertLessEqual(state["tool_calls_used"], 6)
        self.assertGreaterEqual(state["remaining_budget"], 0)
        self.assertEqual(len(fake.search_calls) + len(fake.get_page_calls), state["tool_calls_used"])

    # --- Test 7: Already-used search keyword is not searched again ---
    def test_already_used_keyword_not_searched_again(self):
        fake = FakeTools(search_map={"oauth": [5]})
        planner = lambda q: {"question_type": "factual", "keywords": ["oauth"]}
        agent = DocumentAgent(
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "Tell me about oauth")
        searched_keywords = [c[1] for c in fake.search_calls]
        self.assertEqual(searched_keywords.count("oauth"), 1)

    # --- Test 8: Agent can select a useful keyword ---
    def test_agent_selects_useful_keyword(self):
        fake = FakeTools(search_map={"authentication": [2, 5]})
        planner = lambda q: {"question_type": "factual", "keywords": ["authentication"]}
        agent = DocumentAgent(
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "What authentication is supported?")
        self.assertIn("authentication", [c[1] for c in fake.search_calls])
        self.assertTrue(any(c["page_number"] in [2, 5] for c in state["candidate_pages"]))

    # --- Test 9: Agent updates entropy after an action ---
    def test_agent_updates_entropy_after_action(self):
        fake = FakeTools(
            search_map={"auth": [1, 2, 3]},
            pages={1: "OAuth is supported.", 2: "OAuth authentication details."},
        )
        planner = lambda q: {"question_type": "factual", "keywords": ["auth"]}
        agent = DocumentAgent(
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "How to authenticate?")
        self.assertTrue(len(state["tool_trace"]) >= 1)
        first_call = state["tool_trace"][0]
        self.assertIn("entropy_before", first_call)
        self.assertIn("entropy_after", first_call)

    # --- Test 10: Actual information gain is calculated correctly: IG = H_before - H_after ---
    def test_actual_information_gain_calculated(self):
        fake = FakeTools(
            search_map={"oauth": [5]},
            pages={5: "OAuth authentication is supported."},
        )
        planner = lambda q: {"question_type": "factual", "keywords": ["oauth"]}
        agent = DocumentAgent(
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "Does it use OAuth?")
        for trace_entry in state["tool_trace"]:
            expected_ig = round(trace_entry["entropy_before"] - trace_entry["entropy_after"], 4)
            self.assertAlmostEqual(trace_entry["information_gain"], expected_ig, places=4)

    # --- Test 11: Agent stops when confidence is high enough ---
    def test_agent_stops_when_confidence_high_enough(self):
        # Two pages strongly verify OAuth, boosting confidence >= tau (0.7)
        fake = FakeTools(
            search_map={"oauth": [1, 2]},
            pages={
                1: "OAuth authentication is supported.",
                2: "Users may authenticate through OAuth.",
            },
        )
        planner = lambda q: {"question_type": "factual", "keywords": ["oauth"]}
        agent = DocumentAgent(
            max_budget=6,
            tau=0.7,
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "What is the auth method?")
        self.assertEqual(state["status"], "completed")
        self.assertGreaterEqual(state["confidence"], 0.7)
        # Did not need all 6 calls!
        self.assertLess(state["tool_calls_used"], 6)

    # --- Test 12: Agent stops when no useful action remains ---
    def test_agent_stops_when_no_useful_action_remains(self):
        fake = FakeTools(search_map={"unknown": []})  # No pages found
        planner = lambda q: {"question_type": "factual", "keywords": ["unknown"]}
        agent = DocumentAgent(
            max_budget=6,
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "What is the secret?")
        # Searched once, 0 candidates, stops after 1 call
        self.assertEqual(state["tool_calls_used"], 1)
        self.assertEqual(state["remaining_budget"], 5)

    # --- Test 13: Agent stops when budget is exhausted ---
    def test_agent_stops_when_budget_exhausted(self):
        # Low budget of 2
        fake = FakeTools(
            search_map={"k1": [1, 2, 3]},
            pages={i: f"Page {i} text" for i in range(1, 4)},
        )
        planner = lambda q: {"question_type": "factual", "keywords": ["k1"]}
        agent = DocumentAgent(
            max_budget=2,
            tau=0.99,  # High threshold to prevent early stopping
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "Question")
        self.assertEqual(state["tool_calls_used"], 2)
        self.assertEqual(state["remaining_budget"], 0)

    # --- Test 14: Unresolved contradiction can cause another information-seeking action ---
    def test_unresolved_contradiction_prioritizes_clarification(self):
        # Page 3 and Page 8 have a contradiction on file size
        fake = FakeTools(
            search_map={"file size": [3, 8, 12]},
            pages={
                3: "Maximum file size is 10 MB.",
                8: "Maximum file size is 20 MB.",
                12: "Maximum file size is 20 MB. Effective July 2026, 10 MB is replaced by 20 MB.",
            },
        )
        planner = lambda q: {"question_type": "factual", "keywords": ["file size"]}
        agent = DocumentAgent(
            max_budget=5,
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "What is the maximum file size?")
        # Agent reads pages to resolve contradiction
        self.assertGreaterEqual(len(state["retrieved_pages"]), 2)

    # --- Test 15: Insufficient evidence produces "insufficient_information" state ---
    def test_insufficient_evidence_produces_insufficient_information_state(self):
        fake = FakeTools(search_map={"empty": []})
        planner = lambda q: {"question_type": "factual", "keywords": ["empty"]}
        agent = DocumentAgent(
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "Question about missing topic")
        self.assertEqual(state["status"], "insufficient_information")

    # --- Test 16: Tool failure is handled safely ---
    def test_tool_failure_handled_safely(self):
        fake = FakeTools(
            search_map={"bad": [1], "good": [2]},
            pages={2: "Good page content."},
        )
        fake.fail_keywords.add("bad")
        planner = lambda q: {"question_type": "factual", "keywords": ["bad", "good"]}
        agent = DocumentAgent(
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "Test error tolerance")
        self.assertIsInstance(state, AgentState)
        # Should record failure in trace and proceed with "good" keyword
        self.assertTrue(any("failed" in t["status"] for t in state["tool_trace"]))

    # --- Test 17: Tool calls are logged in tool_trace ---
    def test_tool_calls_logged_in_tool_trace(self):
        fake = FakeTools(
            search_map={"token": [4]},
            pages={4: "Token details on page 4."},
        )
        planner = lambda q: {"question_type": "factual", "keywords": ["token"]}
        agent = DocumentAgent(
            search_fn=fake.search_keyword,
            get_page_fn=fake.get_page,
            planner_fn=planner,
        )

        state = agent.run("doc_test", "What is the token?")
        self.assertGreaterEqual(len(state["tool_trace"]), 1)
        for entry in state["tool_trace"]:
            self.assertIn("call_number", entry)
            self.assertIn("tool", entry)
            self.assertIn("input", entry)
            self.assertIn("information_gain", entry)
            self.assertIn("calls_used", entry)
            self.assertIn("remaining_budget", entry)


if __name__ == "__main__":
    unittest.main()
