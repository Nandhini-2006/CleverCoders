import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.retriever import PageRetriever, retrieve_pages


class FakeDocumentTools:
    """
    In-memory mock for document tools to test PageRetriever
    without requiring real PDF files or disk access.
    """
    def __init__(self, search_map=None, pages=None):
        # search_map: dict of {keyword.lower(): [page_numbers...]}
        self.search_map = {k.lower(): v for k, v in (search_map or {}).items()}
        # pages: dict of {page_number: text}
        self.pages = pages or {}
        self.search_calls = []
        self.get_page_calls = []
        self.fail_keywords = set()
        self.fail_pages = set()
        self.global_search_failure = False
        self.global_get_page_failure = False

    def search_keyword(self, doc_id: str, keyword: str):
        self.search_calls.append((doc_id, keyword))
        if self.global_search_failure or keyword in self.fail_keywords:
            raise RuntimeError(f"Simulated search error for '{keyword}'")

        page_nums = self.search_map.get(keyword.lower(), [])
        return [{"page_number": p, "snippet": f"snippet on page {p}"} for p in page_nums]

    def get_page(self, doc_id: str, page_number: int):
        self.get_page_calls.append((doc_id, page_number))
        if self.global_get_page_failure or page_number in self.fail_pages:
            raise RuntimeError(f"Simulated get_page error for page {page_number}")

        text = self.pages.get(page_number, f"Page {page_number} body text")
        return {"page_number": page_number, "text": text}


class TestPageRetriever(unittest.TestCase):
    # --- Test 1: One keyword returns multiple pages ---
    def test_one_keyword_returns_multiple_pages(self):
        fake = FakeDocumentTools(search_map={"authentication": [2, 5, 8]})
        retriever = PageRetriever(search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"question_type": "factual", "keywords": ["authentication"]}
        result = retriever.retrieve("doc_test", query_plan)

        self.assertEqual(len(result["candidate_pages"]), 3)
        candidate_pages = [c["page_number"] for c in result["candidate_pages"]]
        self.assertEqual(candidate_pages, [2, 5, 8])
        # 1 search + 3 gets = 4 tool calls
        self.assertEqual(result["tool_calls_used"], 4)
        self.assertEqual(len(result["retrieved_pages"]), 3)

    # --- Test 2: Multiple keywords return overlapping pages ---
    def test_multiple_keywords_overlapping_pages(self):
        fake = FakeDocumentTools(search_map={
            "authentication": [3, 5],
            "oauth": [5, 7],
            "login": [3, 5]
        })
        retriever = PageRetriever(search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"question_type": "factual", "keywords": ["authentication", "oauth", "login"]}
        result = retriever.retrieve("doc_test", query_plan)

        # Page 5 should match all 3 keywords
        page_5 = next(c for c in result["candidate_pages"] if c["page_number"] == 5)
        self.assertEqual(sorted(page_5["matched_keywords"]), ["authentication", "login", "oauth"])
        self.assertEqual(page_5["score"], 3)

    # --- Test 3: Duplicate pages are merged ---
    def test_duplicate_pages_are_merged(self):
        fake = FakeDocumentTools(search_map={
            "authentication": [3, 5],
            "oauth": [5, 7],
            "login": [3, 5]
        })
        retriever = PageRetriever(search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["authentication", "oauth", "login"]}
        result = retriever.retrieve("doc_test", query_plan)

        candidate_page_numbers = [c["page_number"] for c in result["candidate_pages"]]
        # Exactly pages 5, 3, 7 (no duplicates)
        self.assertEqual(len(candidate_page_numbers), 3)
        self.assertEqual(set(candidate_page_numbers), {3, 5, 7})

    # --- Test 4: Pages matching more keywords receive higher scores ---
    def test_pages_matching_more_keywords_receive_higher_scores(self):
        fake = FakeDocumentTools(search_map={
            "authentication": [3, 5],
            "oauth": [5, 7],
            "login": [3, 5]
        })
        retriever = PageRetriever(search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["authentication", "oauth", "login"]}
        result = retriever.retrieve("doc_test", query_plan)

        candidates = result["candidate_pages"]
        # Expected scores: page 5 -> 3, page 3 -> 2, page 7 -> 1
        self.assertEqual(candidates[0]["page_number"], 5)
        self.assertEqual(candidates[0]["score"], 3)
        self.assertEqual(candidates[1]["page_number"], 3)
        self.assertEqual(candidates[1]["score"], 2)
        self.assertEqual(candidates[2]["page_number"], 7)
        self.assertEqual(candidates[2]["score"], 1)

    # --- Test 5: Ties are resolved deterministically by page number ---
    def test_ties_resolved_deterministically_by_page_number(self):
        # All pages match exactly 1 keyword (score = 1)
        fake = FakeDocumentTools(search_map={"keyword": [9, 2, 7, 4]})
        retriever = PageRetriever(search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["keyword"]}
        result = retriever.retrieve("doc_test", query_plan)

        candidates = result["candidate_pages"]
        pages = [c["page_number"] for c in candidates]
        # Ascending page numbers for ties
        self.assertEqual(pages, [2, 4, 7, 9])

    # --- Test 6: Search calls count toward the six-call budget ---
    def test_search_calls_count_toward_budget(self):
        fake = FakeDocumentTools(search_map={"k1": [], "k2": [], "k3": []})
        retriever = PageRetriever(max_budget=6, search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["k1", "k2", "k3"]}
        result = retriever.retrieve("doc_test", query_plan)

        # 3 searches, 0 page retrievals
        self.assertEqual(result["tool_calls_used"], 3)
        self.assertEqual(len(fake.search_calls), 3)
        self.assertEqual(len(fake.get_page_calls), 0)

    # --- Test 7: Page retrieval calls count toward the six-call budget ---
    def test_page_retrieval_calls_count_toward_budget(self):
        fake = FakeDocumentTools(search_map={"k1": [1, 2]})
        retriever = PageRetriever(max_budget=6, search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["k1"]}
        result = retriever.retrieve("doc_test", query_plan)

        # 1 search + 2 page gets = 3 calls
        self.assertEqual(result["tool_calls_used"], 3)
        self.assertEqual(len(fake.get_page_calls), 2)
        self.assertEqual(len(result["retrieved_pages"]), 2)

    # --- Test 8: The retriever never exceeds six total calls ---
    def test_retriever_never_exceeds_six_total_calls(self):
        # 3 keywords, candidate pages = 10 pages
        fake = FakeDocumentTools(search_map={
            "k1": [1, 2, 3, 4],
            "k2": [5, 6, 7],
            "k3": [8, 9, 10]
        })
        retriever = PageRetriever(max_budget=6, search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["k1", "k2", "k3"]}
        result = retriever.retrieve("doc_test", query_plan)

        # 3 searches executed, leaves 3 calls for retrieval: 3 + 3 = 6
        self.assertLessEqual(result["tool_calls_used"], 6)
        self.assertEqual(result["tool_calls_used"], 6)
        self.assertEqual(len(result["retrieved_pages"]), 3)

    # --- Test 9: If all six calls are consumed by searches, no page retrieval should occur ---
    def test_no_page_retrieval_when_all_calls_consumed_by_searches(self):
        # 6 keywords, budget = 6
        fake = FakeDocumentTools(search_map={f"k{i}": [i] for i in range(1, 7)})
        retriever = PageRetriever(max_budget=6, search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["k1", "k2", "k3", "k4", "k5", "k6"]}
        result = retriever.retrieve("doc_test", query_plan)

        self.assertEqual(result["tool_calls_used"], 6)
        self.assertEqual(len(result["candidate_pages"]), 6)
        # 0 budget remaining, so 0 page gets
        self.assertEqual(len(result["retrieved_pages"]), 0)
        self.assertEqual(len(fake.get_page_calls), 0)

    # --- Test 10: If only two calls remain, only two pages can be retrieved ---
    def test_only_two_pages_retrieved_when_two_calls_remain(self):
        # Budget = 4, 2 keywords searched (consumes 2, leaves 2)
        fake = FakeDocumentTools(search_map={
            "k1": [10, 20],
            "k2": [30, 40]
        })
        retriever = PageRetriever(max_budget=4, search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["k1", "k2"]}
        result = retriever.retrieve("doc_test", query_plan)

        # 2 searches + 2 page gets = 4 calls total
        self.assertEqual(result["tool_calls_used"], 4)
        self.assertEqual(len(result["retrieved_pages"]), 2)
        # Top 2 candidates were retrieved (10 and 20 by tie-breaking)
        retrieved_page_nums = [p["page_number"] for p in result["retrieved_pages"]]
        self.assertEqual(retrieved_page_nums, [10, 20])

    # --- Test 11: Empty search results are handled correctly ---
    def test_empty_search_results_handled_correctly(self):
        fake = FakeDocumentTools(search_map={})
        retriever = PageRetriever(max_budget=6, search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["nonexistent"]}
        result = retriever.retrieve("doc_test", query_plan)

        self.assertEqual(result["candidate_pages"], [])
        self.assertEqual(result["retrieved_pages"], [])
        self.assertEqual(result["tool_calls_used"], 1)

    # --- Test 12: Tool failures are handled safely ---
    def test_tool_failures_handled_safely(self):
        # Case 12a: One keyword search fails, another succeeds
        fake = FakeDocumentTools(search_map={"working": [4]})
        fake.fail_keywords.add("broken")
        retriever = PageRetriever(max_budget=6, search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["broken", "working"]}
        result = retriever.retrieve("doc_test", query_plan)

        # "broken" consumed 1 call and failed, "working" consumed 1 call and found page 4
        # page 4 retrieved (1 call) -> total 3 calls
        self.assertEqual(result["tool_calls_used"], 3)
        self.assertEqual(len(result["candidate_pages"]), 1)
        self.assertEqual(result["candidate_pages"][0]["page_number"], 4)
        self.assertEqual(len(result["retrieved_pages"]), 1)

        # Case 12b: get_page fails for a specific page
        fake_page_fail = FakeDocumentTools(search_map={"kw": [1, 2]})
        fake_page_fail.fail_pages.add(1)
        retriever_pf = PageRetriever(max_budget=6, search_fn=fake_page_fail.search_keyword, get_page_fn=fake_page_fail.get_page)

        res_pf = retriever_pf.retrieve("doc_test", {"keywords": ["kw"]})
        # Did not crash, retrieved page 2 successfully
        self.assertEqual(len(res_pf["retrieved_pages"]), 1)
        self.assertEqual(res_pf["retrieved_pages"][0]["page_number"], 2)

    # --- Test 13: Relevance R(p,q), Probability P(p|q), and Entropy ---
    def test_relevance_probability_and_entropy_calculation(self):
        fake = FakeDocumentTools(search_map={
            "auth": [1, 2],
            "oauth": [2, 3],
        })
        retriever = PageRetriever(search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["auth", "oauth"]}
        result = retriever.retrieve("doc_test", query_plan)

        candidates = result["candidate_pages"]
        # Total scores: page 2 -> 2, page 1 -> 1, page 3 -> 1. Total relevance = 4.
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["page_number"], 2)
        self.assertEqual(candidates[0]["relevance"], 2)
        self.assertEqual(candidates[0]["probability"], 0.5)

        self.assertEqual(candidates[1]["page_number"], 1)
        self.assertEqual(candidates[1]["relevance"], 1)
        self.assertEqual(candidates[1]["probability"], 0.25)

        self.assertEqual(candidates[2]["page_number"], 3)
        self.assertEqual(candidates[2]["relevance"], 1)
        self.assertEqual(candidates[2]["probability"], 0.25)

        # Probabilities sum to 1.0
        prob_sum = sum(c["probability"] for c in candidates)
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

        # Shannon Entropy H(P) = -(0.5*log2(0.5) + 0.25*log2(0.25) + 0.25*log2(0.25))
        # H(P) = -(0.5*(-1) + 0.25*(-2) + 0.25*(-2)) = -(-0.5 - 0.5 - 0.5) = 1.5
        self.assertAlmostEqual(result["entropy"], 1.5, places=2)

    # --- Test 14: Entropy is zero when single page matches (zero uncertainty) ---
    def test_entropy_zero_when_single_candidate(self):
        fake = FakeDocumentTools(search_map={"unique": [42]})
        retriever = PageRetriever(search_fn=fake.search_keyword, get_page_fn=fake.get_page)

        query_plan = {"keywords": ["unique"]}
        result = retriever.retrieve("doc_test", query_plan)

        self.assertEqual(len(result["candidate_pages"]), 1)
        self.assertEqual(result["candidate_pages"][0]["probability"], 1.0)
        self.assertEqual(result["entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
