import os
import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf
from app.config import DOCUMENT_DIR
from app.tools.list_documents import list_documents
from app.tools.list_headings import list_headings
from app.tools.search_keyword import search_keyword
from app.tools.get_page import get_page


class TestDocumentTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create sample test PDFs in the documents folder."""
        cls.test_doc_id = "doc_testphase2"
        cls.test_pdf_path = DOCUMENT_DIR / f"{cls.test_doc_id}.pdf"

        # Create a 2-page PDF using PyMuPDF
        doc = pymupdf.open()

        # Page 1
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text(
            (50, 72),
            "1. Introduction to Budgeted Agents\nThis agent is strictly budgeted to 6 tool calls.\nEfficiency is essential for answering questions.",
            fontsize=12,
        )

        # Page 2
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text(
            (50, 72),
            "2. Financial Performance\nThe total company revenue was 42 million dollars.\nOperating expenses remained under budget.",
            fontsize=12,
        )

        # Add Table of Contents (bookmarks)
        # Format: [[level, title, page_1_indexed]]
        toc = [
            [1, "Introduction to Budgeted Agents", 1],
            [1, "Financial Performance", 2],
        ]
        doc.set_toc(toc)

        doc.save(str(cls.test_pdf_path))
        doc.close()

    @classmethod
    def tearDownClass(cls):
        """Clean up the test PDF."""
        if cls.test_pdf_path.exists():
            cls.test_pdf_path.unlink()

    # --- Tool 1: list_documents ---
    def test_list_documents(self):
        docs = list_documents()
        self.assertIsInstance(docs, list)
        self.assertTrue(any(d["doc_id"] == self.test_doc_id for d in docs))

        # Check document entry structure
        test_entry = next(d for d in docs if d["doc_id"] == self.test_doc_id)
        self.assertEqual(test_entry["total_pages"], 2)
        self.assertNotIn("file_path", test_entry, "File paths must not be leaked to the agent")

    # --- Tool 2: list_headings ---
    def test_list_headings_valid(self):
        headings = list_headings(self.test_doc_id)
        self.assertIsInstance(headings, list)
        self.assertEqual(len(headings), 2)

        # Check first heading
        self.assertEqual(headings[0]["title"], "Introduction to Budgeted Agents")
        self.assertEqual(headings[0]["page"], 1)
        self.assertEqual(headings[0]["level"], 1)

        # Check second heading
        self.assertEqual(headings[1]["title"], "Financial Performance")
        self.assertEqual(headings[1]["page"], 2)
        self.assertEqual(headings[1]["level"], 1)

    def test_list_headings_invalid_doc_id(self):
        with self.assertRaises(ValueError):
            list_headings("invalid_id")

        with self.assertRaises(FileNotFoundError):
            list_headings("doc_nonexistent")

    # --- Tool 3: search_keyword ---
    def test_search_keyword_found(self):
        matches = search_keyword(self.test_doc_id, "revenue")
        self.assertIsInstance(matches, list)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["page"], 2)
        self.assertGreaterEqual(matches[0]["match_count"], 1)
        self.assertIn("revenue", matches[0]["snippet"].lower())

    def test_search_keyword_case_insensitive(self):
        matches = search_keyword(self.test_doc_id, "REVENUE")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["page"], 2)

    def test_search_keyword_not_found(self):
        matches = search_keyword(self.test_doc_id, "cryptocurrency")
        self.assertEqual(matches, [])

    def test_search_keyword_empty(self):
        matches = search_keyword(self.test_doc_id, "")
        self.assertEqual(matches, [])

    # --- Tool 4: get_page ---
    def test_get_page_valid(self):
        page1 = get_page(self.test_doc_id, 1)
        self.assertEqual(page1["doc_id"], self.test_doc_id)
        self.assertEqual(page1["page_number"], 1)
        self.assertEqual(page1["total_pages"], 2)
        self.assertIn("Budgeted Agents", page1["text"])

        page2 = get_page(self.test_doc_id, 2)
        self.assertEqual(page2["page_number"], 2)
        self.assertIn("42 million", page2["text"])

    def test_get_page_string_and_dict_compatibility(self):
        page1 = get_page(self.test_doc_id, 1)
        # Compatible with string membership
        self.assertIn("budget", str(page1).lower())
        self.assertIn("Efficiency", page1)
        # Compatible with dictionary access
        self.assertIn("text", page1)
        self.assertIn("page_number", page1)

    def test_get_page_out_of_bounds(self):
        with self.assertRaises(ValueError):
            get_page(self.test_doc_id, 0)

        with self.assertRaises(ValueError):
            get_page(self.test_doc_id, 3)

    def test_get_page_invalid_doc_id(self):
        with self.assertRaises(ValueError):
            get_page("invalid_id", 1)


if __name__ == "__main__":
    unittest.main()
