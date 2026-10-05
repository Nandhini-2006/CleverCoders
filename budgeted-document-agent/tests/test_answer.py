import pytest
from unittest.mock import MagicMock, patch

from app.agent.answer import AnswerGenerator, generate_final_answer
from app.agent.planner import MockLLMClient


class TestAnswerGenerator:
    """Test suite for Phase 7 Final Answer Generator."""

    def test_strong_evidence_produces_answer(self):
        """Test 1: Strong evidence produces an answered status with valid citations."""
        evidence_state = {
            "status": "sufficient_evidence",
            "confidence": 0.88,
            "claims": [
                {
                    "claim": "SAML 2.0 is supported for enterprise single sign-on",
                    "supporting_pages": [12],
                }
            ],
            "contradictions": [],
            "unresolved_contradictions": [],
        }
        retrieved_pages = [
            {"page_number": 12, "text": "SAML 2.0 is supported for enterprise single sign-on."}
        ]

        result = generate_final_answer(
            question="What SSO authentication is supported?",
            evidence_state=evidence_state,
            retrieved_pages=retrieved_pages,
            confidence=0.88,
        )

        assert result["status"] == "answered"
        assert "SAML 2.0 is supported" in result["answer"]
        assert 12 in result["sources"]
        assert result["groundedness"] >= 0.80

    def test_weak_evidence_produces_insufficient_information(self):
        """Test 2: Weak evidence (confidence < 0.70) produces insufficient_information."""
        evidence_state = {
            "status": "insufficient_information",
            "confidence": 0.40,
            "claims": [],
            "contradictions": [],
            "unresolved_contradictions": [],
        }
        retrieved_pages = [
            {"page_number": 3, "text": "General overview of the system architecture."}
        ]

        result = generate_final_answer(
            question="What is the root password of the production cluster?",
            evidence_state=evidence_state,
            retrieved_pages=retrieved_pages,
            confidence=0.40,
        )

        assert result["status"] == "insufficient_information"
        assert result["answer"] == "Insufficient information."
        assert 3 in result["sources"]

    def test_superseding_evidence_produces_current_answer(self):
        """Test 3: Superseding evidence produces the current answer and cites winning page."""
        evidence_state = {
            "status": "sufficient_evidence",
            "confidence": 0.85,
            "claims": [
                {
                    "claim": "OAuth is used.",
                    "supporting_pages": [5],
                },
                {
                    "claim": "Effective July 1, 2026, OAuth is replaced by SAML.",
                    "supporting_pages": [12],
                },
            ],
            "contradictions": [
                {
                    "type": "temporal_superseding",
                    "status": "superseded",
                    "winning_page": 12,
                    "superseding_claim": "Effective July 1, 2026, OAuth is replaced by SAML.",
                    "claims": [
                        {"page": 5, "text": "OAuth is used."},
                        {"page": 12, "text": "Effective July 1, 2026, OAuth is replaced by SAML."},
                    ],
                }
            ],
            "unresolved_contradictions": [],
        }
        retrieved_pages = [
            {"page_number": 5, "text": "OAuth is used."},
            {"page_number": 12, "text": "Effective July 1, 2026, OAuth is replaced by SAML."},
        ]

        result = generate_final_answer(
            question="What authentication method is currently supported?",
            evidence_state=evidence_state,
            retrieved_pages=retrieved_pages,
            confidence=0.85,
        )

        assert result["status"] == "answered"
        assert "SAML" in result["answer"]
        assert 12 in result["sources"]
        assert 5 not in result["sources"]

    def test_unresolved_contradiction_not_arbitrarily_resolved(self):
        """Test 4: Unresolved contradiction explains the conflict and does not pick one arbitrarily."""
        evidence_state = {
            "status": "unresolved_contradiction",
            "confidence": 0.35,
            "claims": [
                {"claim": "Session timeout is 15 minutes.", "supporting_pages": [5]},
                {"claim": "Session timeout is 60 minutes.", "supporting_pages": [8]},
            ],
            "contradictions": [
                {
                    "status": "unresolved",
                    "claims": [
                        {"page": 5, "text": "Session timeout is 15 minutes."},
                        {"page": 8, "text": "Session timeout is 60 minutes."},
                    ],
                }
            ],
            "unresolved_contradictions": [
                {
                    "status": "unresolved",
                    "claims": [
                        {"page": 5, "text": "Session timeout is 15 minutes."},
                        {"page": 8, "text": "Session timeout is 60 minutes."},
                    ],
                }
            ],
        }
        retrieved_pages = [
            {"page_number": 5, "text": "Session timeout is 15 minutes."},
            {"page_number": 8, "text": "Session timeout is 60 minutes."},
        ]

        result = generate_final_answer(
            question="What is the session timeout?",
            evidence_state=evidence_state,
            retrieved_pages=retrieved_pages,
            confidence=0.35,
        )

        assert result["status"] == "unresolved_contradiction"
        assert "conflicting information" in result["answer"].lower()
        assert "Page 5" in result["answer"]
        assert "Page 8" in result["answer"]
        assert 5 in result["sources"]
        assert 8 in result["sources"]

    def test_page_citations_are_preserved(self):
        """Test 5: Page citations are accurately retained without hallucinated page numbers."""
        evidence_state = {
            "status": "sufficient_evidence",
            "confidence": 0.90,
            "claims": [
                {"claim": "Database backups run nightly at 2 AM.", "supporting_pages": [7, 14]}
            ],
            "contradictions": [],
            "unresolved_contradictions": [],
        }
        retrieved_pages = [
            {"page_number": 7, "text": "Database backups run nightly at 2 AM."},
            {"page_number": 14, "text": "Database backups run nightly at 2 AM."},
        ]

        result = generate_final_answer(
            question="When do database backups run?",
            evidence_state=evidence_state,
            retrieved_pages=retrieved_pages,
            confidence=0.90,
        )

        assert result["status"] == "answered"
        assert set(result["sources"]) == {7, 14}

    def test_phase7_makes_zero_additional_document_tool_calls(self):
        """Test 11: Phase 7 makes zero calls to document tools."""
        with patch("app.tools.list_documents") as mock_list_docs, \
             patch("app.tools.list_headings") as mock_list_heads, \
             patch("app.tools.search_keyword") as mock_search_kw, \
             patch("app.tools.get_page") as mock_get_page:

            evidence_state = {
                "status": "sufficient_evidence",
                "confidence": 0.85,
                "claims": [{"claim": "Encryption standard is AES-256.", "supporting_pages": [4]}],
                "contradictions": [],
                "unresolved_contradictions": [],
            }
            retrieved_pages = [{"page_number": 4, "text": "Encryption standard is AES-256."}]

            generator = AnswerGenerator()
            res = generator.generate(
                question="What encryption is used?",
                evidence_state=evidence_state,
                retrieved_pages=retrieved_pages,
            )

            assert res["status"] == "answered"

            # Verify ZERO tool calls were made
            mock_list_docs.assert_not_called()
            mock_list_heads.assert_not_called()
            mock_search_kw.assert_not_called()
            mock_get_page.assert_not_called()

    def test_answer_generation_with_mock_llm(self):
        """Test answer generator with a mock LLM client."""
        mock_llm = MockLLMClient({
            "What is the TLS version?": "The system requires TLS 1.3 for all secure endpoints."
        })

        evidence_state = {
            "status": "sufficient_evidence",
            "confidence": 0.90,
            "claims": [{"claim": "The system requires TLS 1.3 for all secure endpoints.", "supporting_pages": [9]}],
            "contradictions": [],
            "unresolved_contradictions": [],
        }
        retrieved_pages = [
            {"page_number": 9, "text": "The system requires TLS 1.3 for all secure endpoints."}
        ]

        generator = AnswerGenerator(llm_client=mock_llm)
        result = generator.generate(
            question="What is the TLS version?",
            evidence_state=evidence_state,
            retrieved_pages=retrieved_pages,
            confidence=0.90,
        )

        assert result["status"] == "answered"
        assert "TLS 1.3" in result["answer"]
        assert 9 in result["sources"]

    def test_unsupported_claims_revised_by_answer_generator(self):
        """Test that draft answers containing unsupported claims are automatically revised."""
        # LLM returns an answer with a hallucinated/unsupported claim
        hallucinating_llm = MockLLMClient({
            "What authentication": "SAML is the current authentication method and is 50% faster than OAuth."
        })

        evidence_state = {
            "status": "sufficient_evidence",
            "confidence": 0.85,
            "claims": [{"claim": "OAuth was replaced by SAML.", "supporting_pages": [12]}],
            "contradictions": [],
            "unresolved_contradictions": [],
        }
        retrieved_pages = [
            {"page_number": 12, "text": "OAuth was replaced by SAML."}
        ]

        generator = AnswerGenerator(llm_client=hallucinating_llm)
        result = generator.generate(
            question="What authentication method is currently supported?",
            evidence_state=evidence_state,
            retrieved_pages=retrieved_pages,
            confidence=0.85,
        )

        assert result["status"] == "answered"
        assert "50% faster" not in result["answer"]
        assert "SAML" in result["answer"]
        assert 12 in result["sources"]

    def test_validation_failed_status(self):
        """Test status becomes validation_failed when groundedness fails and cannot be revised."""
        failing_mock = MagicMock()
        failing_mock.generate.return_value = "Completely unsupported statement."

        generator = AnswerGenerator()
        # Mock validate_answer to simulate persistent validation failure
        from app.agent.validator import ValidationResult
        with patch("app.agent.answer.validate_answer") as mock_val:
            mock_val.return_value = ValidationResult(
                valid=False,
                groundedness=0.2,
                unsupported_claims=["Completely unsupported statement."],
                supported_claims=[],
            )
            with patch("app.agent.answer.revise_answer") as mock_rev:
                mock_rev.return_value = "Still unsupported."
                result = generator.generate(
                    question="What is the database?",
                    evidence_state={
                        "status": "sufficient_evidence",
                        "confidence": 0.85,
                        "claims": [{"claim": "Database is Postgres.", "supporting_pages": [1]}],
                    },
                )

        assert result["status"] == "validation_failed"
        assert result["groundedness"] == 0.2

