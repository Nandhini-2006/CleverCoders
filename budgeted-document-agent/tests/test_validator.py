import json
import pytest

from app.agent.validator import (
    validate_answer,
    revise_answer,
    split_into_verifiable_claims,
    GROUNDING_THRESHOLD,
    ValidationResult,
)
from app.agent.planner import MockLLMClient


class TestAnswerValidator:
    """Test suite for Phase 7 Answer Validator and Groundedness Verification."""

    def test_unsupported_claim_detected_example(self):
        """
        Test 6: Verify the exact prompt scenario:
        Evidence: Page 12: "OAuth was replaced by SAML."
        Draft answer: "SAML is the current authentication method and is 50% faster than OAuth."
        Validator identifies:
        - Supported: "SAML is the current authentication method."
        - Unsupported: "SAML is 50% faster than OAuth."
        - Groundedness: 1 / 2 = 0.5
        """
        question = "What authentication method is currently supported?"
        evidence = [{"page_number": 12, "text": "OAuth was replaced by SAML."}]
        draft = "SAML is the current authentication method and is 50% faster than OAuth."

        val = validate_answer(question, draft, evidence)

        assert val.valid is False
        assert val.groundedness == 0.5
        assert len(val.supported_claims) == 1
        assert "SAML is the current authentication method" in val.supported_claims[0]
        assert len(val.unsupported_claims) == 1
        assert "50% faster" in val.unsupported_claims[0]

    def test_groundedness_calculation_perfect_match(self):
        """Test 7a: Groundedness calculation on 100% grounded answer."""
        question = "What is the primary database?"
        evidence = "The primary database is PostgreSQL 15."
        draft = "The primary database is PostgreSQL 15."

        val = validate_answer(question, draft, evidence)

        assert val.valid is True
        assert val.groundedness == 1.0
        assert len(val.unsupported_claims) == 0
        assert len(val.supported_claims) == 1

    def test_groundedness_calculation_multiple_claims(self):
        """Test 7b: Groundedness calculation with multiple claims (2 supported, 1 unsupported = 2/3 = 0.67)."""
        question = "What are the storage specifications?"
        evidence = "Storage capacity is 100 TB. Storage encryption uses AES-256."
        draft = "Storage capacity is 100 TB. Storage encryption uses AES-256. Storage latency is under 1 ms."

        val = validate_answer(question, draft, evidence)

        # 2 supported, 1 unsupported -> 2 / 3 = 0.6667
        assert val.valid is False  # 0.67 < 0.80
        assert abs(val.groundedness - 0.6667) < 0.01
        assert len(val.supported_claims) == 2
        assert len(val.unsupported_claims) == 1

    def test_low_groundedness_triggers_revision(self):
        """Test 8: Low groundedness causes rejection/revision producing grounded output."""
        question = "What authentication method is currently supported?"
        evidence = "OAuth was replaced by SAML."
        draft = "SAML is the current authentication method and is 50% faster than OAuth."

        val = validate_answer(question, draft, evidence)
        assert val.valid is False

        revised = revise_answer(question, draft, evidence, validation_result=val)
        assert "50% faster" not in revised
        assert "SAML" in revised

        # Validate that the revised answer now passes
        val_revised = validate_answer(question, revised, evidence)
        assert val_revised.valid is True
        assert val_revised.groundedness >= GROUNDING_THRESHOLD

    def test_split_into_verifiable_claims(self):
        """Test splitting compound sentences into atomic claims."""
        compound = "SAML is the current authentication method and is 50% faster than OAuth."
        claims = split_into_verifiable_claims(compound)

        assert len(claims) == 2
        assert "SAML is the current authentication method" in claims[0]
        assert "SAML is 50% faster than OAuth" in claims[1]

    def test_empty_answer_fails_validation(self):
        """Test that empty or whitespace-only answer fails validation."""
        val = validate_answer("What is X?", "", "Some evidence.")
        assert val.valid is False
        assert val.groundedness == 0.0

    def test_insufficient_information_answer_is_grounded(self):
        """Test that 'Insufficient information.' is recognized as grounded."""
        val = validate_answer("What is X?", "Insufficient information.", "Unrelated text.")
        assert val.valid is True
        assert val.groundedness == 1.0

    def test_validator_with_mock_llm(self):
        """Test LLM-driven structured validation using MockLLMClient."""
        mock_response = {
            "valid": True,
            "groundedness": 0.95,
            "supported_claims": ["SAML is the current authentication method."],
            "unsupported_claims": [],
        }
        mock_llm = MockLLMClient(mock_response)

        val = validate_answer(
            question="What authentication method is used?",
            draft_answer="SAML is the current authentication method.",
            evidence="OAuth was replaced by SAML.",
            llm_client=mock_llm,
        )

        assert val.valid is True
        assert val.groundedness == 0.95
        assert len(val.supported_claims) == 1
