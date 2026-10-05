from abc import ABC, abstractmethod
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Union
from dotenv import load_dotenv
from openai import OpenAI

from app.agent.prompts import build_planner_prompt
from app.agent.validator import (
    validate_query_plan,
    parse_query_plan,
    PlanValidationError,
    extract_question_keywords,
    clean_and_filter_keywords,
    is_keyword_in_question,
    extract_technical_phrases,
    normalize_terms,
    build_fallback_retrieval_plan,
    DEFAULT_PLANNER_STOPWORDS,
    DEFAULT_GENERIC_WORDS,
)


class QueryPlan(dict):
    """
    Structured Query Plan and Retrieval Plan returned by the QueryPlanner.
    Subclasses dict so it can be serialized, indexed like a dictionary,
    or accessed via object properties.
    """
    def __init__(
        self,
        question_type: str,
        keywords: List[str],
        search_query: Optional[str] = None,
        query: Optional[str] = None,
        synonyms: Optional[List[str]] = None,
        phrases: Optional[List[str]] = None,
        normalized_terms: Optional[List[str]] = None,
        retrieval_strategy: Optional[List[str]] = None,
        error: Optional[str] = None,
        important_concepts: Optional[List[str]] = None,
        requested_entities: Optional[List[str]] = None,
        requested_attributes: Optional[List[str]] = None,
        expected_answer_structure: Optional[str] = None,
    ):
        sq = search_query or query or (" ".join(keywords) if keywords else "")
        default_strat = [
            "exact_phrase",
            "aho_corasick",
            "stemming",
            "lemmatization",
            "ngram",
            "levenshtein",
            "bm25",
        ]
        super().__init__(
            question_type=question_type,
            search_query=sq,
            query=sq,
            keywords=keywords,
            synonyms=synonyms or [],
            phrases=phrases or [],
            normalized_terms=normalized_terms or [],
            retrieval_strategy=retrieval_strategy or default_strat,
            important_concepts=important_concepts or (keywords.copy() if keywords else []),
            requested_entities=requested_entities or (keywords[:1] if keywords else []),
            requested_attributes=requested_attributes or [],
            expected_answer_structure=expected_answer_structure or "direct answer",
        )
        if error:
            self["error"] = error

    @property
    def question_type(self) -> str:
        return self["question_type"]

    @property
    def search_query(self) -> str:
        return self.get("search_query", self.get("query", ""))

    @property
    def query(self) -> str:
        return self.search_query

    @property
    def keywords(self) -> List[str]:
        return self["keywords"]

    @property
    def synonyms(self) -> List[str]:
        return self.get("synonyms", [])

    @property
    def phrases(self) -> List[str]:
        return self.get("phrases", [])

    @property
    def normalized_terms(self) -> List[str]:
        return self.get("normalized_terms", [])

    @property
    def retrieval_strategy(self) -> List[str]:
        return self.get("retrieval_strategy", [])

    @property
    def important_concepts(self) -> List[str]:
        return self.get("important_concepts", [])

    @property
    def requested_entities(self) -> List[str]:
        return self.get("requested_entities", [])

    @property
    def requested_attributes(self) -> List[str]:
        return self.get("requested_attributes", [])

    @property
    def expected_answer_structure(self) -> str:
        return self.get("expected_answer_structure", "direct answer")

    @property
    def error(self) -> Optional[str]:
        return self.get("error")


class BaseLLMClient(ABC):
    """
    Pluggable LLM client abstraction.
    Allows replacing providers (NVIDIA Nemotron, OpenAI, etc.)
    without changing the query planning logic.
    """
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate completion text from the prompt.
        """
        pass


class NvidiaNemotronClient(BaseLLMClient):
    """
    NVIDIA Nemotron LLM Client accessing nvidia/nemotron-3.5-lightning-30b-a3b
    through NVIDIA's OpenAI-compatible API endpoint.

    Configuration:
    - Base URL: https://integrate.api.nvidia.com/v1
    - Model: nvidia/nemotron-3.5-lightning-30b-a3b
    - API Key: Loaded securely from .env via python-dotenv (NVIDIA_API_KEY).
    - Never hardcodes or logs the API key.
    - Low reasoning budget (256 tokens) suitable for lightweight query understanding.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 512,
        reasoning_budget: int = 256,
        timeout: float = 30.0,
    ):
        load_dotenv()
        self.api_key = api_key if api_key is not None else os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_budget = reasoning_budget
        self.timeout = timeout
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if not self.api_key or not str(self.api_key).strip():
            raise ValueError("Missing NVIDIA_API_KEY. Please set NVIDIA_API_KEY in .env.")
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=str(self.api_key).strip(),
                timeout=self.timeout,
            )
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": True
            },
            "reasoning_budget": self.reasoning_budget
        }

        _EMPTY_OUTPUT_PHRASES = (
            "model output must contain either output text or tool calls",
            "output text or tool calls, these cannot both be empty",
            "both be empty",
        )

        def _is_empty_output_error(exc: Exception) -> bool:
            msg = str(exc).lower()
            return any(phrase in msg for phrase in _EMPTY_OUTPUT_PHRASES)

        def _strip_thinking_preamble(text: str) -> str:
            """
            Remove chain-of-thought / thinking preamble that Nemotron leaks
            into the visible content when enable_thinking=True.

            Handles patterns like:
              - <think>...</think>
              - Here's a thinking process: ...
              - 1. Analyze User Question: ...  (numbered reasoning)
              - **Thinking:** ...
            """
            import re as _re

            # 1. Strip XML-style thinking tags (the standard Nemotron pattern)
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()

            # 2. Strip "Here's a thinking process:" preamble up to first blank line
            #    or until a line that looks like the actual answer begins
            thinking_header = _re.compile(
                r"^(?:here(?:'s| is)(?: a| my)? (?:thinking|thought|reasoning) (?:process|steps?)?[:\s]*|"
                r"\*\*(?:thinking|reasoning|thought process)\*\*[:\s]*|"
                r"let me (?:think|reason|analyze)[:\s]*)",
                _re.IGNORECASE,
            )

            lines = text.splitlines()
            if lines and thinking_header.match(lines[0].strip()):
                # Skip preamble block: drop lines until we hit real answer prose.
                # Real answer prose: non-empty, not a numbered step, not a bullet,
                # long enough to be a sentence (>=30 chars), or starts with JSON {
                cutoff = 1
                for i, line in enumerate(lines[1:], start=1):
                    stripped = line.strip()
                    if not stripped:
                        cutoff = i + 1
                        continue
                    # Still preamble: numbered step "1. ..."
                    if _re.match(r"^\d+\.\s+", stripped):
                        cutoff = i + 1
                        continue
                    # Still preamble: bullet "- ..." or "* ..."
                    if stripped[:2] in ("- ", "* ", "• "):
                        cutoff = i + 1
                        continue
                    # Real answer starts: JSON object or long prose sentence
                    if stripped.startswith("{") or len(stripped) >= 30:
                        cutoff = i
                        break

                text = "\n".join(lines[cutoff:]).strip()

            # 3. Strip any remaining leading numbered/bulleted preamble lines
            #    (e.g. "1. **Analyze User Question:**\n...")
            cleaned_lines = []
            in_preamble = True
            for line in text.splitlines():
                stripped = line.strip()
                if in_preamble and (
                    _re.match(r"^\d+\.\s+\*\*", stripped)
                    or _re.match(r"^[-*•]\s+\*\*", stripped)
                    or not stripped
                ):
                    continue
                in_preamble = False
                cleaned_lines.append(line)
            text = "\n".join(cleaned_lines).strip()

            return text

        def _call_without_extra_body() -> str:
            """Fallback call without thinking/extra_body params."""
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                raw = (resp.choices[0].message.content or "").strip()
                return _strip_thinking_preamble(raw)
            except Exception as inner_e:
                if _is_empty_output_error(inner_e):
                    return ""
                raise

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=False,
                extra_body=extra_body,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _strip_thinking_preamble(raw)
        except Exception as e:
            err_msg = str(e).lower()
            if _is_empty_output_error(e):
                return ""
            if any(kw in err_msg for kw in ("extra_body", "chat_template_kwargs", "reasoning_budget")):
                return _call_without_extra_body()
            raise


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM provider used for unit testing, offline operation,
    and deterministic hackathon prototyping without requiring external API keys.
    """
    def __init__(
        self,
        default_response: Optional[Union[str, Dict[str, Any]]] = None,
        stopwords: Optional[Set[str]] = None,
        generic_words: Optional[Set[str]] = None,
    ):
        self.default_response = default_response
        self.stopwords = stopwords
        self.generic_words = generic_words

    def generate(self, prompt: str) -> str:
        # If an explicit mock response is configured, return it directly
        if self.default_response is not None:
            if isinstance(self.default_response, dict):
                return json.dumps(self.default_response)
            return str(self.default_response)

        # Extract the user question line from the prompt
        matches = re.findall(r"User Question:\s*([^\r\n]+)", prompt)
        non_placeholders = [m.strip() for m in matches if "{{QUESTION}}" not in m]
        question = non_placeholders[-1] if non_placeholders else prompt.strip()
        q_lower = question.lower()

        # Purely generic linguistic question type classification
        if any(term in q_lower for term in ["history", "milestones", "timeline", "chronological", "evolution"]):
            q_type = "timeline"
            structure = "chronological list/summary"
        elif any(term in q_lower for term in ["compare", "versus", " vs ", "difference", "differences"]):
            q_type = "comparison"
            structure = "explicit comparison"
        elif any(term in q_lower for term in ["how to", "how do", "how does", "how can", "steps", "procedure", "guide", "algorithm", "process", "connect"]):
            q_type = "procedural"
            structure = "steps/process"
        elif any(term in q_lower for term in ["what is the formula", "formula for", "equation"]):
            q_type = "lookup"
            structure = "direct formula"
        elif any(term in q_lower for term in ["define", "definition", "what is meant by"]):
            q_type = "definition"
            structure = "definition"
        elif any(term in q_lower for term in ["why", "cause", "reason"]):
            q_type = "why/causal"
            structure = "explanation"
        elif any(term in q_lower for term in ["list", "enumerate", "name all"]):
            q_type = "list"
            structure = "bulleted list"
        elif any(term in q_lower for term in ["summarize", "summary"]):
            q_type = "summary"
            structure = "summary"
        elif any(term in q_lower for term in ["mention", "lookup", "find", "where"]):
            q_type = "lookup"
            structure = "direct lookup"
        elif any(term in q_lower for term in ["what", "which", "who", "when", "is", "are", "does", "did"]):
            q_type = "factual"
            structure = "direct answer"
        else:
            q_type = "other"
            structure = "direct answer"

        # Information-bearing keyword and phrase extraction
        keywords = extract_question_keywords(
            question,
            stopwords=self.stopwords,
            generic_words=self.generic_words,
            max_keywords=3,
        )
        extracted_phrases = extract_technical_phrases(question, max_phrases=5)
        normalized_keywords = normalize_terms(keywords + extracted_phrases)

        retrieval_strategy = [
            "exact_phrase",
            "aho_corasick",
            "stemming",
            "lemmatization",
            "ngram",
            "levenshtein",
            "bm25",
        ]
        if any(c in question for c in "*+-/=<>#0123456789") or q_type == "formula":
            retrieval_strategy.insert(2, "regex")

        # Academic retrieval synonyms for keywords
        synonyms = []
        for kw in keywords:
            kl = kw.lower()
            if kl in ("cost", "priority"):
                synonyms.extend(["score", "weight", "evaluation"])
            elif kl in ("authentication", "credentials"):
                synonyms.extend(["authorization", "login", "identity"])
            elif kl in ("roadmap", "graph"):
                synonyms.extend(["network", "nodes", "milestones"])
            elif kl in ("robot", "agent"):
                synonyms.extend(["manipulator", "system", "actor"])

        return json.dumps({
            "question_type": q_type,
            "search_query": question.strip(),
            "query": question.strip(),
            "keywords": keywords,
            "synonyms": synonyms[:15],
            "phrases": extracted_phrases,
            "normalized_terms": normalized_keywords,
            "retrieval_strategy": retrieval_strategy,
            "important_concepts": keywords,
            "requested_entities": keywords[:1],
            "requested_attributes": [],
            "expected_answer_structure": structure,
        })


class QueryPlanner:
    """
    Converts a natural language user question into a validated query plan
    with classified question type and at most 3 targeted keywords.
    Uses NVIDIA Nemotron through NVIDIA's OpenAI-compatible API.
    """
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        stopwords: Optional[Set[str]] = None,
        generic_words: Optional[Set[str]] = None,
    ):
        load_dotenv()
        self.stopwords = stopwords
        self.generic_words = generic_words
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # Default to NVIDIA Nemotron client
            api_key = os.getenv("NVIDIA_API_KEY")
            if api_key and api_key.strip():
                self.llm_client = NvidiaNemotronClient(api_key=api_key.strip())
            else:
                self.llm_client = NvidiaNemotronClient(api_key=None)

    def plan(self, question: str) -> QueryPlan:
        """
        Plan search keywords and deterministic retrieval strategy for a user question.

        Architecture rules:
        - NEVER passes PDF content to the LLM.
        - NEVER calls any document tool.
        - Receives ONLY the user question.
        - Returns a validated QueryPlan object.
        - Safe Fallback: If LLM fails, times out, returns malformed output or empty plan,
          falls back to deterministic technical phrase, keyword, and normalized retrieval plan.
          Never allows planner failure to result in zero retrieval attempts.
        """
        if not question or not question.strip():
            raise ValueError("User question cannot be empty.")

        # 1. Build prompt strictly containing the user question
        prompt = build_planner_prompt(question)

        # 2. Invoke pluggable LLM provider with fallback handling
        try:
            raw_output = self.llm_client.generate(prompt)
        except Exception as e:
            # Deterministic fallback: do not crash Streamlit
            fallback = build_fallback_retrieval_plan(question)
            return QueryPlan(
                question_type="other",
                keywords=[],
                search_query=question.strip(),
                query=question.strip(),
                synonyms=[],
                phrases=fallback.get("phrases", []),
                normalized_terms=fallback.get("normalized_terms", []),
                retrieval_strategy=fallback.get("retrieval_strategy", []),
                error=f"LLM generation failed: {str(e)}",
            )

        # 3. Deterministically parse and validate JSON output
        # Guard: if the LLM returned empty content, use the deterministic fallback
        if not raw_output or not raw_output.strip():
            fallback = build_fallback_retrieval_plan(question)
            return QueryPlan(
                question_type="other",
                keywords=[],
                search_query=question.strip(),
                query=question.strip(),
                synonyms=[],
                phrases=fallback.get("phrases", []),
                normalized_terms=fallback.get("normalized_terms", []),
                retrieval_strategy=fallback.get("retrieval_strategy", []),
                error="LLM returned empty output; using deterministic fallback.",
            )

        parsed = parse_query_plan(
            raw_output,
            question=question,
            stopwords=self.stopwords,
            generic_words=self.generic_words,
        )

        error_msg = None
        if not parsed.get("keywords") and parsed.get("question_type") == "other":
            if "{" not in str(raw_output):
                error_msg = "Malformed LLM output: not valid JSON."

        return QueryPlan(
            question_type=parsed["question_type"],
            search_query=parsed.get("search_query", question.strip()),
            query=parsed.get("query", question.strip()),
            keywords=parsed["keywords"],
            synonyms=parsed.get("synonyms", []),
            phrases=parsed.get("phrases", []),
            normalized_terms=parsed.get("normalized_terms", []),
            retrieval_strategy=parsed.get("retrieval_strategy", []),
            important_concepts=parsed.get("important_concepts", []),
            requested_entities=parsed.get("requested_entities", []),
            requested_attributes=parsed.get("requested_attributes", []),
            expected_answer_structure=parsed.get("expected_answer_structure", "direct answer"),
            error=error_msg,
        )


def plan_query(
    question: str,
    llm_client: Optional[BaseLLMClient] = None,
    stopwords: Optional[Set[str]] = None,
    generic_words: Optional[Set[str]] = None,
) -> QueryPlan:
    """
    Convenience function to run query planning on a user question.
    """
    planner = QueryPlanner(
        llm_client=llm_client,
        stopwords=stopwords,
        generic_words=generic_words,
    )
    return planner.plan(question)


