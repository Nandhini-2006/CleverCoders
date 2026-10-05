import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default path for agent query traces
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
DEFAULT_TRACE_FILE = DEFAULT_LOG_DIR / "agent_traces.jsonl"

# Keys and patterns that must NEVER appear in logs
SENSITIVE_KEY_PATTERNS = [
    re.compile(r"api[-_]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
]

NVAPI_PATTERN = re.compile(r"nvapi-[A-Za-z0-9_-]{20,}")


def sanitize_data(obj: Any) -> Any:
    """
    Recursively sanitize data to guarantee no API keys, tokens,
    or sensitive secrets are recorded in logs.
    """
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            k_str = str(k)
            if any(p.search(k_str) for p in SENSITIVE_KEY_PATTERNS):
                sanitized[k] = "[REDACTED_SECRET]"
            else:
                sanitized[k] = sanitize_data(v)
        return sanitized
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_data(item) for item in obj]
    elif isinstance(obj, str):
        # Scrub any potential raw API keys embedded in strings
        cleaned = NVAPI_PATTERN.sub("[REDACTED_API_KEY]", obj)
        return cleaned
    elif hasattr(obj, "__dict__"):
        return sanitize_data(obj.__dict__)
    else:
        return obj


def log_query_trace(
    trace_data: Dict[str, Any],
    log_file: Optional[Path | str] = None,
) -> Path:
    """
    Append a structured execution trace for a question to the JSONL log file.

    Trace records include:
    - timestamp
    - question
    - question_type
    - keywords
    - selected_pages
    - tool_calls
    - entropy
    - information_gain
    - budget_before
    - budget_after
    - evidence
    - contradiction_state
    - confidence
    - groundedness
    - final_status
    """
    target_path = Path(log_file) if log_file else DEFAULT_TRACE_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure timestamp is set
    record = dict(trace_data)
    if "timestamp" not in record:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Sanitize everything recursively
    clean_record = sanitize_data(record)

    # Serialize to JSONL
    line = json.dumps(clean_record, ensure_ascii=False, default=str)
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    return target_path


def read_query_traces(log_file: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """
    Read all traces from the JSONL log file.
    """
    target_path = Path(log_file) if log_file else DEFAULT_TRACE_FILE
    if not target_path.exists():
        return []

    traces: List[Dict[str, Any]] = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    traces.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue
    return traces
