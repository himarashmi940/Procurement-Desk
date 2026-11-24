"""In-memory adapter between Streamlit and the Procurement Desk engine."""

import re
from typing import Any, Optional, Tuple

import pandas as pd

from src.data_loader import load_threads
from src.decision_engine import decide_label


RESULT_COLUMNS = ("thread_id", "label", "action", "reason", "counter_price")
_VALID_REASONS = {
    "ACCEPT": {"ACCEPTABLE_PRICE"},
    "REJECT": {"BLACKLISTED_SUPPLIER", "ABOVE_WALKAWAY"},
    "COUNTER": {"COUNTER_OFFER"},
    "ESCALATE": {
        "MISSING_INFO",
        "PROMPT_INJECTION",
        "BANNED_TERM",
        "OVER_APPROVAL_LIMIT",
    },
}


def _parse_decision_label(
    label: Any,
) -> Tuple[str, str, Optional[int]]:
    """Split a known engine label, marking malformed values for review."""
    invalid_result = ("UNKNOWN", "INVALID_LABEL", None)
    if not isinstance(label, str):
        return invalid_result

    parts = label.split(":")
    if len(parts) != 3:
        return invalid_result

    action, reason, counter_text = parts
    if reason not in _VALID_REASONS.get(action, set()):
        return invalid_result

    if action == "COUNTER":
        if not re.fullmatch(r"[0-9]+", counter_text):
            return invalid_result
        return action, reason, int(counter_text)

    if counter_text:
        return invalid_result
    return action, reason, None


def _analyze_threads(threads: pd.DataFrame) -> pd.DataFrame:
    """Return parsed decisions for the supplied thread records."""
    result_rows = []
    for row in threads.to_dict(orient="records"):
        label = decide_label(row)
        action, reason, counter_price = _parse_decision_label(label)
        result_rows.append(
            {
                "thread_id": row.get("thread_id"),
                "label": label,
                "action": action,
                "reason": reason,
                "counter_price": counter_price,
            }
        )

    results = pd.DataFrame.from_records(result_rows, columns=RESULT_COLUMNS)
    results["counter_price"] = pd.array(results["counter_price"], dtype="Int64")
    return results


def run_analysis() -> pd.DataFrame:
    """Load supplier threads and return decisions without writing output files."""
    return _analyze_threads(load_threads())


def run_analysis_with_threads() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return decisions and original thread records from a single dataset load."""
    threads = load_threads()
    results = _analyze_threads(threads)
    return results, threads.copy(deep=True)