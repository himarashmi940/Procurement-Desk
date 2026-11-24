"""Generate and validate the Procurement_Desk Kaggle submission."""

import re
from typing import Dict

import pandas as pd

from .data_loader import PROJECT_ROOT, load_threads
from .decision_engine import decide_label


EXPECTED_THREAD_COUNT = 2000
SUBMISSION_COLUMNS = ("thread_id", "label")
ACTION_ORDER = ("ACCEPT", "REJECT", "COUNTER", "ESCALATE")
NON_COUNTER_LABELS = {
    "ACCEPT:ACCEPTABLE_PRICE:",
    "REJECT:BLACKLISTED_SUPPLIER:",
    "REJECT:ABOVE_WALKAWAY:",
    "ESCALATE:MISSING_INFO:",
    "ESCALATE:PROMPT_INJECTION:",
    "ESCALATE:BANNED_TERM:",
    "ESCALATE:OVER_APPROVAL_LIMIT:",
}
COUNTER_LABEL_PATTERN = re.compile(r"^COUNTER:COUNTER_OFFER:\d+$")


def _validate_submission(submission: pd.DataFrame) -> None:
    """Raise a clear error if the generated submission violates the output contract."""
    if len(submission) != EXPECTED_THREAD_COUNT:
        raise ValueError(
            f"Submission must contain exactly {EXPECTED_THREAD_COUNT} rows; "
            f"found {len(submission)}."
        )
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            "Submission columns must be exactly "
            f"{list(SUBMISSION_COLUMNS)} in that order."
        )
    if submission["thread_id"].isna().any():
        raise ValueError("Submission contains missing thread IDs.")
    if submission["label"].isna().any():
        raise ValueError("Submission contains missing labels.")

    duplicate_ids = submission.loc[
        submission["thread_id"].duplicated(keep=False), "thread_id"
    ].unique()
    if len(duplicate_ids):
        raise ValueError(
            "Submission contains duplicate thread IDs: "
            + ", ".join(str(thread_id) for thread_id in duplicate_ids)
        )

    invalid_labels = [
        label
        for label in submission["label"].astype(str)
        if label not in NON_COUNTER_LABELS
        and not COUNTER_LABEL_PATTERN.fullmatch(label)
    ]
    if invalid_labels:
        raise ValueError(
            "Submission contains labels with an invalid format: "
            + ", ".join(invalid_labels[:5])
        )


def _action_counts(submission: pd.DataFrame) -> Dict[str, int]:
    actions = submission["label"].str.split(":", n=1).str[0]
    counts = actions.value_counts()
    return {action: int(counts.get(action, 0)) for action in ACTION_ORDER}


def main() -> None:
    """Predict all threads, validate the submission, and save it under outputs/."""
    threads = load_threads()
    if len(threads) != EXPECTED_THREAD_COUNT:
        raise ValueError(
            f"threads.csv must contain exactly {EXPECTED_THREAD_COUNT} rows; "
            f"found {len(threads)}."
        )

    rows = threads.to_dict(orient="records")
    predictions = [decide_label(row) for row in rows]
    submission = pd.DataFrame(
        {
            "thread_id": threads["thread_id"].tolist(),
            "label": predictions,
        },
        columns=SUBMISSION_COLUMNS,
    )
    _validate_submission(submission)

    output_path = PROJECT_ROOT / "outputs" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(submission)} predictions to {output_path}")
    print("Action counts:")
    for action, count in _action_counts(submission).items():
        print(f"  {action}: {count}")


if __name__ == "__main__":
    main()