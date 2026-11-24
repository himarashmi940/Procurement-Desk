"""Load and inspect the Procurement_Desk CSV datasets."""

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

THREAD_COLUMNS = (
    "thread_id",
    "date",
    "supplier",
    "category",
    "quantity",
    "round",
    "our_last_counter",
    "supplier_message",
)
LABEL_COLUMNS = ("thread_id", "label")


def _load_csv(filename: str, required_columns: Iterable[str]) -> pd.DataFrame:
    """Read a project-root CSV and check that its required columns exist."""
    csv_path = PROJECT_ROOT / filename
    if not csv_path.is_file():
        raise FileNotFoundError(
            "Required dataset file was not found: "
            f"{csv_path}. Make sure {filename} is in the project root."
        )

    data = pd.read_csv(csv_path)
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(
            f"Dataset '{filename}' is missing required column(s): {missing_list}."
        )

    return data


def load_threads() -> pd.DataFrame:
    """Load and validate the main threads dataset."""
    return _load_csv("threads.csv", THREAD_COLUMNS)


def load_sample_threads() -> pd.DataFrame:
    """Load and validate the sample threads dataset."""
    return _load_csv("sample_threads.csv", THREAD_COLUMNS)


def load_sample_answers() -> pd.DataFrame:
    """Load and validate the sample answers dataset."""
    return _load_csv("sample_answers.csv", LABEL_COLUMNS)


def load_sample_submission() -> pd.DataFrame:
    """Load and validate the sample submission dataset."""
    return _load_csv("sample_submission.csv", LABEL_COLUMNS)


def print_dataset_summary(data: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """Print shape, columns, missing-value counts, and duplicate thread IDs."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    print(f"{dataset_name} shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("Missing values by column:")
    print(data.isna().sum().to_string())

    if "thread_id" not in data.columns:
        print("Duplicate thread IDs: unavailable (no 'thread_id' column).")
        return

    duplicate_rows = data["thread_id"].duplicated(keep="first")
    duplicate_ids = (
        data.loc[data["thread_id"].duplicated(keep=False), "thread_id"]
        .dropna()
        .unique()
        .tolist()
    )
    print(f"Duplicate thread ID rows (after first): {int(duplicate_rows.sum())}")
    print(f"Duplicate thread IDs: {duplicate_ids}")