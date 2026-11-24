"""Apply Procurement_Desk playbook rules to parsed supplier offers."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .data_loader import load_sample_answers, load_sample_threads
from .offer_parser import parse_offer
from .playbook_parser import (
    load_category_rules,
    load_dated_amendments,
    load_supplier_information,
)


_MONEY = r"\d[\d,]*(?:\.\d+)?"
_TABLE_EXCHANGE_RATES = {
    "USD": Decimal("84.5"),
    "EUR": Decimal("91"),
    "GBP": Decimal("106"),
}
_AMENDMENT_CATEGORY_ALIASES = {
    "courier-services": "courier",
}


@lru_cache(maxsize=1)
def _playbook_documents() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load category, supplier, and amendment documents once per process."""
    return (
        load_category_rules(),
        load_supplier_information(),
        load_dated_amendments(),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _category_key(value: str) -> str:
    slug = _slug(value)
    return _AMENDMENT_CATEGORY_ALIASES.get(slug, slug)


def _required_document(documents: Mapping[str, str], relative_path: str) -> str:
    try:
        return documents[relative_path]
    except KeyError as error:
        raise FileNotFoundError(
            f"Required playbook document was not found: {relative_path}"
        ) from error


def _parse_decimal(text: str, context: str) -> Decimal:
    try:
        return Decimal(text.replace(",", "").strip())
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {context}: {text!r}") from error


def _category_setting(document: str, label: str) -> Decimal:
    match = re.search(
        rf"^\s*-\s*\*\*{re.escape(label)}:\*\*\s*(?:₹|INR|Rs\.?)?\s*"
        rf"(?P<value>{_MONEY})",
        document,
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Category playbook is missing '{label}'.")
    return _parse_decimal(match.group("value"), label)


def _effective_category_settings(
    category: str, thread_date: date
) -> Tuple[Dict[str, Decimal], str]:
    category_documents, _, amendment_documents = _playbook_documents()
    category_path = f"categories/{_slug(category)}.md"
    document = _required_document(category_documents, category_path)
    settings = {
        "target": _category_setting(document, "Target price"),
        "walkaway": _category_setting(document, "Walk-away price"),
        "rounds": _category_setting(document, "Negotiation rounds (R)"),
        "beta": _category_setting(document, "Concession exponent (β)"),
        "approval_limit": _category_setting(document, "Approval limit"),
    }

    amendment_pattern = re.compile(
        rf"^\s*-\s*\*\*Change:\*\*\s*the\s+"
        r"(?P<metric>target price|walk-away price|approval limit)\s+for\s+"
        r"(?P<category>[^.\r\n]+?)\s+is now\s*(?:₹|INR|Rs\.?)?\s*"
        rf"(?P<value>{_MONEY})",
        re.IGNORECASE | re.MULTILINE,
    )
    effective_amendments = []
    for relative_path, amendment in amendment_documents.items():
        date_match = re.search(
            r"^\s*-\s*\*\*Effective from:\*\*\s*(\d{4}-\d{2}-\d{2})",
            amendment,
            re.IGNORECASE | re.MULTILINE,
        )
        change_match = amendment_pattern.search(amendment)
        if not date_match or not change_match:
            continue

        effective_date = date.fromisoformat(date_match.group(1))
        amendment_category = _category_key(change_match.group("category"))
        if effective_date > thread_date or amendment_category != _category_key(category):
            continue

        metric = change_match.group("metric").lower()
        setting_name = {
            "target price": "target",
            "walk-away price": "walkaway",
            "approval limit": "approval_limit",
        }[metric]
        effective_amendments.append(
            (
                effective_date,
                relative_path,
                setting_name,
                _parse_decimal(change_match.group("value"), relative_path),
            )
        )

    for _, _, setting_name, value in sorted(effective_amendments):
        settings[setting_name] = value
    return settings, document


def _supplier_status(supplier_document: str) -> str:
    match = re.search(
        r"^\s*-\s*\*\*Status:\*\*\s*(.+?)\s*$",
        supplier_document,
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        raise ValueError("Supplier playbook is missing its Status field.")
    return match.group(1).split("—", 1)[0].strip().upper()


def _supplier_contract_cap(
    supplier_document: str, category: str
) -> Optional[Decimal]:
    cap_pattern = re.compile(
        rf"\bprice cap for\s+(?P<category>.+?)\s+of\s+"
        rf"(?:₹|INR|Rs\.?)\s*(?P<value>{_MONEY})\b",
        re.IGNORECASE,
    )
    for match in cap_pattern.finditer(supplier_document):
        if _category_key(match.group("category")) == _category_key(category):
            return _parse_decimal(match.group("value"), "supplier contract price cap")
    return None


def _supplier_contract_exchange_rate(
    supplier_document: str, currency: Optional[str]
) -> Optional[Decimal]:
    if not currency or currency not in _TABLE_EXCHANGE_RATES:
        return None

    patterns = (
        re.compile(
            rf"\b(?:invoices|payments|amounts)\s+in\s+{re.escape(currency)}\s+"
            rf"(?:are\s+)?converted\s+at\s+(?:₹|INR|Rs\.?)\s*"
            rf"(?P<value>{_MONEY})\s+per\s+{re.escape(currency)}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:contract\s+)?(?:exchange\s+)?rate(?:\s+of)?\s*"
            rf"(?:₹|INR|Rs\.?)\s*(?P<value>{_MONEY})\s+"
            rf"(?:per|for)\s+{re.escape(currency)}\b",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(supplier_document)
        if match:
            return _parse_decimal(match.group("value"), f"{currency} contract rate")
    return None


def _correct_for_contract_exchange_rate(
    parsed_offer: Mapping[str, Any], supplier_document: str
) -> Optional[Decimal]:
    price = parsed_offer.get("extracted_price")
    currency = parsed_offer.get("currency")
    if price is None:
        return None

    contract_rate = _supplier_contract_exchange_rate(supplier_document, currency)
    if contract_rate is None:
        return price

    parser_used_contract_rate = any(
        "Used the supplier's contract rate" in str(note)
        for note in parsed_offer.get("extraction_notes", [])
    )
    if parser_used_contract_rate:
        return price

    table_rate = _TABLE_EXCHANGE_RATES[currency]
    return price / table_rate * contract_rate


def _thread_date(row: Mapping[str, Any]) -> date:
    value = row.get("date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError("Thread is missing its 'date'.")
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as error:
        raise ValueError(f"Invalid thread date: {value!r}") from error


def _row_decimal(row: Mapping[str, Any], field: str) -> Decimal:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Thread is missing '{field}'.")
    try:
        result = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for '{field}': {value!r}") from error
    if not result.is_finite():
        raise ValueError(f"Invalid numeric value for '{field}': {value!r}")
    return result


def _counter_price(
    target: Decimal,
    walkaway: Decimal,
    round_number: int,
    total_rounds: int,
    beta: int,
) -> Decimal:
    progress = Decimal(round_number - 1) / Decimal(total_rounds - 1)
    return target + (walkaway - target) * (progress ** beta)


def _format_counter(price: Decimal) -> str:
    return format(price.quantize(Decimal("1"), rounding=ROUND_HALF_UP), "f")


def decide_label(row: Mapping[str, Any]) -> str:
    """Return the single playbook label for a thread row."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping, such as a dict or pandas row.")

    offer = parse_offer(row)
    if not offer["has_usable_price"]:
        return "ESCALATE:MISSING_INFO:"

    supplier = row.get("supplier")
    category = row.get("category")
    if not isinstance(supplier, str) or not supplier.strip():
        raise ValueError("Thread is missing its 'supplier'.")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Thread is missing its 'category'.")

    _, supplier_documents, _ = _playbook_documents()
    supplier_path = f"suppliers/{_slug(supplier)}.md"
    supplier_document = _required_document(supplier_documents, supplier_path)
    status = _supplier_status(supplier_document)
    if "BLACKLISTED" in status:
        return "REJECT:BLACKLISTED_SUPPLIER:"
    if offer["possible_prompt_injection"]:
        return "ESCALATE:PROMPT_INJECTION:"
    if offer["possible_banned_term"]:
        return "ESCALATE:BANNED_TERM:"

    settings, _ = _effective_category_settings(category, _thread_date(row))
    price = _correct_for_contract_exchange_rate(offer, supplier_document)
    if price is None:
        return "ESCALATE:MISSING_INFO:"

    target = settings["target"]
    walkaway = _supplier_contract_cap(supplier_document, category)
    if walkaway is None:
        walkaway = settings["walkaway"]

    rounds = int(settings["rounds"])
    beta = int(settings["beta"])
    round_number = int(_row_decimal(row, "round"))
    if rounds <= 1 or round_number < 1 or round_number > rounds:
        raise ValueError(
            f"Invalid negotiation round {round_number} for category '{category}' "
            f"with {rounds} rounds."
        )
    if Decimal(beta) != settings["beta"]:
        raise ValueError(f"Category '{category}' has a non-integer concession exponent.")

    if "PROBATION" in status:
        limit_price = target
    else:
        limit_price = _counter_price(target, walkaway, round_number, rounds, beta)

    if price > limit_price:
        if round_number == rounds:
            return "REJECT:ABOVE_WALKAWAY:"
        return f"COUNTER:COUNTER_OFFER:{_format_counter(limit_price)}"

    quantity = _row_decimal(row, "quantity")
    order_value = price * quantity
    if order_value > settings["approval_limit"]:
        return "ESCALATE:OVER_APPROVAL_LIMIT:"
    return "ACCEPT:ACCEPTABLE_PRICE:"


def evaluate_samples() -> float:
    """Compare sample predictions with sample_answers.csv and print all mismatches."""
    thread_records = load_sample_threads().to_dict(orient="records")
    answer_records = load_sample_answers().to_dict(orient="records")
    expected_by_id = {str(row["thread_id"]): str(row["label"]) for row in answer_records}
    if len(expected_by_id) != len(answer_records):
        raise ValueError("sample_answers.csv contains duplicate thread IDs.")
    if not thread_records:
        raise ValueError("sample_threads.csv contains no threads to evaluate.")

    missing_answers = [
        str(row.get("thread_id"))
        for row in thread_records
        if str(row.get("thread_id")) not in expected_by_id
    ]
    if missing_answers:
        raise ValueError(
            "sample_answers.csv is missing thread IDs: " + ", ".join(missing_answers)
        )

    correct = 0
    mismatches = []
    for row in thread_records:
        thread_id = str(row["thread_id"])
        expected = expected_by_id[thread_id]
        predicted = decide_label(row)
        if predicted == expected:
            correct += 1
        else:
            mismatches.append((thread_id, expected, predicted))

    accuracy = correct / len(thread_records)
    print(f"Accuracy: {correct}/{len(thread_records)} ({accuracy:.2%})")
    if mismatches:
        print("Mismatches:")
        for thread_id, expected, predicted in mismatches:
            print(f"  {thread_id}: expected={expected} predicted={predicted}")
    else:
        print("Mismatches: none")
    return accuracy