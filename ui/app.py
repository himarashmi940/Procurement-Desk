"""Streamlit frontend shell for the Procurement Desk dashboard."""

import html
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.offer_parser import parse_offer
from ui.services.prediction_service import run_analysis_with_threads


STYLESHEET_PATH = APP_DIR / "styles" / "forest_luxe.css"
NAVIGATION_PAGES = ("Dashboard", "Negotiations", "Analytics", "About")
ANALYSIS_RESULTS_KEY = "procurement_desk_analysis_results"
THREAD_RECORDS_KEY = "procurement_desk_thread_records"
ANALYSIS_GENERATION_KEY = "procurement_desk_analysis_generation"
DOWNLOAD_RESULT_COLUMNS = (
    "thread_id",
    "label",
    "action",
    "reason",
    "counter_price",
)
ANALYTICS_ACTIONS = ("ACCEPT", "REJECT", "COUNTER", "ESCALATE")
ANALYTICS_ACTION_COLORS = {
    "ACCEPT": "#4f8064",
    "REJECT": "#a45e52",
    "COUNTER": "#c9aa70",
    "ESCALATE": "#9a793f",
    "UNKNOWN": "#9aa29b",
}


def _load_stylesheet() -> None:
    """Load the UI stylesheet relative to this file."""
    try:
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
    except OSError as error:
        st.warning(f"The Forest Luxe stylesheet could not be loaded: {error}")
        return

    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def _render_empty_state(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-mark" aria-hidden="true"></div>
            <h3>{title}</h3>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _action_total(
    results: Optional[pd.DataFrame], action: str
) -> Union[str, int]:
    if results is None:
        return "—"
    return int(results["action"].eq(action).sum())


def _render_decision_distribution(results: Optional[pd.DataFrame]) -> None:
    st.markdown(
        '<p class="eyebrow">DECISION MIX</p><h2>Decision distribution</h2>',
        unsafe_allow_html=True,
    )
    if results is None or results.empty:
        _render_empty_state(
            "Decision distribution is not available",
            "Run an analysis to see the distribution of actual decisions.",
        )
        return

    action_counts = (
        results["action"]
        .value_counts()
        .reindex(("ACCEPT", "REJECT", "COUNTER", "ESCALATE"), fill_value=0)
        .rename("Threads")
    )
    st.bar_chart(action_counts)


def _analytics_action_distribution(results: pd.DataFrame) -> pd.DataFrame:
    total = len(results)
    actions = results["action"].astype("string")
    rows = [
        {"action": action, "count": int(actions.eq(action).sum())}
        for action in ANALYTICS_ACTIONS
    ]
    unknown_count = total - sum(row["count"] for row in rows)
    if unknown_count:
        rows.append({"action": "UNKNOWN", "count": unknown_count})

    distribution = pd.DataFrame.from_records(rows)
    distribution["percentage"] = (
        distribution["count"].astype(float) * 100 / total if total else 0.0
    )
    distribution["total"] = total
    if int(distribution["count"].sum()) != total:
        raise ValueError("Analytics action counts do not match analyzed threads.")
    return distribution


def _analytics_reason_counts(results: pd.DataFrame) -> pd.Series:
    reasons = results["reason"].astype("string").fillna("").str.strip()
    reasons = reasons.replace("", "Unspecified")
    return reasons.value_counts()


def _analytics_category_table(
    results: pd.DataFrame, thread_records: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    if (
        thread_records is None
        or "category" not in thread_records.columns
        or len(results) != len(thread_records)
    ):
        return None

    categories = thread_records["category"].astype("string").fillna("").str.strip()
    categories = categories.replace("", "Unspecified")
    categories.name = "Category"

    actions = results["action"].astype("string").fillna("")
    actions = actions.where(actions.isin(ANALYTICS_ACTIONS), "UNKNOWN")
    actions.name = "Action"

    summary = pd.crosstab(categories, actions)
    has_unknown = bool(actions.eq("UNKNOWN").any())
    display_actions = list(ANALYTICS_ACTIONS)
    if has_unknown:
        display_actions.append("UNKNOWN")
    summary = summary.reindex(columns=display_actions, fill_value=0)
    summary["Total"] = summary.sum(axis=1)
    summary = summary.sort_values("Total", ascending=False, kind="stable")
    return summary.reset_index().loc[:, ["Category", "Total", *display_actions]]


def _category_table_markup(category_table: pd.DataFrame) -> str:
    headers = ("Category", "Total", *category_table.columns[2:])
    header_markup = "".join(
        f"<th scope=\"col\">{html.escape(str(header))}</th>"
        for header in headers
    )
    rows_markup = []
    for row in category_table.to_dict(orient="records"):
        cells = "".join(
            f"<td>{html.escape(str(row[header]))}</td>" for header in headers
        )
        rows_markup.append(f"<tr>{cells}</tr>")
    return (
        '<div class="analytics-table-wrap"><table class="analytics-table">'
        f"<thead><tr>{header_markup}</tr></thead>"
        f"<tbody>{''.join(rows_markup)}</tbody>"
        "</table></div>"
    )


def _render_category_decision_comparison(
    category_table: Optional[pd.DataFrame],
    expected_total: int,
) -> None:
    st.markdown(
        '<div class="analytics-section-heading">'
        "<h2>Category-wise Negotiation Decision Comparison</h2>"
        "<p>Decision counts by procurement category.</p></div>",
        unsafe_allow_html=True,
    )
    if category_table is None or category_table.empty:
        st.info("Category-wise decision comparison is unavailable.")
        return

    required_columns = ("Category", "Total", *ANALYTICS_ACTIONS)
    if any(column not in category_table.columns for column in required_columns):
        st.info("Category-wise decision comparison is missing required data.")
        return

    category_total = int(category_table["Total"].sum())
    decision_total = int(
        category_table.loc[:, list(ANALYTICS_ACTIONS)].to_numpy().sum()
    )
    if category_total != expected_total or decision_total != expected_total:
        st.warning(
            "The four requested decisions do not account for every analyzed "
            "thread, so the comparison chart was not rendered."
        )
        return

    category_order = category_table["Category"].astype(str).tolist()
    chart_data = category_table.loc[
        :, ["Category", *ANALYTICS_ACTIONS]
    ].melt(
        id_vars="Category",
        var_name="Decision",
        value_name="Count",
    )
    chart_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 720,
        "height": max(220, 34 * len(category_order)),
        "mark": {"type": "bar", "size": 20},
        "encoding": {
            "x": {
                "field": "Count",
                "type": "quantitative",
                "stack": "zero",
                "axis": {
                    "title": "Negotiation count",
                    "format": ",d",
                    "labelColor": "#26352e",
                    "titleColor": "#7d857d",
                    "gridColor": "#e6e1d8",
                    "labelFont": "Avenir Next",
                    "titleFont": "Avenir Next",
                },
            },
            "y": {
                "field": "Category",
                "type": "nominal",
                "sort": category_order,
                "axis": {
                    "title": None,
                    "labelColor": "#26352e",
                    "labelFont": "Avenir Next",
                    "labelFontSize": 11,
                    "labelLimit": 260,
                },
            },
            "color": {
                "field": "Decision",
                "type": "nominal",
                "scale": {
                    "domain": list(ANALYTICS_ACTIONS),
                    "range": [
                        ANALYTICS_ACTION_COLORS[action]
                        for action in ANALYTICS_ACTIONS
                    ],
                },
                "legend": {
                    "title": "Decision",
                    "orient": "top",
                    "direction": "horizontal",
                    "labelColor": "#26352e",
                    "titleColor": "#7d857d",
                    "labelFont": "Avenir Next",
                    "titleFont": "Avenir Next",
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                },
            },
            "order": {
                "field": "Decision",
                "type": "nominal",
                "sort": list(ANALYTICS_ACTIONS),
            },
            "tooltip": [
                {"field": "Category", "type": "nominal", "title": "Category"},
                {"field": "Decision", "type": "nominal", "title": "Decision"},
                {
                    "field": "Count",
                    "type": "quantitative",
                    "title": "Negotiations",
                    "format": ",",
                },
            ],
        },
        "config": {"view": {"stroke": None}},
    }
    st.vega_lite_chart(
        chart_data,
        chart_spec,
        use_container_width=True,
        key="analytics_category_decision_comparison",
    )


def _render_analytics_kpis(results: pd.DataFrame) -> None:
    cards = (
        ("Total negotiations analyzed", len(results)),
        ("ACCEPT", int(results["action"].eq("ACCEPT").sum())),
        ("REJECT", int(results["action"].eq("REJECT").sum())),
        ("COUNTER", int(results["action"].eq("COUNTER").sum())),
        ("ESCALATE", int(results["action"].eq("ESCALATE").sum())),
    )
    card_markup = "".join(
        '<div class="analytics-kpi-card">'
        f'<span class="analytics-kpi-label">{html.escape(label)}</span>'
        f'<span class="analytics-kpi-value">{value:,}</span>'
        "</div>"
        for label, value in cards
    )
    st.markdown(
        f'<div class="analytics-kpi-grid">{card_markup}</div>',
        unsafe_allow_html=True,
    )
    actions = results["action"].astype("string")
    unclassified_count = int((~actions.isin(ANALYTICS_ACTIONS)).sum())
    if unclassified_count:
        st.caption(
            f"{unclassified_count:,} unclassified result(s) are included as "
            "UNKNOWN in the decision distribution."
        )


def _render_analytics_donut(results: pd.DataFrame) -> pd.DataFrame:
    distribution = _analytics_action_distribution(results)
    if results.empty:
        _render_empty_state(
            "Decision distribution is not available",
            "No analyzed threads are available for the chart.",
        )
        return distribution

    chart_data = distribution.rename(
        columns={
            "action": "Action",
            "count": "Count",
            "percentage": "Percentage",
            "total": "Total",
        }
    )
    actions = chart_data["Action"].tolist()
    colors = [ANALYTICS_ACTION_COLORS[action] for action in actions]
    chart_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 420,
        "height": 320,
        "layer": [
            {
                "mark": {
                    "type": "arc",
                    "innerRadius": 78,
                    "outerRadius": 132,
                    "stroke": "#ffffff",
                    "strokeWidth": 2,
                },
                "encoding": {
                    "theta": {"field": "Count", "type": "quantitative", "stack": True},
                    "color": {
                        "field": "Action",
                        "type": "nominal",
                        "scale": {"domain": actions, "range": colors},
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "Action", "type": "nominal", "title": "Decision"},
                        {
                            "field": "Count",
                            "type": "quantitative",
                            "title": "Threads",
                            "format": ",",
                        },
                        {
                            "field": "Percentage",
                            "type": "quantitative",
                            "title": "Share (%)",
                            "format": ".1f",
                        },
                    ],
                },
            },
            {
                "transform": [
                    {
                        "aggregate": [
                            {"op": "max", "field": "Total", "as": "TotalCount"}
                        ]
                    }
                ],
                "mark": {
                    "type": "text",
                    "dy": -4,
                    "font": "Avenir Next",
                    "fontSize": 25,
                    "fontWeight": 700,
                    "color": "#173d30",
                },
                "encoding": {
                    "radius": {"value": 0},
                    "theta": {"value": 0},
                    "text": {
                        "field": "TotalCount",
                        "type": "quantitative",
                        "format": ",",
                    }
                },
            },
            {
                "transform": [
                    {
                        "aggregate": [
                            {"op": "max", "field": "Total", "as": "TotalCount"}
                        ]
                    }
                ],
                "mark": {
                    "type": "text",
                    "dy": 20,
                    "font": "Avenir Next",
                    "fontSize": 10,
                    "fontWeight": 650,
                    "color": "#7d857d",
                },
                "encoding": {
                    "radius": {"value": 0},
                    "theta": {"value": 0},
                    "text": {"value": "TOTAL"},
                },
            },
        ],
        "config": {"view": {"stroke": None}},
    }
    st.vega_lite_chart(
        chart_data,
        chart_spec,
        use_container_width=True,
        key="analytics_action_donut",
    )

    legend_items = "".join(
        '<div class="analytics-legend-item">'
        f'<span class="analytics-legend-swatch" '
        f'style="--legend-color:{ANALYTICS_ACTION_COLORS[row.action]}"></span>'
        '<span class="analytics-legend-name">'
        f"{html.escape(str(row.action))}</span>"
        '<span class="analytics-legend-count">'
        f"{int(row.count):,} ({row.percentage:.1f}%)</span>"
        "</div>"
        for row in distribution.itertuples(index=False)
    )
    st.markdown(
        f'<div class="analytics-donut-legend">{legend_items}</div>',
        unsafe_allow_html=True,
    )
    return distribution


def _render_analytics_reasons(reason_counts: pd.Series) -> None:
    if reason_counts.empty:
        st.info("No decision reasons are available in this analysis.")
        return
    reason_items = "".join(
        '<div class="analytics-insight-row">'
        f'<span>{html.escape(str(reason))}</span>'
        f'<strong>{int(count):,}</strong>'
        "</div>"
        for reason, count in reason_counts.head(5).items()
    )
    st.markdown(
        f'<div class="analytics-insight-list">{reason_items}</div>',
        unsafe_allow_html=True,
    )


def _render_analytics_insights(
    results: pd.DataFrame,
    distribution: pd.DataFrame,
    reason_counts: pd.Series,
    category_table: Optional[pd.DataFrame],
) -> None:
    if results.empty:
        st.info("Procurement observations will appear when analysis results are available.")
        return

    most_frequent_action_count = int(distribution["count"].max())
    most_frequent_actions = distribution.loc[
        distribution["count"] == most_frequent_action_count, "action"
    ].tolist()
    most_frequent_reason_count = int(reason_counts.max()) if not reason_counts.empty else 0
    most_frequent_reasons = (
        reason_counts[reason_counts == most_frequent_reason_count].index.tolist()
        if most_frequent_reason_count
        else []
    )

    if category_table is None:
        category_observation = "Unavailable: original category records are missing."
    elif category_table.empty:
        category_observation = "No category data is available."
    else:
        highest_category_count = int(category_table.iloc[0]["Total"])
        highest_categories = category_table.loc[
            category_table["Total"] == highest_category_count, "Category"
        ].astype(str).tolist()
        category_observation = (
            f"{', '.join(highest_categories)} ({highest_category_count:,})"
        )

    observations = (
        (
            "Most frequent action",
            f"{', '.join(most_frequent_actions)} ({most_frequent_action_count:,})",
        ),
        (
            "Most frequent decision reason",
            (
                f"{', '.join(str(reason) for reason in most_frequent_reasons)} "
                f"({most_frequent_reason_count:,})"
                if most_frequent_reasons
                else "No decision reasons are available."
            ),
        ),
        ("Category with most negotiations", category_observation),
    )
    items = "".join(
        '<div class="analytics-observation">'
        f'<span class="analytics-kpi-label">{html.escape(label)}</span>'
        f'<span class="analytics-observation-value">'
        f"{html.escape(value)}</span>"
        "</div>"
        for label, value in observations
    )
    st.markdown(
        f'<div class="analytics-observations">{items}</div>',
        unsafe_allow_html=True,
    )


def _render_decisions_table(results: Optional[pd.DataFrame]) -> None:
    st.markdown(
        '<p class="eyebrow">THREAD REVIEW</p><h2>Negotiation decisions</h2>',
        unsafe_allow_html=True,
    )
    if results is None:
        _render_empty_state(
            "No analysis results yet",
            "Run an analysis from the Dashboard to review supplier decisions.",
        )
        return
    if results.empty:
        _render_empty_state(
            "No supplier threads found",
            "The completed analysis did not contain any threads.",
        )
        return

    search_column, action_column = st.columns((2, 1))
    with search_column:
        search_text = st.text_input(
            "Search thread ID or reason",
            key="procurement_desk_search",
            placeholder="Search by thread ID, label, or reason",
        ).strip()
    with action_column:
        selected_action = st.selectbox(
            "Filter by decision",
            ("All actions", "ACCEPT", "REJECT", "COUNTER", "ESCALATE", "UNKNOWN"),
            key="procurement_desk_action_filter",
        )

    filtered_results = results
    if selected_action != "All actions":
        filtered_results = filtered_results.loc[
            filtered_results["action"] == selected_action
        ]
    if search_text:
        searchable = filtered_results[
            ["thread_id", "label", "reason"]
        ].astype("string")
        matches = searchable.apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                regex=False,
                na=False,
            )
        ).any(axis=1)
        filtered_results = filtered_results.loc[matches]

    if filtered_results.empty:
        st.info("No decisions match the selected search and filter.")
        return

    display_results = filtered_results[
        ["thread_id", "action", "reason", "counter_price"]
    ].rename(
        columns={
            "thread_id": "Thread ID",
            "action": "Decision",
            "reason": "Reason",
            "counter_price": "Counter Price (INR)",
        }
    )
    st.dataframe(display_results, hide_index=True, use_container_width=True)


def _render_results_download(results: Optional[pd.DataFrame]) -> None:
    st.markdown(
        """
        <div class="section-heading section-heading-spaced">
            <div>
                <h2>Export analysis</h2>
                <p>Download the complete set of decisions from this analysis.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if results is None:
        st.info("Run an analysis first to enable the results CSV download.")
        return

    csv_data = (
        results.loc[:, list(DOWNLOAD_RESULT_COLUMNS)]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    st.download_button(
        label="Download Results CSV",
        data=csv_data,
        file_name="procurement_analysis_results.csv",
        mime="text/csv",
        type="primary",
        key="download_procurement_analysis_results",
    )
    st.caption(
        f"Includes all {len(results):,} processed thread(s), regardless of table filters."
    )


def _display_value(value: object) -> str:
    if value is None:
        return "Not available"
    try:
        if bool(pd.isna(value)):
            return "Not available"
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text.strip() else "Not available"


def _thread_id_text(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _thread_selector_labels(results: pd.DataFrame) -> List[str]:
    normalized_ids = [_thread_id_text(value) for value in results["thread_id"]]
    totals: Dict[str, int] = {}
    for thread_id in normalized_ids:
        if thread_id is not None:
            totals[thread_id] = totals.get(thread_id, 0) + 1

    occurrences: Dict[str, int] = {}
    labels = []
    for row_index, thread_id in enumerate(normalized_ids):
        if thread_id is None:
            labels.append(f"(missing thread ID) — row {row_index + 1}")
        elif totals[thread_id] > 1:
            occurrences[thread_id] = occurrences.get(thread_id, 0) + 1
            labels.append(
                f"{thread_id} — duplicate {occurrences[thread_id]} "
                f"of {totals[thread_id]}"
            )
        else:
            labels.append(thread_id)
    return labels


def _detail_field_markup(label: str, value: str, modifier: str = "") -> str:
    return (
        f'<div class="detail-field {modifier}">'
        f'<span class="detail-label">{html.escape(label)}</span>'
        f'<span class="detail-value">{html.escape(value)}</span>'
        "</div>"
    )


def _render_individual_negotiation_details(
    results: Optional[pd.DataFrame],
    thread_records: Optional[pd.DataFrame],
) -> None:
    st.markdown(
        '<p class="eyebrow">THREAD DETAIL</p><h2>Individual Negotiation Details</h2>',
        unsafe_allow_html=True,
    )
    if results is None:
        st.info("Run an analysis to view individual negotiation details.")
        return
    if thread_records is None:
        st.info(
            "Original thread details are unavailable. "
            "Run Analysis again to load them."
        )
        return
    if results.empty:
        _render_empty_state(
            "No negotiation details available",
            "The completed analysis did not contain any threads.",
        )
        return
    if len(results) != len(thread_records):
        st.warning(
            "Thread details could not be matched safely to the current decisions. "
            "Run Analysis again to refresh them."
        )
        return

    selector_labels = _thread_selector_labels(results)
    generation = st.session_state.get(ANALYSIS_GENERATION_KEY, 0)
    selected_index = st.selectbox(
        "Search and select a thread by ID",
        options=list(range(len(results))),
        format_func=lambda index: selector_labels[index],
        help=(
            "Type to search thread IDs. Duplicate and missing IDs are "
            "distinguished by row."
        ),
        key=f"individual_negotiation_thread_{generation}",
    )
    thread = thread_records.iloc[selected_index]
    decision = results.iloc[selected_index]

    thread_fields = (
        ("Thread ID", thread.get("thread_id")),
        ("Supplier", thread.get("supplier")),
        ("Category", thread.get("category")),
        ("Date", thread.get("date")),
        ("Quantity", thread.get("quantity")),
        ("Round", thread.get("round")),
        ("Our last counter", thread.get("our_last_counter")),
    )
    thread_fields_markup = "".join(
        _detail_field_markup(label, _display_value(value))
        for label, value in thread_fields
    )

    action = _display_value(decision.get("action"))
    decision_fields = [
        _detail_field_markup(
            "Action",
            action,
            "detail-decision-action",
        ),
        _detail_field_markup(
            "Decision reason",
            _display_value(decision.get("reason")),
        ),
    ]
    if action == "COUNTER":
        counter_price_text = _display_value(decision.get("counter_price"))
        if counter_price_text != "Not available":
            try:
                counter_price_text = f"INR {int(decision['counter_price']):,}"
            except (TypeError, ValueError, OverflowError):
                counter_price_text = f"INR {counter_price_text}"
        decision_fields.append(
            _detail_field_markup("Counter price", counter_price_text)
        )

    try:
        offer = parse_offer(thread.to_dict())
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as error:
        offer = None
        offer_error = str(error)
    else:
        offer_error = ""

    if offer is None:
        offer_fields = (
            ("Raw price", "Not available"),
            ("Currency", "Not available"),
            ("Price type", "Not available"),
            ("Extracted normalized INR price", "Not available"),
        )
        notes_markup = (
            '<p class="detail-note">Parsed offer details are unavailable.</p>'
            f'<p class="detail-note">{html.escape(offer_error)}</p>'
        )
    else:
        extracted_price = offer.get("extracted_price")
        try:
            extracted_price_text = (
                f"INR {extracted_price:,.2f}"
                if extracted_price is not None
                else "Not available"
            )
        except (TypeError, ValueError):
            extracted_price_text = _display_value(extracted_price)
        offer_fields = (
            ("Raw price", _display_value(offer.get("raw_price"))),
            ("Currency", _display_value(offer.get("currency"))),
            ("Price type", _display_value(offer.get("price_type"))),
            ("Extracted normalized INR price", extracted_price_text),
        )
        notes = offer.get("extraction_notes")
        if isinstance(notes, (list, tuple)) and notes:
            notes_markup = (
                '<ul class="detail-notes-list">'
                + "".join(
                    f"<li>{html.escape(str(note))}</li>" for note in notes
                )
                + "</ul>"
            )
        else:
            notes_markup = (
                '<p class="detail-note">No extraction notes are available.</p>'
            )
        if not offer.get("has_usable_price"):
            notes_markup = (
                '<p class="detail-note">A usable normalized INR price is '
                "unavailable.</p>" + notes_markup
            )

    offer_fields_markup = "".join(
        _detail_field_markup(label, value) for label, value in offer_fields
    )
    supplier_message = _display_value(thread.get("supplier_message"))
    if supplier_message == "Not available":
        message_markup = (
            '<p class="detail-note">No original supplier message is available.</p>'
        )
    else:
        message_markup = (
            f'<div class="detail-message">{html.escape(supplier_message)}</div>'
        )

    st.markdown(
        f"""
        <div class="individual-detail-content">
            <div class="detail-decision-grid">
                {''.join(decision_fields)}
            </div>
            <div class="individual-details-grid">
                <section class="individual-detail-card">
                    <p class="detail-card-title">Thread information</p>
                    <div class="detail-fields-grid">{thread_fields_markup}</div>
                </section>
                <section class="individual-detail-card">
                    <p class="detail-card-title">Parsed offer details</p>
                    <div class="detail-fields-grid">{offer_fields_markup}</div>
                    <div class="detail-notes">
                        <span class="detail-label">Extraction notes</span>
                        {notes_markup}
                    </div>
                </section>
            </div>
            <section class="individual-detail-card detail-message-card">
                <p class="detail-card-title">Original supplier message</p>
                {message_markup}
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard() -> None:
    st.markdown('<p class="eyebrow">OVERVIEW</p>', unsafe_allow_html=True)
    st.title("Procurement Intelligence Dashboard")
    st.markdown(
        '<p class="page-subtitle">A considered view of AI-powered supplier '
        "negotiation decisions, built for clear and confident procurement.</p>",
        unsafe_allow_html=True,
    )

    if st.button("Run Analysis", type="primary", key="run_procurement_analysis"):
        try:
            with st.spinner("Analyzing supplier threads..."):
                results, thread_records = run_analysis_with_threads()
            st.session_state[ANALYSIS_RESULTS_KEY] = results
            st.session_state[THREAD_RECORDS_KEY] = thread_records
            st.session_state[ANALYSIS_GENERATION_KEY] = (
                st.session_state.get(ANALYSIS_GENERATION_KEY, 0) + 1
            )
            st.success(f"Analysis complete. {len(results):,} thread(s) processed.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            st.session_state.pop(ANALYSIS_RESULTS_KEY, None)
            st.session_state.pop(THREAD_RECORDS_KEY, None)
            st.error(
                "Analysis could not be completed. Check the thread data and "
                "playbook files, then try again."
            )
            st.caption(str(error))

    results: Optional[pd.DataFrame] = st.session_state.get(ANALYSIS_RESULTS_KEY)
    if results is not None:
        malformed_count = int((results["action"] == "UNKNOWN").sum())
        if malformed_count:
            st.warning(
                f"{malformed_count} result(s) had missing or malformed labels "
                "and are marked UNKNOWN."
            )

    status_text = "Results ready" if results is not None else "Awaiting analysis"
    st.markdown(
        f"""
        <div class="section-heading">
            <div>
                <h2>Prediction overview</h2>
                <p>Live totals from the latest in-memory analysis.</p>
            </div>
            <span class="status-pill">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = (
        (
            "Total Threads",
            "Threads in the current review",
            len(results) if results is not None else "—",
            "total",
        ),
        (
            "Accepted",
            "Offers within approved terms",
            _action_total(results, "ACCEPT"),
            "accepted",
        ),
        (
            "Rejected",
            "Offers beyond walk-away limits",
            _action_total(results, "REJECT"),
            "rejected",
        ),
        (
            "Countered",
            "Offers with a counter price",
            _action_total(results, "COUNTER"),
            "total",
        ),
        (
            "Escalated",
            "Threads requiring further review",
            _action_total(results, "ESCALATE"),
            "escalated",
        ),
    )
    columns = st.columns(5)
    for column, (label, description, value, tone) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="metric-card metric-card-{tone}">
                    <div class="metric-card-top">
                        <span class="metric-label">{label}</span>
                        <span
                            class="metric-accent metric-accent-{tone}"
                            aria-hidden="true"
                        ></span>
                    </div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _render_results_download(results)

    st.markdown(
        """
        <div class="section-heading section-heading-spaced">
            <div>
                <h2>Decision distribution</h2>
                <p>Action counts from the latest analysis.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if results is None:
        _render_empty_state(
            "No prediction results yet",
            "Prediction results will appear here after you run an analysis.",
        )
    else:
        _render_decision_distribution(results)

    st.markdown(
        """
        <div class="section-heading section-heading-spaced">
            <div>
                <h2>Negotiation results</h2>
                <p>Search by thread ID or reason, or filter by decision.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_decisions_table(results)


def _render_analytics(
    results: Optional[pd.DataFrame],
    thread_records: Optional[pd.DataFrame],
) -> None:
    st.markdown('<p class="eyebrow">INSIGHTS</p>', unsafe_allow_html=True)
    st.title("Analytics")
    st.markdown(
        '<p class="page-subtitle">A live view of decisions from the current analysis.</p>',
        unsafe_allow_html=True,
    )
    if results is None:
        _render_empty_state(
            "Analytics are not available yet",
            "Run an analysis to see actual negotiation metrics and insights.",
        )
        return

    st.markdown(
        '<div class="analytics-section-heading"><h2>Negotiation overview</h2>'
        "<p>Counts from the current in-memory analysis.</p></div>",
        unsafe_allow_html=True,
    )
    _render_analytics_kpis(results)

    st.markdown(
        '<div class="analytics-section-heading"><h2>Decision distribution</h2>'
        "<p>Share of analyzed threads by final action.</p></div>",
        unsafe_allow_html=True,
    )
    distribution = _render_analytics_donut(results)

    st.markdown(
        '<div class="analytics-section-heading"><h2>Decision reason insights</h2>'
        "<p>The most frequent reason codes in the current analysis.</p></div>",
        unsafe_allow_html=True,
    )
    reason_counts = _analytics_reason_counts(results)
    _render_analytics_reasons(reason_counts)

    st.markdown(
        '<div class="analytics-section-heading"><h2>Category insights</h2>'
        "<p>Negotiation volume and decision mix by thread category.</p></div>",
        unsafe_allow_html=True,
    )
    category_table = _analytics_category_table(results, thread_records)
    if category_table is None:
        st.info(
            "Category insights are unavailable because original thread records "
            "are missing or do not match these results."
        )
    elif category_table.empty:
        st.info("No category information is available in this analysis.")
    else:
        st.markdown(
            _category_table_markup(category_table),
            unsafe_allow_html=True,
        )
        st.caption("Blank or missing categories are grouped as Unspecified.")
    _render_category_decision_comparison(category_table, len(results))

    st.markdown(
        '<div class="analytics-section-heading"><h2>Procurement insights</h2>'
        "<p>Observations derived from the current analysis only.</p></div>",
        unsafe_allow_html=True,
    )
    _render_analytics_insights(
        results,
        distribution,
        reason_counts,
        category_table,
    )


def _render_negotiations(results: Optional[pd.DataFrame]) -> None:
    st.markdown('<p class="eyebrow">THREAD REVIEW</p>', unsafe_allow_html=True)
    st.title("Negotiations")
    st.markdown(
        '<p class="page-subtitle">Search and filter the latest negotiation decisions.</p>',
        unsafe_allow_html=True,
    )
    _render_decisions_table(results)
    _render_individual_negotiation_details(
        results,
        st.session_state.get(THREAD_RECORDS_KEY),
    )


def _render_about() -> None:
    st.markdown('<p class="eyebrow">ABOUT</p>', unsafe_allow_html=True)
    st.title("The Procurement Desk")
    st.markdown(
        '<p class="page-subtitle">A clear, consistent workspace for AI-assisted '
        "supplier negotiation decision-making.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-heading">
            <div>
                <h2>Purpose-built for thoughtful procurement</h2>
                <p>Bring supplier offers and procurement guidance into one decision workflow.</p>
            </div>
            <span class="status-pill">Playbook guided</span>
        </div>
        <div class="empty-state">
            <h3>Consistent decisions, grounded in context</h3>
            <p>
                The Procurement Desk supports AI-assisted supplier negotiation
                decisions. Each outcome is based on the supplier's negotiation
                information together with the configured procurement playbook,
                including applicable supplier terms, category rules, and amendments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-heading section-heading-spaced">
            <div>
                <h2>Four possible decisions</h2>
                <p>Each outcome communicates the next procurement action.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    decisions = (
        (
            "ACCEPT",
            "status-accepted",
            "The offer meets the applicable target and approval requirements.",
        ),
        (
            "REJECT",
            "status-rejected",
            "The offer exceeds a walk-away limit or supplier eligibility rules require rejection.",
        ),
        (
            "COUNTER",
            "status-pending",
            "Continue negotiation with the counter price returned by the decision engine.",
        ),
        (
            "ESCALATE",
            "status-escalated",
            "Missing information, a restricted term, a safety flag, or an approval exception needs review.",
        ),
    )
    columns = st.columns(2)
    for column, (action, status_class, description) in zip(columns * 2, decisions):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-top">
                        <span class="metric-label">DECISION</span>
                        <span class="status-pill {status_class}">{action}</span>
                    </div>
                    <div class="metric-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="section-heading section-heading-spaced">
            <div>
                <h2>How analysis works</h2>
                <p>A concise workflow from supplier offer to decision.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    workflow = (
        (
            "01 · Review the thread",
            "Supplier negotiation details are interpreted to identify the relevant offer.",
        ),
        (
            "02 · Apply the playbook",
            "The configured supplier terms, category rules, limits, and effective amendments guide evaluation.",
        ),
        (
            "03 · Present the outcome",
            "The dashboard shows the decision and reason, with a counter price when one applies.",
        ),
    )
    workflow_columns = st.columns(3)
    for column, (heading, description) in zip(workflow_columns, workflow):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{heading}</div>
                    <div class="metric-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_placeholder_page(title: str, description: str) -> None:
    st.markdown(f'<p class="eyebrow">{title.upper()}</p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{description}</p>', unsafe_allow_html=True)
    _render_empty_state(
        f"{title} is ready for your workflow",
        "This UI prototype is not connected to data processing yet.",
    )


def main() -> None:
    st.set_page_config(
        page_title="Procurement Desk",
        page_icon="P",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_stylesheet()

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="brand-monogram" aria-hidden="true">PD</div>
                <div>
                    <div class="brand-name">PROCUREMENT DESK</div>
                    <div class="brand-caption">AI NEGOTIATION ASSISTANT</div>
                </div>
            </div>
            <p class="sidebar-section-label">WORKSPACE</p>
            """,
            unsafe_allow_html=True,
        )
        selected_page = st.radio(
            "Navigation",
            NAVIGATION_PAGES,
            index=0,
            key="procurement_desk_page",
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div class="sidebar-footer">
                <span class="sidebar-footer-dot"></span>
                <span>Decision workspace</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    results: Optional[pd.DataFrame] = st.session_state.get(ANALYSIS_RESULTS_KEY)
    thread_records: Optional[pd.DataFrame] = st.session_state.get(THREAD_RECORDS_KEY)
    if selected_page == "Dashboard":
        _render_dashboard()
    elif selected_page == "Negotiations":
        _render_negotiations(results)
    elif selected_page == "Analytics":
        _render_analytics(results, thread_records)
    elif selected_page == "About":
        _render_about()
    else:
        _render_placeholder_page(
            selected_page,
            "Explore the Procurement Desk workspace.",
        )


if __name__ == "__main__":
    main()