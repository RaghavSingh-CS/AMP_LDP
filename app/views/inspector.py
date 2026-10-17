"""Customer observations and frozen-model context; no explanation generation."""

import altair as alt
import pandas as pd
import streamlit as st

from app.config import COHORT_NAME
from app.consumption_loader import load_history, summarize
from app.data_loader import ArtifactError


def consumption_chart(readings: pd.DataFrame):
    observed = readings.loc[~readings["missing"]]
    tooltip = [alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
               alt.Tooltip("consumption:Q", title="Original consumption", format=".8g"),
               alt.Tooltip("status:N", title="Reading status"),
               alt.Tooltip("model_calendar_note:N", title="Frozen calendar")]
    base = alt.Chart(observed).encode(
        x=alt.X("date:T", title="Calendar date", axis=alt.Axis(format="%b %Y", tickCount=8, labelOverlap=True)),
        y=alt.Y("consumption:Q", title="Consumption (source units)", scale=alt.Scale(zero=True)),
        tooltip=tooltip,
    )
    lines = base.mark_line(strokeWidth=1.5).encode(detail="segment:N")
    points = base.mark_point(filled=True, size=16, opacity=0.7)
    plot = (lines + points).properties(height=310).interactive(bind_y=False)
    coverage = alt.Chart(readings).mark_tick(thickness=2, size=18).encode(
        x=alt.X("date:T", title="Calendar date", axis=alt.Axis(format="%b %Y", tickCount=8, labelOverlap=True)),
        color=alt.Color("status:N", title="Reading coverage",
                        scale=alt.Scale(domain=["Observed", "Missing reading", "No source column"],
                                        range=["#318c9f", "#d99c36", "#8c6bb1"]),
                        legend=alt.Legend(orient="bottom", columns=1)),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"), "status:N", "model_calendar_note:N"],
    ).properties(height=24)
    # Streamlit otherwise applies pad autosizing to this layered vconcat,
    # adding axis widths outside the narrow screen's available chart width.
    return (alt.vconcat(plot, coverage, spacing=12).resolve_scale(x="shared")
            .properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding")))


def reset_interval(state, customer, start, end):
    state["intervals"][customer] = (start, end)
    state["chart_revision"] = state.get("chart_revision", 0) + 1
    st.session_state["_amp_interval"] = (start, end)


def render(result, state) -> None:
    customer = state["customer_id"]
    row = result.frame.loc[result.frame["CONS_NO"].eq(customer)]
    if st.button("Back to customer ranking"):
        state["view"] = "ranking"
        st.rerun()
    if len(row) != 1:
        st.error("The selected identifier is unavailable in the verified scored cohort.")
        return
    row = row.iloc[0]
    st.header("Customer inspection")
    st.text(customer)
    st.caption(f"Evaluation data only · {COHORT_NAME}")
    cols = st.columns(3)
    cols[0].metric("Frozen AMP-LDP risk score", f"{float(row['risk_score']):.6f}")
    cols[0].caption(f"Exact stored score: {format(float(row['risk_score']), '.17g')}")
    cols[1].metric("Global risk rank", f"{int(row['rank']):,}")
    label = int(row["FLAG"])
    cols[2].metric("Evaluation label (FLAG)", str(label))
    cols[2].caption("Theft-labelled" if label else "Normal-labelled")
    st.caption("FLAG is ground-truth evaluation information, not a model output. The score is not an established calibrated probability.")
    st.caption("Local inspection only. Customer-history CSV and clipboard exports are disabled; sharing permission is not established.")
    previous, following = st.columns(2)
    rank = int(row["rank"])
    for column, name, offset in ((previous, "Previous rank", -1), (following, "Next rank", 1)):
        target = result.frame.loc[result.frame["rank"].eq(rank + offset)]
        if column.button(name, disabled=target.empty):
            state["customer_id"] = target.iloc[0]["CONS_NO"]
            st.rerun()
    with st.expander("Frozen score provenance"):
        st.text(result.source)
        st.write(result.message)
        if result.provenance_source:
            st.text(result.provenance_source)

    try:
        with st.spinner("Verifying and loading original consumption history…"):
            history = load_history(customer)
    except ArtifactError as exc:
        st.warning("Customer consumption history unavailable")
        st.write(str(exc))
        st.info("Use the Demo Guide preflight instructions to identify and restore the original source. The frozen score and Back control remain available.")
        return
    readings = history.readings
    stats = summarize(readings)
    start, end = readings.date.min().date(), readings.date.max().date()
    st.subheader("Original consumption history")
    st.caption(f"{start.isoformat()} to {end.isoformat()} · {stats['calendar_days']:,} calendar days · {history.source_date_columns:,} source date columns")
    quality = st.columns(4)
    for column, title, value in zip(quality,
                                    ["Observed readings", "Missing readings", "Missing readings (%)", "Observed zero readings"],
                                    [f"{stats['observed']:,}", f"{stats['missing']:,}", f"{stats['missing_percent']:.2f}", f"{stats['zero_readings']:,}"]):
        column.metric(title, value)
    st.caption("Counts cover the full observation period, including the absent 18 September 2016 source column as missing. A recorded zero is an observed reading.")
    if state.get("interval_customer") != customer or "_amp_interval" not in st.session_state:
        st.session_state["_amp_interval"] = state["intervals"].get(customer, (start, end))
        state["interval_customer"] = customer
    interval = st.date_input("Viewed reading interval", min_value=start, max_value=end,
                             key="_amp_interval", format="YYYY-MM-DD")
    st.button("Reset viewed interval", on_click=reset_interval, args=(state, customer, start, end))
    st.caption("Choose both interval endpoints. Pan or wheel-zoom horizontally in the chart; hover a point to inspect its original reading. Reset restores the full period.")
    if len(interval) != 2:
        st.info("Choose an interval end date to display the chart.")
        return
    state["intervals"][customer] = tuple(interval)
    visible = readings.loc[readings.date.between(pd.Timestamp(interval[0]), pd.Timestamp(interval[1]))]
    if visible["missing"].all():
        st.info("No observed readings in this interval; the coverage strip shows missing days.")
    st.altair_chart(consumption_chart(visible), width="stretch", key=f"history_chart_{customer}_{interval}_{state.get('chart_revision', 0)}",
                    alt="Original customer consumption with explicit gaps and a separate observed/missing reading coverage strip.")
    st.caption("Original CSV numeric values, with no interpolation, inverse transformation or silent zero fill. Lines stop at missing days. Measurement units were not established in the source metadata.")
    st.dataframe(pd.DataFrame({"Observed-value statistic": ["Mean", "Median", "Minimum", "Maximum", "Population standard deviation"],
                               "Full-period source value": [stats["mean"], stats["median"], stats["minimum"], stats["maximum"], stats["population_sd"]]}),
                 hide_index=True, width="stretch")
    st.subheader("What AMP-LDP evaluated")
    st.write(f"This customer has **{len(history.windows)} evaluated windows** in the frozen usable test manifest. Each window contains 30 consecutive positions on the processed calendar.")
    st.write("The frozen pipeline imputes within-customer gaps, applies log1p and training-fitted scaling, then computes 17 window summaries. A learned encoder produces 16 latent dimensions; six summaries per dimension form a 96-feature customer profile. Five frozen HGB scores are averaged for the customer-level ranking.")
    st.caption("The processed calendar excludes 3 October 2014 and inserts 18 September 2016 as missing. It has 34 full candidate windows and 14 trailing unused positions. Fully missing windows are excluded. Thirty positions can span a removed calendar date; the raw chart retains any source reading on that date.")
    st.info("We can inspect readings, missingness, score and rank. The fitted model produces a customer-level score. This output does not establish an exact theft day, an independently verified theft event, or a causal explanation for the score.")
    with st.expander("Evaluated window coverage and history provenance"):
        st.dataframe(history.windows[["window_number_within_customer", "start_date", "end_date", "missing_count"]],
                     hide_index=True, width="stretch",
                     column_config={"window_number_within_customer": "Frozen window number (zero-based)",
                                    "missing_count": "Originally missing positions (of 30)"})
        st.text(history.source)
        st.text(history.provenance)
