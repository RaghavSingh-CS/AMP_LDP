import streamlit as st

from app.config import COHORT_NAME
from app.data_loader import explore, load_predictions
from app.views import inspector


def render() -> None:
    result = load_predictions()
    state = st.session_state.setdefault("amp_explorer", {"view": "ranking", "customer_id": None,
                                                         "query": "", "order": "Highest risk first", "intervals": {}})
    if result.frame is not None and state["view"] == "detail":
        inspector.render(result, state)
        return
    st.header("Customer Risk Explorer")
    st.caption(f"Evaluation data only · {COHORT_NAME}")
    st.info("Scores express model ranking, not calibrated theft probabilities. Ground-truth FLAG is a customer-level evaluation label, not evidence of theft on a particular day.")
    if result.frame is None:
        st.warning("AMP-LDP customer scores unavailable")
        st.write(result.message)
        st.caption("Scored customer count cannot be verified without a canonical score artifact. The aggregate cohort count is not a loaded-score count.")
        st.text_input("Search customer identifier", disabled=True)
        st.selectbox("Sort customers", ["Highest risk first", "Lowest risk first", "Customer identifier"], disabled=True)
        return
    st.metric("Total scored customers", f"{len(result.frame):,}")
    st.caption(f"Score artifact: {result.source}")
    st.write(result.message)
    if result.provenance_source:
        with st.expander("Score provenance"):
            st.text(result.provenance_source)
            st.caption("Hash-verified manifest records frozen inputs, fitted checkpoints, replay code, metric agreement and research-file integrity.")
    st.session_state.setdefault("_amp_query", state["query"])
    st.session_state.setdefault("_amp_order", state["order"])
    query = st.text_input("Search customer identifier", placeholder="Enter all or part of the original identifier", key="_amp_query")
    order = st.selectbox("Sort customers", ["Highest risk first", "Lowest risk first", "Customer identifier"], key="_amp_order")
    state["query"], state["order"] = query, order
    filtered = explore(result.frame, query, order)
    st.caption(f"{len(filtered):,} matching customers · global ordinal rank, ties ordered by identifier")
    if filtered.empty:
        st.info("No matching customer identifiers.")
    else:
        options = filtered["CONS_NO"].tolist()
        if st.session_state.get("_amp_picker") not in options:
            st.session_state["_amp_picker"] = state["customer_id"] if state["customer_id"] in options else options[0]
        ranks = filtered.set_index("CONS_NO")["rank"].to_dict()
        customer = st.selectbox("Customer to inspect", options, key="_amp_picker",
                                format_func=lambda identifier: f"Rank {ranks[identifier]:,} · {identifier}")
        if st.button("Open customer inspector"):
            state["customer_id"], state["view"] = customer, "detail"
            st.rerun()
    st.dataframe(filtered, hide_index=True, width="stretch", height=540,
                 column_config={"rank": st.column_config.NumberColumn("Global rank", format="%d"),
                                "CONS_NO": st.column_config.TextColumn("Customer identifier"),
                                "risk_score": st.column_config.NumberColumn("AMP-LDP risk score", format="%.6f"),
                                "FLAG": st.column_config.NumberColumn("Evaluation label (FLAG)", help="0 = normal-labelled; 1 = theft-labelled", format="%d")})
