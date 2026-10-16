import streamlit as st

from app.config import COHORT_NAME
from app.data_loader import explore, load_predictions


def render() -> None:
    st.header("Customer Risk Explorer")
    st.caption(f"Evaluation data only · {COHORT_NAME}")
    st.info("Scores express model ranking, not calibrated theft probabilities. Ground-truth FLAG is a customer-level evaluation label, not evidence of theft on a particular day.")
    result = load_predictions()
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
    query = st.text_input("Search customer identifier", placeholder="Enter all or part of the original identifier")
    order = st.selectbox("Sort customers", ["Highest risk first", "Lowest risk first", "Customer identifier"])
    filtered = explore(result.frame, query, order)
    st.caption(f"{len(filtered):,} matching customers · global ordinal rank, ties ordered by identifier")
    if filtered.empty:
        st.info("No matching customer identifiers.")
    st.dataframe(filtered, hide_index=True, width="stretch", height=540,
                 column_config={"rank": st.column_config.NumberColumn("Global rank", format="%d"),
                                "CONS_NO": st.column_config.TextColumn("Customer identifier"),
                                "risk_score": st.column_config.NumberColumn("AMP-LDP risk score", format="%.6f"),
                                "FLAG": st.column_config.NumberColumn("Evaluation label (FLAG)", help="0 = normal-labelled; 1 = theft-labelled", format="%d")})
