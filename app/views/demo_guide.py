"""Optional presenter cues; the ordinary navigation remains available."""

import streamlit as st

from app.data_loader import ArtifactError
from app.demo import load_demo
from app.navigation import go_to, inspect_demo


def render():
    st.header("Demo Guide")
    st.write("A five-minute research walkthrough. Use any step or the sidebar at any time.")
    st.caption("This interface reads frozen evidence. Selecting a customer does not run live inference.")
    if st.session_state.get("amp_demo_error"):
        st.error(st.session_state.pop("amp_demo_error"))
    with st.expander("Before presenting: preflight and local startup"):
        st.write("Run the preflight before the audience arrives. It checks installed dependencies, evidence hashes, real examples and original history access without changing artifacts.")
        st.code('.\\.venv\\Scripts\\python.exe -m app.preflight\n.\\.venv\\Scripts\\python.exe -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false', language="powershell", wrap_lines=True)
        st.caption("Expected landing page: Overview. Keep data local; source redistribution permission has not been established. See app/DEMO_GUIDE.md for recovery instructions and evaluator questions.")

    st.subheader("1 · Overview · 45 seconds")
    st.write("Frame the problem: prioritize SGCC customers for review using their daily consumption. Explain customer-level FLAG supervision and the disclosed historical baseline access to the final cohort.")
    st.button("Open Overview", on_click=go_to, args=("Overview",))

    st.subheader("2 · Risk Explorer · 50 seconds")
    st.write("Show the genuine held-out ranking, search an original identifier and change sorting. A high score is a review priority, not confirmed theft or a calibrated probability.")
    st.button("Open customer rankings", on_click=go_to, args=("Customer Risk Explorer",), kwargs={"ranking": True})

    st.subheader("3 · Customer Inspector · 85 seconds")
    st.write("Inspect the theft-labelled example, then contrast the lower-ranked normal-labelled example. Hover real readings and explain gaps, observed zeros and measurement units remaining unverified. Use the difficult case only if time permits.")
    st.caption("Examples are deliberately selected using evaluation label and global rank. They illustrate the interface and are not representative of all customers. Labels do not become inference inputs.")
    try:
        demo = load_demo()
        examples = demo.examples.set_index("key")
        key = st.selectbox("Presenter example", list(examples.index), format_func=lambda k: examples.loc[k, "title"])
        row = examples.loc[key]
        st.text(row.CONS_NO)
        st.write(f"Global rank **{int(row['rank']):,}** · frozen score **{float(row.risk_score):.6f}** · evaluation FLAG **{int(row.FLAG)}**")
        st.caption(row.criterion + "; original identifier is checked against Stage 1B scores and Stage 2 history lineage.")
        st.button("Inspect verified example", on_click=inspect_demo, args=(key,))
    except ArtifactError as exc:
        st.warning(f"Presenter examples unavailable: {exc}")
        st.info("Run the preflight and restore the identified original artifact. No example is invented or substituted.")

    st.subheader("4 · Model Performance · 80 seconds")
    st.write("Explain customer ROC-AUC and Average Precision, class prevalence and the three compatible frozen methods. Show the full-scale comparison and source-supported AMP-LDP curves; AP is not trapezoidal PR area. The aggregate evidence download includes definitions and provenance.")
    st.button("Open Model Performance", on_click=go_to, args=("Model Performance",))

    st.subheader("5 · Conclusion · 40 seconds")
    st.write("Learning window representations before distribution aggregation adds customer-level discrimination beyond DIRECT on this frozen SGCC evaluation. The contribution respects customer-level labels; it does not establish exact theft days, causal explanations, calibrated probabilities or external utility deployment validity.")
    st.caption("45 + 50 + 85 + 80 + 40 = approximately five minutes. Return here from the sidebar to pick up any step.")
