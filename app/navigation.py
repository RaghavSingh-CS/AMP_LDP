"""Callbacks keep sidebar and guided navigation on the same native radio state."""

import streamlit as st

from app.data_loader import ArtifactError
from app.demo import load_demo

PAGES = ["Overview", "Customer Risk Explorer", "Model Performance", "Demo Guide"]


def go_to(page: str, *, ranking: bool = False):
    if page not in PAGES:
        raise ValueError("Unknown interface page.")
    if ranking:
        state = st.session_state.setdefault("amp_explorer", {"view": "ranking", "customer_id": None,
                                                             "query": "", "order": "Highest risk first", "intervals": {}})
        state["view"] = "ranking"
    st.session_state["amp_page"] = page


def inspect_demo(key: str):
    try:
        examples = load_demo().examples
        selected = examples.loc[examples.key.eq(key)]
        if len(selected) != 1:
            raise ArtifactError("The presenter example is not in the verified selection.")
        go_to("Customer Risk Explorer", ranking=True)
        state = st.session_state["amp_explorer"]
        state["customer_id"] = selected.iloc[0].CONS_NO
        state["view"] = "detail"
        st.session_state.pop("amp_demo_error", None)
    except ArtifactError as exc:
        st.session_state["amp_demo_error"] = str(exc)
        go_to("Demo Guide")
