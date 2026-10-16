import streamlit as st

from app.config import COHORT_NAME
from app.data_loader import ArtifactError, load_evidence


def render() -> None:
    st.header("Research overview")
    st.write("Customer-level electricity theft risk prioritization using learned smart-meter consumption profiles.")
    try:
        evidence = load_evidence()
    except ArtifactError as exc:
        st.error(f"Frozen metrics unavailable: {exc}")
        st.info("Restore the reviewed frozen evidence. Historical values are not substituted.")
        return

    columns = st.columns(4)
    for column, label, value in zip(columns,
                                    ["Evaluation customers", "Theft-labelled customers", "Final ROC-AUC", "Final Average Precision"],
                                    [f"{evidence.customers:,}", f"{evidence.theft_customers:,}",
                                     f"{evidence.roc_auc:.4f}", f"{evidence.average_precision:.4f}"]):
        column.metric(label, value)
    st.caption(f"Study 2 · {COHORT_NAME} · frozen Stage 27B results. Ranking metrics use four-decimal reporting.")
    st.subheader("How AMP-LDP works")
    st.write("Daily consumption → non-overlapping 30-day windows → 17 summary features → learned 17 → 32 → 16 encoder → 96 customer-level latent distribution features → HistGradientBoosting → arithmetic mean of five seed scores.")
    st.write("Each latent dimension is summarized by mean, stabilized population standard deviation, q10, median, q90 and maximum. The ensemble uses seeds 42–46. Models were fitted on DEV_TRAIN only and frozen before terminal evaluation.")
    st.info("Risk ranking supports research prioritization. A high score is not confirmed electricity theft, legal proof, or a verified theft period. Scores have not been established as calibrated theft probabilities.")
    st.subheader("Evaluation context")
    st.write(evidence.disclosure)
    st.write("Evidence is internal to SGCC customer-level labels. It does not establish external generalization or operational deployment validity. The frozen research and model development remain closed.")
    with st.expander("Model and artifact provenance"):
        st.write("**AMP-LDP** · LATENT_DISTRIBUTION_HGB_FIVE_SEED_MEAN")
        st.caption(f"Evidence commit: {evidence.catalog['evidence_commit']}")
        for name in ("metrics", "integrity", "model", "protocol", "access"):
            record = evidence.catalog["artifacts"][name]
            st.text(record["path"])
            st.caption(f"SHA-256 (LF-normalized text): {record['sha256']}")
