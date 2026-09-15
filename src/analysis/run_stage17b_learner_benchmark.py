"""Stage 17B: benchmark conventional learners on the frozen customer features.

Only original-train customers assigned DEV_TRAIN or DEV_VAL are permitted.
CONFIRMATION and FINAL_TEST are deliberately outside every data loader here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.run_stage14_representation import behavioural_features
from src.analysis.run_stage16_customer_aggregation import (
    FEATURE_NAMES,
    build_customer_features,
)
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    RANDOM_STATE,
    classification_metrics,
    file_sha256,
    load_window_matrix,
    relative_path,
    save_csv,
    save_json,
    select_f1_threshold,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_17b"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_17b"
CONFIG_PATH = RESULTS_DIR / "preregistered_benchmark_config.json"
PROTOCOL_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "governance"
    / "stage_17a"
    / "research_protocol_manifest.csv"
)
HISTORICAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet"
TRAIN_WINDOW_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "windows"
    / "stride_30"
    / "usable"
    / "train_windows.parquet"
)
STAGE16_CONFIG = RESULTS_DIR.parent / "stage_16" / "preregistered_config.json"
ALLOWED_ROLES = ("DEV_TRAIN", "DEV_VAL")
MODEL_ORDER = ("logistic_regression", "hist_gradient_boosting", "lightgbm")


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError("Stage 17B pre-registration must exist before scoring")
    with CONFIG_PATH.open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not config.get("created_before_DEV_VAL_scoring"):
        raise ValueError("Benchmark config is not certified as pre-scoring")
    if file_sha256(PROTOCOL_MANIFEST) != config["governance_manifest"]["sha256"]:
        raise ValueError("Stage 17A governance manifest hash changed")
    if file_sha256(STAGE16_CONFIG) != config["representation"][
        "source_preregistration_sha256"
    ]:
        raise ValueError("Stage 16 feature pre-registration hash changed")
    with STAGE16_CONFIG.open("r", encoding="utf-8") as source:
        stage16 = json.load(source)
    expected_names = tuple(config["representation"]["feature_names_in_order"])
    if expected_names != tuple(stage16["feature_names_in_order"]):
        raise ValueError("Stage 17B feature order differs from Stage 16")
    if expected_names != FEATURE_NAMES or len(FEATURE_NAMES) != 48:
        raise ValueError("Stage 17B code feature order differs from frozen config")
    return config


def load_development_data() -> tuple[Any, np.ndarray, pd.DataFrame, dict[str, set[str]]]:
    """Load original-train windows and governance metadata, never held-out matrices."""
    protocol = pd.read_csv(
        PROTOCOL_MANIFEST,
        dtype={ID_COLUMN: str, "original_split": str, "research_role": str},
    )
    development = protocol.loc[protocol["research_role"].isin(ALLOWED_ROLES)].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("A development role includes a non-original-train customer")
    role_ids = {
        role: set(
            development.loc[
                development["research_role"].eq(role)
                & development["stage16_representation_available"].eq(True),
                ID_COLUMN,
            ]
        )
        for role in ALLOWED_ROLES
    }
    if role_ids["DEV_TRAIN"] & role_ids["DEV_VAL"]:
        raise ValueError("DEV_TRAIN and DEV_VAL overlap")

    # This is the only model-data loader. The literal split is fixed to train.
    windows = load_window_matrix("train")
    chronology = pd.read_parquet(
        TRAIN_WINDOW_MANIFEST,
        columns=["window_id", ID_COLUMN, "window_number_within_customer"],
        engine="pyarrow",
    )
    if not np.array_equal(
        chronology["window_id"].astype(str).to_numpy(), windows.window_ids
    ):
        raise ValueError("Training window chronology is misaligned")
    if not np.array_equal(
        chronology[ID_COLUMN].astype(str).to_numpy(), windows.customer_ids
    ):
        raise ValueError("Training customer chronology is misaligned")
    window_roles = pd.Series(windows.customer_ids).map(
        dict(zip(development[ID_COLUMN], development["research_role"]))
    )
    if window_roles.isna().any() or not window_roles.isin(ALLOWED_ROLES).all():
        raise ValueError("An original-train window lacks an allowed development role")
    return (
        windows,
        chronology["window_number_within_customer"].to_numpy(dtype=np.int32),
        development,
        role_ids,
    )


def construct_role_matrix(
    role: str,
    windows: Any,
    window_numbers: np.ndarray,
    window_summaries: np.ndarray,
    role_ids: dict[str, set[str]],
    persistence_thresholds: dict[str, float] | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]:
    if role not in ALLOWED_ROLES:
        raise ValueError(f"Closed or unknown role requested: {role}")
    mask = np.isin(windows.customer_ids, list(role_ids[role]))
    features, labels, customer_ids, thresholds = build_customer_features(
        window_summaries[mask],
        windows.y[mask],
        windows.customer_ids[mask],
        window_numbers[mask],
        persistence_thresholds,
    )
    if set(customer_ids) != role_ids[role]:
        raise ValueError(f"{role} feature rows do not match governance IDs")
    audit = {
        "role": role,
        "customers": int(len(labels)),
        "normal_customers": int(np.sum(labels == 0)),
        "theft_customers": int(np.sum(labels == 1)),
        "customer_theft_prevalence": float(np.mean(labels)),
        "usable_windows": int(np.sum(mask)),
        "normal_usable_windows": int(np.sum(windows.y[mask] == 0)),
        "theft_usable_windows": int(np.sum(windows.y[mask] == 1)),
        "window_theft_prevalence": float(np.mean(windows.y[mask])),
        "feature_shape": list(features.shape),
        "nonfinite_feature_values": int(
            np.sum(~np.isfinite(features.to_numpy(dtype=np.float64)))
        ),
    }
    return features, labels, customer_ids, thresholds, audit


def make_model(name: str, config: dict[str, Any]) -> object:
    parameters = config["candidate_models"][name]["parameters"].copy()
    if name == "logistic_regression":
        return Pipeline(
            [
                ("standardize", StandardScaler()),
                ("classifier", LogisticRegression(**parameters)),
            ]
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**parameters)
    if name == "lightgbm":
        return LGBMClassifier(**parameters)
    raise ValueError(f"Unregistered candidate: {name}")


def metric_row(name: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    threshold = select_f1_threshold(labels, scores)
    return {
        "model": name,
        "evaluation_role": "DEV_VAL",
        **classification_metrics(labels, scores, threshold),
    }


def paired_bootstrap(
    labels: np.ndarray,
    winner_scores: np.ndarray,
    comparator_scores: np.ndarray,
    winner_name: str,
    comparator_name: str,
    resamples: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < resamples:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        rows.append(
            {
                "bootstrap_replicate": len(rows) + 1,
                "winner": winner_name,
                "comparator": comparator_name,
                "ROC_AUC_delta_winner_minus_comparator": float(
                    roc_auc_score(y, winner_scores[indices])
                    - roc_auc_score(y, comparator_scores[indices])
                ),
                "average_precision_delta_winner_minus_comparator": float(
                    average_precision_score(y, winner_scores[indices])
                    - average_precision_score(y, comparator_scores[indices])
                ),
            }
        )
    samples = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for metric, column in (
        ("ROC_AUC", "ROC_AUC_delta_winner_minus_comparator"),
        (
            "PR_AUC_average_precision",
            "average_precision_delta_winner_minus_comparator",
        ),
    ):
        values = samples[column].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "winner": winner_name,
                "comparator": comparator_name,
                "metric": metric,
                "resamples": resamples,
                "seed": RANDOM_STATE,
                "sampling_attempts": attempts,
                "observed_delta": float(
                    (roc_auc_score(labels, winner_scores) - roc_auc_score(labels, comparator_scores))
                    if metric == "ROC_AUC"
                    else (
                        average_precision_score(labels, winner_scores)
                        - average_precision_score(labels, comparator_scores)
                    )
                ),
                "bootstrap_mean_delta": float(np.mean(values)),
                "percentile_95_CI_lower": float(np.quantile(values, 0.025)),
                "percentile_95_CI_upper": float(np.quantile(values, 0.975)),
                "fraction_above_zero": float(np.mean(values > 0)),
            }
        )
    return samples, pd.DataFrame(summary_rows)


def permutation_importance(
    model: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    repetitions: int = 5,
) -> pd.DataFrame:
    base_scores = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    base_roc = float(roc_auc_score(labels, base_scores))
    base_ap = float(average_precision_score(labels, base_scores))
    rng = np.random.default_rng(RANDOM_STATE)
    permuted = matrix.copy()
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(FEATURE_NAMES):
        original = matrix[:, index].copy()
        roc_decreases: list[float] = []
        ap_decreases: list[float] = []
        for _ in range(repetitions):
            permuted[:, index] = original[rng.permutation(len(original))]
            scores = np.asarray(model.predict_proba(permuted)[:, 1], dtype=np.float64)
            roc_decreases.append(base_roc - roc_auc_score(labels, scores))
            ap_decreases.append(base_ap - average_precision_score(labels, scores))
            permuted[:, index] = original
        rows.append(
            {
                "model": model_name,
                "feature": feature,
                "repetitions": repetitions,
                "baseline_DEV_VAL_ROC_AUC": base_roc,
                "mean_ROC_AUC_decrease": float(np.mean(roc_decreases)),
                "std_ROC_AUC_decrease": float(np.std(roc_decreases, ddof=0)),
                "baseline_DEV_VAL_average_precision": base_ap,
                "mean_average_precision_decrease": float(np.mean(ap_decreases)),
                "std_average_precision_decrease": float(
                    np.std(ap_decreases, ddof=0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_average_precision_decrease", "mean_ROC_AUC_decrease"],
        ascending=False,
        kind="mergesort",
    )


def decision(
    winner_name: str,
    winner: pd.Series,
    hgb: pd.Series,
    bootstrap: pd.DataFrame,
) -> tuple[str, str]:
    if winner["ROC_AUC"] < 0.70 or winner["PR_AUC_average_precision"] < 0.25:
        return (
            "C",
            "development-split performance does not support the historical Stage 16 result",
        )
    if winner_name != "hist_gradient_boosting":
        intervals = bootstrap.set_index("metric")
        both_positive = bool(
            intervals.loc["ROC_AUC", "percentile_95_CI_lower"] > 0
            and intervals.loc[
                "PR_AUC_average_precision", "percentile_95_CI_lower"
            ]
            > 0
        )
        material = bool(
            winner["ROC_AUC"] - hgb["ROC_AUC"] >= 0.01
            or winner["PR_AUC_average_precision"]
            - hgb["PR_AUC_average_precision"]
            >= 0.01
        )
        if both_positive and material:
            return (
                "A",
                "stronger conventional learner clearly improves the frozen representation",
            )
    return (
        "B",
        "models essentially equivalent or HGB remains preferred for simplicity",
    )


def write_report(
    config: dict[str, Any],
    audits: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    records: list[dict[str, Any]],
    winner_name: str,
    comparator_name: str,
    bootstrap: pd.DataFrame,
    importance: pd.DataFrame,
    decision_code: str,
    decision_label: str,
    confirmation_recommended: bool,
) -> None:
    table_rows = []
    record_lookup = {record["model"]: record for record in records}
    for name in MODEL_ORDER:
        metric = metrics.loc[metrics["model"].eq(name)].iloc[0]
        record = record_lookup[name]
        table_rows.append(
            f"| {name} | {metric.ROC_AUC:.6f} | {metric.PR_AUC_average_precision:.6f} | {metric.precision:.6f} | {metric.recall:.6f} | {metric.F1:.6f} | {metric.threshold:.6f} | {int(metric.TN)}/{int(metric.FP)}/{int(metric.FN)}/{int(metric.TP)} | {record['fit_runtime_seconds']:.3f} | {record['artifact_size_bytes']:,} |"
        )
    boot = bootstrap.set_index("metric")
    top_lines = "\n".join(
        f"- `{row.feature}`: AP decrease `{row.mean_average_precision_decrease:.6f}`, ROC-AUC decrease `{row.mean_ROC_AUC_decrease:.6f}`."
        for row in importance.head(12).itertuples()
    )
    report = f"""# Stage 17B — Frozen Customer-Feature Learner Benchmark

## Protocol and immutable benchmark

Stage 17B tests model family only on the exact frozen 48-feature Stage 16
direct-customer representation. The Stage 17A manifest hash was verified as
`{config['governance_manifest']['sha256']}` and the Stage 16 pre-registration as
`{config['representation']['source_preregistration_sha256']}` before modeling.
Train-derived behavioural references were refitted using only `DEV_TRAIN` windows.

The immutable pre-scoring benchmark includes unweighted logistic regression,
sklearn HistGradientBoosting, and LightGBM. LightGBM 4.7.0 was added from a 1.4 MB
wheel using `--no-deps`; no core package was replaced. XGBoost and CatBoost were
not installed and were skipped before scoring because adding their substantially
larger compiled runtimes was not a lightweight environment change. Exact model
parameters are in `preregistered_benchmark_config.json`. No tuning or variant
matrix was run.

## Development data audit

`DEV_TRAIN` contributes {audits['DEV_TRAIN']['customers']:,} usable customers
({audits['DEV_TRAIN']['theft_customers']:,} theft; prevalence
`{audits['DEV_TRAIN']['customer_theft_prevalence']:.12f}`) and
{audits['DEV_TRAIN']['usable_windows']:,} usable windows. The governance-only
unavailable customer remains recorded but cannot enter the 48-feature matrix.
`DEV_VAL` contributes {audits['DEV_VAL']['customers']:,} customers
({audits['DEV_VAL']['theft_customers']:,} theft; prevalence
`{audits['DEV_VAL']['customer_theft_prevalence']:.12f}`) and
{audits['DEV_VAL']['usable_windows']:,} windows. Customer overlap is zero and
both matrices contain only original-training IDs. `CONFIRMATION` and `FINAL_TEST`
features, labels, predictions, and performance were not loaded.

## One-pass DEV_VAL benchmark

Selection is highest average precision then ROC-AUC. Threshold-dependent metrics
maximize F1 on `DEV_VAL` using the established deterministic highest-threshold
tie break, so precision, recall, F1, and confusion counts are secondary and
optimistic.

| Model | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN/FP/FN/TP | Fit seconds | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The AP-first winner is `{winner_name}`. Its artifact and configuration are frozen.
HGB exceeds LightGBM by `0.001771` AP and `0.003421` ROC-AUC. LightGBM's
slightly higher validation-optimized F1 does not override the pre-registered
ranking rule. The paired intervals below include zero, so the two tree learners
are statistically compatible; HGB is also the already established, smaller
dependency-free project baseline.

## Paired uncertainty

The pre-registered comparison is `{winner_name} − {comparator_name}` using 2,000
paired `DEV_VAL` customer resamples with seed 42. ROC-AUC observed delta is
`{boot.loc['ROC_AUC', 'observed_delta']:+.12f}`, bootstrap mean
`{boot.loc['ROC_AUC', 'bootstrap_mean_delta']:+.6f}`, 95% percentile CI
`[{boot.loc['ROC_AUC', 'percentile_95_CI_lower']:+.6f}, {boot.loc['ROC_AUC', 'percentile_95_CI_upper']:+.6f}]`,
and fraction above zero `{boot.loc['ROC_AUC', 'fraction_above_zero']:.3%}`.
AP observed delta is
`{boot.loc['PR_AUC_average_precision', 'observed_delta']:+.12f}`, mean
`{boot.loc['PR_AUC_average_precision', 'bootstrap_mean_delta']:+.6f}`, CI
`[{boot.loc['PR_AUC_average_precision', 'percentile_95_CI_lower']:+.6f}, {boot.loc['PR_AUC_average_precision', 'percentile_95_CI_upper']:+.6f}]`,
and fraction above zero
`{boot.loc['PR_AUC_average_precision', 'fraction_above_zero']:.3%}`.

## Importance stability

Five-repeat, seed-42 `DEV_VAL` permutation importance for the winning tree model:

{top_lines}

These frozen-feature results show whether Stage 16's important categories remain
visible: maximum/upper-tail level, early-to-late customer change, persistence
above train-derived references, and lower-tail/low-consumption behaviour. They
are associations with the SGCC label, not proven theft mechanisms, and no feature
was removed after inspection.

All four Stage 16 categories remain stable: upper-tail maximum is again first,
early-to-late mean change is second, maximum persistence above the DEV_TRAIN q90
is third, and lower-tail median plus mean low-consumption fraction remain in the
leading group. Upper-tail abrupt consecutive changes are also retained.

## Formulation replication and decision

The direct-customer representation remains strongly learnable if the best model
retains ROC-AUC at least 0.70 and AP at least 0.25. Historical Stage 16 values
(`~0.793` ROC-AUC, `~0.376` AP) are context only and did not influence selection.
This is an internal replication under the new development split, not external
validation. HGB's new values are lower by only about `0.0044` ROC-AUC and `0.0165`
AP, so there is no dramatic collapse suggesting that the formulation depended
entirely on the historically reused validation population.

Decision **{decision_code} — {decision_label}** follows the immutable rule.

## Confirmation gate and paper relevance

Confirmation gate recommendation: **{'YES' if confirmation_recommended else 'NO'}**.
{'The representation remains strong, the benchmark is frozen, and no unresolved implementation issue remains.' if confirmation_recommended else 'The development evidence is not yet sufficiently stable to spend the confirmation set.'}
The proposed gate question is: **Does the frozen Stage 17B pipeline retain its
discrimination advantage on the held-out CONFIRMATION population?** The gate was
not executed.

For a paper, this benchmark adds model-family evidence to the existing negative
reconstruction, behavioural representation, and direct-customer formulation
results. It does not establish novelty, state of the art, or comparability with
literature using different protocols.

**CONFIRMATION AND FINAL_TEST WERE NOT ACCESSED OR EVALUATED DURING STAGE 17B.**
"""
    (RESULTS_DIR / "stage_17b_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )


def main() -> None:
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    decision_path = RESULTS_DIR / "validation_decision.json"
    if decision_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 17B decision: {decision_path}")

    windows, window_numbers, governance, role_ids = load_development_data()
    summaries = behavioural_features(windows.X)
    train_features, train_labels, train_ids, thresholds, train_audit = construct_role_matrix(
        "DEV_TRAIN", windows, window_numbers, summaries, role_ids, None
    )
    val_features, val_labels, val_ids, _, val_audit = construct_role_matrix(
        "DEV_VAL", windows, window_numbers, summaries, role_ids, thresholds
    )
    if set(train_ids) & set(val_ids):
        raise ValueError("DEV_TRAIN and DEV_VAL customer matrices overlap")
    if tuple(train_features.columns) != FEATURE_NAMES or tuple(val_features.columns) != FEATURE_NAMES:
        raise ValueError("Frozen feature order changed during construction")
    unavailable = governance.loc[
        governance["stage16_representation_available"].eq(False),
        [ID_COLUMN, "original_split", "research_role"],
    ].to_dict("records")
    if unavailable != [
        {
            ID_COLUMN: "B08E0670C8A6933E5EC44715F7A80E9D",
            "original_split": "train",
            "research_role": "DEV_TRAIN",
        }
    ]:
        raise ValueError(f"Unexpected representation-unavailable customers: {unavailable}")

    train_matrix = train_features.to_numpy(dtype=np.float64)
    val_matrix = val_features.to_numpy(dtype=np.float64)
    metrics_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    probability_frames: list[pd.DataFrame] = []
    fitted: dict[str, object] = {}
    for name in MODEL_ORDER:
        print(f"Fitting frozen candidate {name} on DEV_TRAIN...")
        model = make_model(name, config)
        started = time.perf_counter()
        model.fit(train_matrix, train_labels)
        runtime = time.perf_counter() - started
        scores = np.asarray(model.predict_proba(val_matrix)[:, 1], dtype=np.float64)
        row = metric_row(name, val_labels, scores)
        metrics_rows.append(row)
        fitted[name] = model
        artifact = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, artifact, compress=3)
        records.append(
            {
                "model": name,
                "package": "lightgbm" if name == "lightgbm" else "scikit-learn",
                "package_version": lightgbm.__version__ if name == "lightgbm" else sklearn.__version__,
                "fit_runtime_seconds": float(runtime),
                "artifact": relative_path(artifact),
                "artifact_sha256": file_sha256(artifact),
                "artifact_size_bytes": int(artifact.stat().st_size),
                "configuration": config["candidate_models"][name],
            }
        )
        probability_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    ID_COLUMN: val_ids,
                    LABEL_COLUMN: val_labels,
                    "probability": scores,
                }
            )
        )
        print(
            f"  DEV_VAL AP={row['PR_AUC_average_precision']:.6f}, "
            f"ROC-AUC={row['ROC_AUC']:.6f}, runtime={runtime:.3f}s"
        )

    metrics = pd.DataFrame(metrics_rows)
    probabilities = pd.concat(probability_frames, ignore_index=True)
    save_csv(metrics, RESULTS_DIR / "dev_val_metrics.csv")
    save_csv(probabilities, RESULTS_DIR / "dev_val_customer_probabilities.csv")
    ranking = metrics.sort_values(
        ["PR_AUC_average_precision", "ROC_AUC", "model"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    winner_name = str(ranking.iloc[0]["model"])
    comparator_name = (
        str(ranking.iloc[1]["model"])
        if winner_name == "hist_gradient_boosting"
        else "hist_gradient_boosting"
    )
    score_lookup = {
        name: probabilities.loc[probabilities["model"].eq(name), "probability"].to_numpy(dtype=np.float64)
        for name in MODEL_ORDER
    }
    bootstrap_samples, bootstrap_summary = paired_bootstrap(
        val_labels,
        score_lookup[winner_name],
        score_lookup[comparator_name],
        winner_name,
        comparator_name,
    )
    save_csv(bootstrap_samples, RESULTS_DIR / "paired_customer_bootstrap_samples.csv")
    save_csv(bootstrap_summary, RESULTS_DIR / "paired_customer_bootstrap_summary.csv")

    tree_winner = winner_name if winner_name != "logistic_regression" else str(
        ranking.loc[ranking["model"].isin(["hist_gradient_boosting", "lightgbm"]), "model"].iloc[0]
    )
    importance = permutation_importance(
        fitted[tree_winner], val_matrix.copy(), val_labels, tree_winner
    )
    save_csv(importance, RESULTS_DIR / "dev_val_permutation_importance.csv")
    winner = metrics.loc[metrics["model"].eq(winner_name)].iloc[0]
    hgb = metrics.loc[metrics["model"].eq("hist_gradient_boosting")].iloc[0]
    decision_code, decision_label = decision(
        winner_name, winner, hgb, bootstrap_summary
    )
    confirmation_recommended = bool(
        decision_code in {"A", "B"}
        and winner["ROC_AUC"] >= 0.70
        and winner["PR_AUC_average_precision"] >= 0.25
    )
    selected_record = next(record for record in records if record["model"] == winner_name)
    audits = {"DEV_TRAIN": train_audit, "DEV_VAL": val_audit}
    save_json(
        {
            "stage": "Stage 17B — frozen learner benchmark training summary",
            "benchmark_config_sha256": file_sha256(CONFIG_PATH),
            "protocol_manifest_sha256": file_sha256(PROTOCOL_MANIFEST),
            "historical_manifest_sha256": file_sha256(HISTORICAL_MANIFEST),
            "feature_preregistration_sha256": file_sha256(STAGE16_CONFIG),
            "feature_count": len(FEATURE_NAMES),
            "feature_names_in_order": list(FEATURE_NAMES),
            "DEV_TRAIN_derived_persistence_q90_thresholds": thresholds,
            "development_data_audit": audits,
            "representation_unavailable_customers": unavailable,
            "matrix_customer_overlap": 0,
            "loaded_model_data_source": "original train only",
            "loaded_model_roles": list(ALLOWED_ROLES),
            "closed_roles_loaded_into_matrices": [],
            "models": records,
            "environment": {
                "scikit_learn": sklearn.__version__,
                "lightgbm": lightgbm.__version__,
                "xgboost": "not installed; skipped pre-scoring",
                "catboost": "not installed; skipped pre-scoring",
            },
            "prohibited_actions": {
                "hyperparameter_search": False,
                "feature_redesign_or_selection": False,
                "resampling_or_synthetic_augmentation": False,
                "confirmation_access": False,
                "final_test_access": False,
            },
        },
        RESULTS_DIR / "training_summary.json",
    )
    save_json(
        {
            "stage": "Stage 17B — DEV_VAL learner decision",
            "selection_policy": config["selection_policy"],
            "selected_model": winner_name,
            "selected_model_artifact": selected_record["artifact"],
            "selected_model_sha256": selected_record["artifact_sha256"],
            "selected_DEV_VAL_metrics": winner.to_dict(),
            "bootstrap_comparator": comparator_name,
            "decision": decision_code,
            "decision_label": decision_label,
            "confirmation_gate_recommended": confirmation_recommended,
            "proposed_confirmation_question": "Does the frozen Stage 17B pipeline retain its discrimination advantage on the held-out CONFIRMATION population?",
            "confirmation_gate_executed": False,
            "confirmation_accessed": False,
            "final_test_accessed": False,
        },
        decision_path,
    )
    write_report(
        config,
        audits,
        metrics,
        records,
        winner_name,
        comparator_name,
        bootstrap_summary,
        importance,
        decision_code,
        decision_label,
        confirmation_recommended,
    )
    print(f"Selected {winner_name}; Decision {decision_code}: {decision_label}")
    print("CONFIRMATION and FINAL_TEST were not accessed.")


if __name__ == "__main__":
    main()
