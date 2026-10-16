"""Inference-only replay authorized by Interface Stage 1B (17 October 2026).

Never run a historical main(), change its access gate, refit a transformation,
or write into data/, models/, results/ or src/. Derived outputs are create-only.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

# Preserve research source files, including their bytecode directories.
sys.dont_write_bytecode = True

from app.config import MODEL_ID, REPOSITORY_ROOT, SEEDS
from app.data_loader import ArtifactError, load_evidence, validate_predictions
from app.recovery.verification import verify_metrics

EVIDENCE_COMMIT = "dbd978ffeebe7cee772f8ec693243895be7ff2c8"
SCORING_SOURCES = [
    "src/evaluation/evaluate_stage27b_terminal.py",
    "src/analysis/run_stage25b_distribution_experiment.py",
    "src/analysis/run_stage14_representation.py",
    "src/analysis/run_stage16_customer_aggregation.py",
    "src/analysis/supervised_signal_utils.py",
]
TEST_PATH = "data/processed/test.parquet"
WINDOW_PATH = "data/windows/stride_30/usable/test_windows.parquet"
SPLIT_PATH = "data/processed/split_manifest.parquet"
GOVERNANCE_PATH = "results/governance/stage_17a/research_protocol_manifest.csv"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True, encoding="utf-8").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(relative: str) -> dict:
    path = REPOSITORY_ROOT / relative
    return {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size,
            "hash_policy": "exact file bytes"}


def read_json(relative: str) -> dict:
    return json.loads((REPOSITORY_ROOT / relative).read_text(encoding="utf-8"))


def require(condition, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def snapshot(extra: list[str]) -> dict[str, str]:
    paths = sorted(set(git("ls-files", "results", "models", "data", "src").splitlines() + extra))
    return {relative: sha256(REPOSITORY_ROOT / relative) for relative in paths}


@contextmanager
def forbid_fitting():
    """Runtime tripwires supplement the reviewed no-fit call graph."""
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def denied(*args, **kwargs):
        raise RuntimeError("Fitting/backpropagation is prohibited during frozen replay.")

    with ExitStack() as stack:
        for cls in (HistGradientBoostingClassifier, StandardScaler, Pipeline):
            for name in ("fit", "fit_transform", "partial_fit"):
                if hasattr(cls, name):
                    stack.enter_context(patch.object(cls, name, denied))
        stack.enter_context(patch.object(torch.Tensor, "backward", denied))
        yield


def prerequisites() -> tuple[dict, dict]:
    evidence = load_evidence()
    require(not git("diff", EVIDENCE_COMMIT, "--name-only", "--", *SCORING_SOURCES),
            "Original scoring helpers differ from the frozen evidence revision.")
    require(not git("diff", "HEAD", "--name-only", "--", "results", "models", "data", "src"),
            "Protected tracked research paths have uncommitted changes.")
    frozen_input_hashes = read_json("results/cnn_autoencoder/stride_30/test_evaluation/test_metrics.json")["protected_pretest_sha256"]
    protocol = read_json("results/governance/stage_17a/protocol_summary.json")
    expected = {TEST_PATH: frozen_input_hashes[TEST_PATH],
                WINDOW_PATH: frozen_input_hashes[WINDOW_PATH],
                SPLIT_PATH: protocol["historical_split"]["sha256"],
                GOVERNANCE_PATH: protocol["manifest"]["sha256"],
                evidence.model["method"]["standardizer_artifact"]: evidence.model["method"]["standardizer_artifact_sha256"],
                "models/preprocessing/scaler.joblib": read_json("results/supervised/stage_23b/pre_access_audit.json")["verified_artifacts"]["preprocessing_scaler"]["expected_sha256"]}
    for record in evidence.model["checkpoints"]:
        for key in ("encoder", "hgb"):
            expected[record[key]] = record[f"{key}_sha256"]
    for relative, expected_hash in expected.items():
        require(sha256(REPOSITORY_ROOT / relative) == expected_hash, f"Frozen byte hash mismatch: {relative}")
    require(sha256(REPOSITORY_ROOT / evidence.model["implementation_reference"]["path"])
            == evidence.model["implementation_reference"]["sha256"], "Frozen encoder implementation hash mismatch.")
    for relative in SCORING_SOURCES:
        committed = subprocess.check_output(["git", "show", f"{EVIDENCE_COMMIT}:{relative}"], cwd=REPOSITORY_ROOT)
        current = (REPOSITORY_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        require(current == committed.replace(b"\r\n", b"\n"), f"Source content mismatch: {relative}")
    input_paths = set(expected) | set(SCORING_SOURCES)
    input_paths.update(record["path"] for record in evidence.catalog["artifacts"].values())
    input_paths.update([
        "results/governance/stage_17a/protocol_summary.json",
        "results/cnn_autoencoder/stride_30/test_evaluation/test_metrics.json",
        "results/supervised/stage_23b/pre_access_audit.json",
        "results/supervised/stage_25b/stage25b_execution_spec.json",
        "results/supervised/stage_27b/final_seed_metrics.csv",
    ])
    return {p: artifact(p) for p in sorted(input_paths)}, expected


def run(output_relative: str) -> dict:
    output = (REPOSITORY_ROOT / output_relative).resolve()
    allowed = (REPOSITORY_ROOT / "app/derived").resolve()
    require(output != allowed and output.is_relative_to(allowed), "Outputs must be a new directory inside app/derived/.")
    require(not output.exists(), f"Output directory already exists; replay never overwrites: {output}")
    inputs, expected = prerequisites()
    before = snapshot([TEST_PATH, WINDOW_PATH, SPLIT_PATH])
    evidence = load_evidence()

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.utils.validation import check_is_fitted
    from src.evaluation import evaluate_stage27b_terminal as terminal

    loaded = []
    with forbid_fitting():
        # Every component is checked before any customer's score is generated.
        for record in evidence.model["checkpoints"]:
            checkpoint = torch.load(REPOSITORY_ROOT / record["encoder"], map_location="cpu", weights_only=True)
            require(checkpoint["seed"] == record["seed"]
                    and checkpoint["feature_names"] == list(terminal.SUMMARY_NAMES)
                    and checkpoint["execution_spec_sha256"] == terminal.STAGE25B_SPEC_SHA256,
                    "Encoder checkpoint metadata mismatch.")
            model = terminal.DistributionModel()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            require(all(torch.equal(model.state_dict()[key], value) for key, value in checkpoint["state_dict"].items()),
                    "Not all initialized tensors were replaced by frozen fitted state.")
            hgb = joblib.load(REPOSITORY_ROOT / record["hgb"])
            check_is_fitted(hgb)
            require(hgb.classes_.tolist() == [0, 1] and hgb.n_features_in_ == 96,
                    "Fitted HGB class orientation or input dimension changed.")
            loaded.append((record["seed"], model, hgb))
        require([seed for seed, _, _ in loaded] == SEEDS, "Frozen seed order changed.")
        print("All fitted components and historical byte hashes verified; preparing FINAL_TEST only.", flush=True)
        windows = terminal.load_window_matrix("test")
        manifest = pd.read_parquet(REPOSITORY_ROOT / WINDOW_PATH, engine="pyarrow")
        governance = pd.read_csv(REPOSITORY_ROOT / GOVERNANCE_PATH, dtype=str, keep_default_na=False)
        x, mask, ids, labels, _unused_full48, integrity = terminal.prepare_representations(windows, manifest, governance)
        for key in ("governed_customers", "eligible_customers", "normal_customers", "theft_customers",
                    "eligible_windows", "max_bag_size", "customer_membership_sha256",
                    "window_membership_sha256", "labels_sha256", "split_overlaps"):
            require(integrity[key] == evidence.integrity[key], f"Stage 27B cohort integrity mismatch: {key}")
        historical = pd.read_parquet(REPOSITORY_ROOT / SPLIT_PATH, engine="pyarrow")
        historical = historical.loc[historical["split"].eq("test")].set_index("CONS_NO")
        require(set(ids) == set(historical.index.astype(str)), "Historical test split membership mismatch.")
        require(np.array_equal(historical.loc[ids, "FLAG"].to_numpy(dtype=np.int8), labels), "Historical test labels misaligned.")
        seed_scores = {}
        seed_checks = {}
        seed_metrics = pd.read_csv(REPOSITORY_ROOT / "results/supervised/stage_27b/final_seed_metrics.csv")
        with torch.no_grad():
            for seed, model, hgb in loaded:
                representation = terminal.latent_features(model, x, mask)
                scores = hgb.predict_proba(representation)[:, 1]
                frozen = seed_metrics.loc[seed_metrics["seed"].eq(seed)].iloc[0]
                seed_checks[str(seed)] = verify_metrics(labels, scores, roc_auc=float(frozen["ROC_AUC"]),
                                                        average_precision=float(frozen["average_precision"]))
                seed_scores[seed] = scores
                print(f"Seed {seed}: frozen metric agreement passed.", flush=True)
        scores = sum(seed_scores[seed] for seed in SEEDS) / 5.0
        require(np.allclose(scores, np.mean(np.stack([seed_scores[s] for s in SEEDS]), axis=0), rtol=0, atol=1e-15),
                "Arithmetic ensemble consistency failure.")
        verification = verify_metrics(labels, scores, roc_auc=evidence.roc_auc, average_precision=evidence.average_precision)
        frame = pd.DataFrame({"CONS_NO": ids, "FLAG": labels, "risk_score": scores})
        validate_predictions(frame, governance, customers=evidence.customers, theft_customers=evidence.theft_customers,
                             membership_sha256=evidence.integrity["customer_membership_sha256"],
                             labels_sha256=evidence.integrity["labels_sha256"])

    after = snapshot([TEST_PATH, WINDOW_PATH, SPLIT_PATH])
    require(before == after, "Research byte snapshot changed; integration prohibited.")
    # Only now create the separate DERIVED interface artifact.
    output.mkdir(parents=True, exist_ok=False)
    score_path = output / "customer_scores.csv"
    frame.to_csv(score_path, index=False, float_format="%.17g", lineterminator="\n")
    roundtrip = pd.read_csv(score_path, dtype={"CONS_NO": str}, keep_default_na=False, float_precision="round_trip")
    require(np.array_equal(roundtrip["risk_score"].to_numpy(), scores), "CSV roundtrip changed frozen scores.")
    verify_metrics(roundtrip["FLAG"], roundtrip["risk_score"], roc_auc=evidence.roc_auc, average_precision=evidence.average_precision)
    snapshot_path = output / "research_snapshot.json"
    snapshot_path.write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8", newline="\n")
    relative = output.relative_to(REPOSITORY_ROOT).as_posix()
    report = {
        "schema_version": 1,
        "stage": "Interface Stage 1B",
        "origin": "reconstructed_inference_only",
        "original_stage27b_persisted_predictions": False,
        "authorization": "User's Interface Stage 1B conditional authorization, 17 October 2026; frozen inference only and metric recomputation for integrity only.",
        "created_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "model_id": MODEL_ID,
        "source_protocol_stage": "27B",
        "source_evidence_commit": EVIDENCE_COMMIT,
        "replay_source_head": git("rev-parse", "HEAD"),
        "replay_script": artifact("app/recovery/replay_stage1b.py"),
        "verification_script": artifact("app/recovery/verification.py"),
        "model_freeze_sha256": evidence.catalog["artifacts"]["model"]["sha256"],
        "protocol_sha256": evidence.catalog["artifacts"]["protocol"]["sha256"],
        "inputs": inputs,
        "historical_expected_byte_hashes_verified": expected,
        "preprocessing": "Read existing scaled test.parquet; original SUMMARY_17 helper; saved Stage 25B DEV_TRAIN mean/scale only. No scaler fitting or split regeneration.",
        "scoring": {"seeds": SEEDS, "ensemble": "arithmetic_mean", "seed_order": SEEDS,
                    "source_helper": "evaluate_stage27b_terminal.latent_features", "batch_size": terminal.BATCH_SIZE,
                    "class_column": 1, "model_mode": "eval", "device": "cpu", "torch_num_threads": torch.get_num_threads(),
                    "fit_calls": 0, "backpropagation_calls": 0,
                    "guards": "HGB, StandardScaler and Pipeline fitting methods plus Tensor.backward raise if called",
                    "original_preparation_reused": "prepare_representations validates original FULL_48 alignment with saved thresholds; no comparator estimator is loaded or scored."},
        "cohort_integrity": integrity,
        "verification": verification,
        "seed_metric_verification": seed_checks,
        "output": {**artifact(f"{relative}/customer_scores.csv"), "customers": len(frame),
                   "theft_customers": int(labels.sum()), "columns": list(frame.columns),
                   "ordering": "Original Stage 27B first-customer-occurrence order; windows sorted within customer",
                   "labels": "FLAG: evaluation-only customer annotations; not theft dates"},
        "research_integrity": {"unchanged": True, "files_checked": len(before),
                               "snapshot": artifact(f"{relative}/research_snapshot.json")},
        "reproduction_command": f'.\\.venv\\Scripts\\python.exe -m app.recovery.replay_stage1b --output-dir {relative}-reproduced',
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
                        ("numpy", "pandas", "pyarrow", "torch", "scikit-learn", "joblib")}},
        "no_threshold_changes": True,
        "no_training_or_validation_features_loaded": True,
        "status": "VERIFIED",
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "output": report["output"], "verification": verification,
                      "protected_files_unchanged": len(before)}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="app/derived/stage1b")
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
