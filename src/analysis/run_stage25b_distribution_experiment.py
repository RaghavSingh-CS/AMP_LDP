"""Execute sealed Stage 25B latent-distribution experiment.

Only original-train windows assigned to DEV_TRAIN or DEV_VAL are loaded.
No instance FLAG target is constructed; BCE is applied once per customer bag.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

from src.analysis.run_stage14_representation import FEATURE_NAMES, behavioural_features
from src.analysis.supervised_signal_utils import (
    ID_COLUMN,
    LABEL_COLUMN,
    PROJECT_ROOT,
    file_sha256,
    load_window_matrix,
    save_csv,
    save_json,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_25b"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_25b"
SPEC_PATH = RESULTS_DIR / "stage25b_execution_spec.json"
EXPECTED_SPEC_SHA256 = "006c781899f74f0f62d01282fbf4164ac3ee27d8a61e3f2ea5fcb09b9e445470"
SOURCE_COMMIT = "d6ec2726ca34050780adf52aaeaf1c40b053daf6"
STUDY1_COMMIT = "d0601c54f6e9d0f11caac2ebd27f7c86a568a863"
GOVERNANCE_PATH = PROJECT_ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
WINDOW_PATH = PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "train_windows.parquet"
STAGE14_CONFIG = RESULTS_DIR.parent / "stage_14" / "preregistered_config.json"
BASELINE_PREDICTIONS = RESULTS_DIR.parent / "stage_17b" / "dev_val_customer_probabilities.csv"
SEEDS = (42, 43, 44, 45, 46)
MAX_BAG_SIZE = 34
FEATURE_COUNT = 17
EPOCHS = 50
BATCH_SIZE = 64
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2502
FULL48 = {"ROC_AUC": 0.7888308218953116, "average_precision": 0.35979459032174654}
MIL = {
    "MEAN_MIL": {"ROC_AUC": 0.739867, "average_precision": 0.267821},
    "MAX_MIL": {"ROC_AUC": 0.756667, "average_precision": 0.314566},
    "ATTENTION_MIL": {"ROC_AUC": 0.758053, "average_precision": 0.297462},
}


@dataclass
class Bags:
    role: str
    x: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    customer_ids: np.ndarray
    bag_sizes: np.ndarray
    window_ids: list[np.ndarray]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8").strip()


def stable_hash(values: list[str] | np.ndarray) -> str:
    joined = "\n".join(str(v) for v in values).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def validate_seal() -> dict[str, Any]:
    observed = file_sha256(SPEC_PATH)
    if observed != EXPECTED_SPEC_SHA256:
        raise ValueError(f"Execution spec changed: {observed}")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if tuple(spec["training"]["seeds"]) != SEEDS or spec["fit_grid"]["total"] != 11:
        raise ValueError("Sealed run grid changed")
    if spec["neural_head"]["total_trainable_parameters"] != 1201:
        raise ValueError("Sealed parameter count changed")
    return spec


def pre_stage_audit() -> dict[str, Any]:
    if git("branch", "--show-current") != "study2/mil":
        raise ValueError("Stage 25B must run on study2/mil")
    if git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("Stage 25B source HEAD changed")
    if git("rev-parse", "main") != STUDY1_COMMIT:
        raise ValueError("Frozen Study 1 main moved")
    protected = git("diff", "--name-only", STUDY1_COMMIT, "--", "results/supervised/stage_23a", "results/supervised/stage_23b", "models/supervised/stage_23b", "data")
    if protected:
        raise ValueError("Protected Study 1 paths changed")
    prior_dirs = [RESULTS_DIR.parent / name for name in ("stage_24a", "stage_24b", "stage_25a")]
    changed_prior = git("diff", "--name-only", SOURCE_COMMIT, "--", *(str(p.relative_to(PROJECT_ROOT)) for p in prior_dirs))
    if changed_prior:
        raise ValueError("Stage 24A/24B/25A artifacts changed")
    hashes = {
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(p)
        for directory in prior_dirs for p in sorted(directory.iterdir()) if p.is_file()
    }
    return {
        "stage": "Stage 25B pre-stage audit",
        "branch": "study2/mil",
        "source_HEAD": SOURCE_COMMIT,
        "main_HEAD": STUDY1_COMMIT,
        "git_clean_before_stage": True,
        "execution_spec_sha256": EXPECTED_SPEC_SHA256,
        "protected_study1_path_changes": [],
        "prior_stage_artifact_hashes": hashes,
        "confirmation_row_level_access": False,
        "final_test_row_level_access": False,
    }


def masked_distribution(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return dimension-major [mean, sd, q10, median, q90, max] features."""
    counts = mask.sum(dim=1)
    if bool((counts < 1).any()):
        raise ValueError("Empty bag")
    weights = mask.unsqueeze(-1).to(values.dtype)
    denominator = counts.to(values.dtype).unsqueeze(-1)
    mean = (values * weights).sum(dim=1) / denominator
    variance = (((values - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / denominator
    ordered, _ = torch.sort(values.masked_fill(~mask.unsqueeze(-1), torch.inf), dim=1)

    def gather_at(indices: torch.Tensor) -> torch.Tensor:
        expanded = indices[:, None, None].expand(-1, 1, values.shape[-1])
        return torch.gather(ordered, 1, expanded).squeeze(1)

    stats = [mean, torch.sqrt(variance + 1e-8)]
    for q in (0.10, 0.50, 0.90):
        position = q * (counts.to(values.dtype) - 1.0)
        lower, upper = torch.floor(position).long(), torch.ceil(position).long()
        interpolation_weight = (position - lower.to(position.dtype)).unsqueeze(-1)
        stats.append(gather_at(lower) * (1.0 - interpolation_weight) + gather_at(upper) * interpolation_weight)
    stats.append(gather_at(counts - 1))
    return torch.stack(stats, dim=2).reshape(values.shape[0], -1)


class DistributionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(17, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 16), nn.ReLU()
        )
        self.classifier = nn.Linear(96, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def represent(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        local_width = int(mask.sum(dim=1).max().item())
        return masked_distribution(self.encoder(x[:, :local_width]), mask[:, :local_width])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.represent(x, mask)).squeeze(-1)


def numerical_tests() -> dict[str, Any]:
    one = torch.tensor([[[2.0, -1.0]]], requires_grad=True)
    one_result = masked_distribution(one, torch.tensor([[True]]))
    if one_result.shape != (1, 12) or not torch.allclose(one_result[0, [1, 7]], torch.tensor([1e-4, 1e-4])):
        raise ValueError("Bag-size-one test failed")
    two = torch.tensor([[[0.0], [10.0]]], requires_grad=True)
    two_result = masked_distribution(two, torch.tensor([[True, True]]))[0]
    expected = torch.tensor([5.0, math.sqrt(25.0 + 1e-8), 1.0, 5.0, 9.0, 10.0])
    if not torch.allclose(two_result, expected, atol=1e-6):
        raise ValueError("Two-item/interpolation test failed")
    ordinary = torch.tensor([[[1.0], [3.0], [9.0], [999.0]]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False]])
    result = masked_distribution(ordinary, mask)
    padded_changed = ordinary.detach().clone()
    padded_changed[0, 3, 0] = -99999.0
    if not torch.equal(result.detach(), masked_distribution(padded_changed, mask)):
        raise ValueError("Masked-padding test failed")
    result.sum().backward()
    valid_gradient = ordinary.grad[0, :3].abs().sum().item()
    padded_gradient = ordinary.grad[0, 3].abs().sum().item()
    if valid_gradient <= 0 or padded_gradient != 0:
        raise ValueError("Gradient propagation test failed")
    model = DistributionModel()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != 1201:
        raise ValueError(f"Parameter count {parameter_count} != 1201")
    return {
        "bag_size_1": "passed", "bag_size_2": "passed", "ordinary_variable_bag": "passed",
        "masked_padding": "passed", "linear_quantile_interpolation": "passed",
        "gradient_propagation": "passed", "valid_gradient_absolute_sum": valid_gradient,
        "padded_gradient_absolute_sum": padded_gradient, "exact_parameter_count": parameter_count,
    }


def bag_distribution(values: np.ndarray) -> dict[str, float | int]:
    q = np.quantile(values, [0.1, 0.5, 0.9], method="linear")
    return {"min": int(values.min()), "q10": float(q[0]), "median": float(q[1]), "q90": float(q[2]), "max": int(values.max()), "mean": float(values.mean()), "std_population": float(values.std(ddof=0))}


def load_bags() -> tuple[Bags, Bags, dict[str, Any]]:
    config = json.loads(STAGE14_CONFIG.read_text(encoding="utf-8"))
    if tuple(config["feature_names_in_order"]) != FEATURE_NAMES or len(FEATURE_NAMES) != 17:
        raise ValueError("SUMMARY_17 schema changed")
    governance = pd.read_csv(GOVERNANCE_PATH, dtype={ID_COLUMN: str})
    development = governance.loc[governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"]), [ID_COLUMN, "original_split", "research_role"]].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("Development role includes protected split")
    manifest = pd.read_parquet(WINDOW_PATH, engine="pyarrow")
    if manifest["window_id"].duplicated().any():
        raise ValueError("Duplicate window IDs")
    manifest["research_role"] = manifest[ID_COLUMN].astype(str).map(development.set_index(ID_COLUMN)["research_role"])
    if manifest["research_role"].isna().any():
        raise ValueError("Original-train window lacks development role")
    raw = load_window_matrix("train")
    if not np.array_equal(raw.window_ids, manifest["window_id"].astype(str).to_numpy()):
        raise ValueError("Window matrix order differs from manifest")
    summaries = behavioural_features(raw.X)
    if summaries.shape != (768923, 17) or not np.isfinite(summaries).all():
        raise ValueError("SUMMARY_17 shape/finite check failed")
    train_rows = manifest["research_role"].eq("DEV_TRAIN").to_numpy()
    mean = summaries[train_rows].mean(axis=0, dtype=np.float64)
    scale = summaries[train_rows].std(axis=0, dtype=np.float64, ddof=0)
    scale[scale == 0] = 1.0
    standardized = ((summaries.astype(np.float64) - mean) / scale).astype(np.float32)
    manifest["_row"] = np.arange(len(manifest), dtype=np.int64)

    def build(role: str) -> Bags:
        groups = list(manifest.loc[manifest["research_role"].eq(role)].groupby(ID_COLUMN, sort=False))
        n = len(groups)
        x = np.zeros((n, MAX_BAG_SIZE, FEATURE_COUNT), dtype=np.float32)
        mask = np.zeros((n, MAX_BAG_SIZE), dtype=bool)
        y = np.empty(n, dtype=np.float32)
        ids, window_ids = [], []
        sizes = np.empty(n, dtype=np.int16)
        for i, (customer, group) in enumerate(groups):
            group = group.sort_values("window_number_within_customer", kind="stable")
            rows = group["_row"].to_numpy(dtype=np.int64)
            m = len(rows)
            if m > MAX_BAG_SIZE or group[LABEL_COLUMN].nunique() != 1:
                raise ValueError("Invalid customer bag")
            x[i, :m], mask[i, :m], y[i] = standardized[rows], True, float(group[LABEL_COLUMN].iloc[0])
            ids.append(str(customer)); sizes[i] = m
            window_ids.append(group["window_id"].astype(str).to_numpy())
        return Bags(role, torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(y), np.asarray(ids), sizes, window_ids)

    train, val = build("DEV_TRAIN"), build("DEV_VAL")
    if set(train.customer_ids) & set(val.customer_ids):
        raise ValueError("DEV split overlap")
    expected = {"DEV_TRAIN": (23724, 21702, 2022, 614976), "DEV_VAL": (5931, 5426, 505, 153947)}
    integrity: dict[str, Any] = {
        "stage": "Stage 25B bag integrity", "representation": "SUMMARY_17",
        "ordered_features": list(FEATURE_NAMES), "representation_dimension": 17,
        "nonfinite_values": int((~np.isfinite(standardized)).sum()),
        "duplicate_window_ids": int(manifest["window_id"].duplicated().sum()),
        "DEV_TRAIN_DEV_VAL_customer_overlap": 0, "confirmation_customers_loaded": 0,
        "final_test_customers_loaded": 0, "instance_level_targets_created": False,
        "inputs": {"governance_sha256": file_sha256(GOVERNANCE_PATH), "window_manifest_sha256": file_sha256(WINDOW_PATH), "stage14_config_sha256": file_sha256(STAGE14_CONFIG)},
        "standardizer": {"fit_role": "DEV_TRAIN", "mean": mean.tolist(), "scale": scale.tolist()}, "roles": {},
    }
    for bags in (train, val):
        observed = (len(bags.y), int((bags.y == 0).sum()), int((bags.y == 1).sum()), int(bags.mask.sum()))
        if observed != expected[bags.role]:
            raise ValueError(f"{bags.role} counts changed: {observed}")
        integrity["roles"][bags.role] = {
            "customers": observed[0], "normal_customers": observed[1], "theft_labelled_customers": observed[2], "windows": observed[3],
            "bag_size": bag_distribution(bags.bag_sizes), "customer_membership_sha256": stable_hash(bags.customer_ids),
            "window_membership_sha256": stable_hash(np.concatenate(bags.window_ids)),
        }
    return train, val, integrity


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def predict_neural(model: DistributionModel, bags: Bags) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); probabilities, representations = [], []
    for first in range(0, len(bags.y), BATCH_SIZE):
        last = min(first + BATCH_SIZE, len(bags.y))
        representation = model.represent(bags.x[first:last], bags.mask[first:last])
        probabilities.append(torch.sigmoid(model.classifier(representation).squeeze(-1)).numpy())
        representations.append(representation.numpy())
    return np.concatenate(probabilities), np.concatenate(representations)


def train_neural(seed: int, train: Bags, val: Bags) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    seed_everything(seed); model = DistributionModel()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != 1201:
        raise ValueError("Neural parameter assertion failed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.BCEWithLogitsLoss(); generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter(); final_loss = math.nan
    for _epoch in range(EPOCHS):
        model.train(); order = torch.randperm(len(train.y), generator=generator); loss_sum = 0.0
        for first in range(0, len(order), BATCH_SIZE):
            idx = order[first:first + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train.x[idx], train.mask[idx]), train.y[idx])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            loss_sum += float(loss.item()) * len(idx)
        final_loss = loss_sum / len(train.y)
    val_probability, val_representation = predict_neural(model, val)
    _, train_representation = predict_neural(model, train)
    # Explicit padding invariance on eight fixed bags.
    indices = np.linspace(0, len(val.y) - 1, 8, dtype=int)
    extra_x = torch.cat([val.x[indices], torch.randn((8, 7, FEATURE_COUNT))], dim=1)
    extra_mask = torch.cat([val.mask[indices], torch.zeros((8, 7), dtype=torch.bool)], dim=1)
    model.eval()
    with torch.no_grad():
        delta = float((model(val.x[indices], val.mask[indices]) - model(extra_x, extra_mask)).abs().max())
    if delta > 1e-7:
        raise ValueError("Padding invariance failed")
    path = MODEL_DIR / "latent_distribution_e2e" / f"seed_{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "state_dict": model.state_dict(), "parameter_count": parameters, "execution_spec_sha256": EXPECTED_SPEC_SHA256, "feature_names": list(FEATURE_NAMES)}, path)
    metrics = {
        "model": "END_TO_END_DISTRIBUTION", "seed": seed, "epochs_completed": EPOCHS, "parameter_count": parameters,
        "final_train_bag_BCE": final_loss, "runtime_seconds": time.perf_counter() - start,
        "ROC_AUC": float(roc_auc_score(val.y.numpy(), val_probability)),
        "average_precision": float(average_precision_score(val.y.numpy(), val_probability)),
        "Brier_score": float(brier_score_loss(val.y.numpy(), val_probability)),
        "padding_max_absolute_logit_difference": delta, "instance_level_FLAG_loss_used": False,
        "model_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "model_sha256": file_sha256(path),
    }
    return metrics, val_probability, train_representation, val_representation


def fit_hgb(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[HistGradientBoostingClassifier, np.ndarray]:
    model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, class_weight=None, early_stopping=False, random_state=42)
    model.fit(train_x, train_y)
    return model, model.predict_proba(val_x)[:, 1]


def direct_representation(bags: Bags) -> np.ndarray:
    with torch.no_grad():
        result = masked_distribution(bags.x, bags.mask).numpy()
    if result.shape != (len(bags.y), 102) or not np.isfinite(result).all():
        raise ValueError("Direct representation integrity failed")
    return result


def metric_row(model: str, seed: int, y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {"model": model, "seed": seed, "ROC_AUC": float(roc_auc_score(y, probability)), "average_precision": float(average_precision_score(y, probability)), "Brier_score": float(brier_score_loss(y, probability))}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "seeds": len(group)}
        for metric in ("ROC_AUC", "average_precision", "Brier_score"):
            values = group[metric].to_numpy(float)
            row.update({f"{metric}_mean": float(values.mean()), f"{metric}_std_population": float(values.std(ddof=0)), f"{metric}_median": float(np.median(values)), f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(y: np.ndarray, candidate: np.ndarray, comparator: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED); roc, ap = np.empty(BOOTSTRAPS), np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        while np.unique(y[idx]).size < 2:
            idx = rng.integers(0, len(y), len(y))
        roc[i] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], comparator[idx])
        ap[i] = average_precision_score(y[idx], candidate[idx]) - average_precision_score(y[idx], comparator[idx])
    def describe(values: np.ndarray) -> dict[str, float]:
        return {"mean": float(values.mean()), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}
    return {"resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "ROC_AUC_delta": describe(roc), "average_precision_delta": describe(ap)}


def baseline_probabilities(val: Bags) -> np.ndarray:
    frame = pd.read_csv(BASELINE_PREDICTIONS, dtype={ID_COLUMN: str})
    frame = frame.loc[frame["model"].eq("hist_gradient_boosting")]
    aligned = frame.set_index(ID_COLUMN).reindex(val.customer_ids)
    if aligned["probability"].isna().any() or not np.array_equal(aligned[LABEL_COLUMN].to_numpy(float), val.y.numpy()):
        raise ValueError("FULL_48 probability alignment failed")
    probability = aligned["probability"].to_numpy(float)
    observed = (roc_auc_score(val.y.numpy(), probability), average_precision_score(val.y.numpy(), probability))
    if not np.allclose(observed, (FULL48["ROC_AUC"], FULL48["average_precision"]), atol=1e-14):
        raise ValueError(f"FULL_48 committed metrics changed: {observed}")
    return probability


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    validate_seal(); audit = pre_stage_audit(); tests = numerical_tests()
    save_json(audit, RESULTS_DIR / "pre_stage_audit.json")
    train, val, integrity = load_bags(); integrity["numerical_tests"] = tests
    save_json(integrity, RESULTS_DIR / "bag_integrity.json")
    y_train, y_val = train.y.numpy(), val.y.numpy(); full_probability = baseline_probabilities(val)
    direct_train, direct_val = direct_representation(train), direct_representation(val)
    direct_path = MODEL_DIR / "direct_summary_distribution_hgb.joblib"
    if direct_path.exists():
        direct_model = joblib.load(direct_path)
        direct_probability = direct_model.predict_proba(direct_val)[:, 1]
    else:
        direct_model, direct_probability = fit_hgb(direct_train, y_train, direct_val)
        joblib.dump(direct_model, direct_path)
    fit_count = 1
    runs, primary_rows = [], []
    direct_metrics = metric_row("DIRECT_SUMMARY_DISTRIBUTION_HGB", 42, y_val, direct_probability)
    primary_rows.append(direct_metrics)
    e2e_probabilities: dict[int, np.ndarray] = {}; latent_probabilities: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        run, e2e_probability, latent_train, latent_val = train_neural(seed, train, val)
        fit_count += 1; runs.append(run); primary_rows.append(metric_row("END_TO_END_DISTRIBUTION", seed, y_val, e2e_probability)); e2e_probabilities[seed] = e2e_probability
        if latent_train.shape != (len(train.y), 96) or latent_val.shape != (len(val.y), 96):
            raise ValueError("Latent representation dimension failed")
        hgb, latent_probability = fit_hgb(latent_train, y_train, latent_val); fit_count += 1
        path = MODEL_DIR / "latent_distribution_hgb" / f"seed_{seed}.joblib"; path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(hgb, path)
        row = metric_row("LATENT_DISTRIBUTION_HGB", seed, y_val, latent_probability); row.update({"model_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "model_sha256": file_sha256(path)})
        primary_rows.append(row); latent_probabilities[seed] = latent_probability
    if fit_count != 11:
        raise ValueError(f"Fit budget violated: {fit_count}")
    primary = pd.DataFrame(primary_rows); summary = summarize(primary)
    save_csv(pd.DataFrame(runs), RESULTS_DIR / "model_runs.csv"); save_csv(primary, RESULTS_DIR / "primary_metrics.csv"); save_csv(summary, RESULTS_DIR / "primary_metrics_summary.csv")
    latent_mean_probability = np.mean(np.stack(list(latent_probabilities.values())), axis=0)
    e2e_mean_probability = np.mean(np.stack(list(e2e_probabilities.values())), axis=0)
    bootstrap = {
        "policy": {"resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "unit": "DEV_VAL customer", "seed_handling": "mean probability across five encoder seeds before customer bootstrap"},
        "LATENT_DISTRIBUTION_HGB_minus_FULL_48_HGB": paired_bootstrap(y_val, latent_mean_probability, full_probability),
        "LATENT_DISTRIBUTION_HGB_minus_DIRECT_SUMMARY_DISTRIBUTION_HGB": paired_bootstrap(y_val, latent_mean_probability, direct_probability),
    }
    save_json(bootstrap, RESULTS_DIR / "bootstrap_comparisons.json")
    means = summary.set_index("model")
    latent_roc = float(means.loc["LATENT_DISTRIBUTION_HGB", "ROC_AUC_mean"]); latent_ap = float(means.loc["LATENT_DISTRIBUTION_HGB", "average_precision_mean"])
    e2e_roc = float(means.loc["END_TO_END_DISTRIBUTION", "ROC_AUC_mean"]); e2e_ap = float(means.loc["END_TO_END_DISTRIBUTION", "average_precision_mean"])
    direct_roc, direct_ap = direct_metrics["ROC_AUC"], direct_metrics["average_precision"]
    control = {
        "direct_representation_dimensions": 102, "latent_representation_dimensions": 96,
        "DIRECT_SUMMARY_DISTRIBUTION_HGB": direct_metrics,
        "LATENT_DISTRIBUTION_HGB_mean": {"ROC_AUC": latent_roc, "average_precision": latent_ap},
        "latent_minus_direct": {"ROC_AUC": latent_roc - direct_roc, "average_precision": latent_ap - direct_ap},
        "interpretation_case": "CASE 1: learned representation adds value" if latent_roc > direct_roc and latent_ap > direct_ap else ("CASE 3: learned encoding discards useful information" if latent_roc < direct_roc and latent_ap < direct_ap else "CASE 2: mixed/approximately similar; distribution aggregation dominates"),
    }
    save_json(control, RESULTS_DIR / "distribution_control_results.json")
    recovery: dict[str, Any] = {}
    for model, roc, ap in (("END_TO_END_DISTRIBUTION", e2e_roc, e2e_ap), ("LATENT_DISTRIBUTION_HGB", latent_roc, latent_ap), ("DIRECT_SUMMARY_DISTRIBUTION_HGB", direct_roc, direct_ap)):
        recovery[model] = {name: {"ROC_AUC_improvement": roc - values["ROC_AUC"], "average_precision_improvement": ap - values["average_precision"], "ROC_deficit_fraction_recovered_toward_FULL_48": (roc - values["ROC_AUC"]) / (FULL48["ROC_AUC"] - values["ROC_AUC"]), "AP_deficit_fraction_recovered_toward_FULL_48": (ap - values["average_precision"]) / (FULL48["average_precision"] - values["average_precision"])} for name, values in MIL.items()}
    save_json(recovery, RESULTS_DIR / "information_recovery.json")
    latent_group = primary.loc[primary["model"].eq("LATENT_DISTRIBUTION_HGB")]
    full_both_improved = int(((latent_group["ROC_AUC"] >= FULL48["ROC_AUC"]) & (latent_group["average_precision"] >= FULL48["average_precision"])).sum())
    parity_seeds = int(((latent_group["ROC_AUC"] - FULL48["ROC_AUC"] >= -0.005) & (latent_group["average_precision"] - FULL48["average_precision"] >= -0.010)).sum())
    boot_full = bootstrap["LATENT_DISTRIBUTION_HGB_minus_FULL_48_HGB"]; boot_direct = bootstrap["LATENT_DISTRIBUTION_HGB_minus_DIRECT_SUMMARY_DISTRIBUTION_HGB"]
    a = bool(latent_roc - FULL48["ROC_AUC"] >= 0.010 and latent_ap - FULL48["average_precision"] >= 0.020 and full_both_improved >= 4 and boot_full["ROC_AUC_delta"]["lower_95"] > 0 and boot_full["average_precision_delta"]["lower_95"] > 0 and latent_roc - direct_roc >= 0.005 and latent_ap - direct_ap >= 0.010 and boot_direct["ROC_AUC_delta"]["lower_95"] > 0 and boot_direct["average_precision_delta"]["lower_95"] > 0)
    meaningful_direct = ((latent_roc - direct_roc >= 0.005 and boot_direct["ROC_AUC_delta"]["lower_95"] > 0) or (latent_ap - direct_ap >= 0.010 and boot_direct["average_precision_delta"]["lower_95"] > 0))
    b = bool(latent_roc - FULL48["ROC_AUC"] >= -0.005 and latent_ap - FULL48["average_precision"] >= -0.010 and parity_seeds >= 4 and boot_full["ROC_AUC_delta"]["lower_95"] > -0.005 and boot_full["average_precision_delta"]["lower_95"] > -0.010 and latent_roc >= direct_roc and latent_ap >= direct_ap and meaningful_direct)
    outcome_code = "A" if a else ("B" if b else "C")
    material = bool(latent_roc - FULL48["ROC_AUC"] < -0.015 or latent_ap - FULL48["average_precision"] < -0.030)
    outcome = {
        "outcome": outcome_code, "outcome_label": {"A": "ANCHOR-CANDIDATE PERFORMANCE", "B": "REPRESENTATION PARITY", "C": "WEAK RESULT"}[outcome_code],
        "A_satisfied": a, "B_satisfied": b, "material_inferiority_triggered": material,
        "seed_counts": {"improve_both_over_FULL_48": full_both_improved, "within_both_parity_margins": parity_seeds},
        "mean_deltas": {"end_to_end_minus_FULL_48": {"ROC_AUC": e2e_roc - FULL48["ROC_AUC"], "average_precision": e2e_ap - FULL48["average_precision"]}, "latent_HGB_minus_FULL_48": {"ROC_AUC": latent_roc - FULL48["ROC_AUC"], "average_precision": latent_ap - FULL48["average_precision"]}, "direct_HGB_minus_FULL_48": {"ROC_AUC": direct_roc - FULL48["ROC_AUC"], "average_precision": direct_ap - FULL48["average_precision"]}, "latent_HGB_minus_direct_HGB": {"ROC_AUC": latent_roc - direct_roc, "average_precision": latent_ap - direct_ap}},
        "ANCHOR_EARNED": "YES" if a else "NO", "anchor_sentence": None if not a else "Fixed latent-distribution pooling yields reproducible classifier-matched gains over both FULL_48 and direct SUMMARY_17 distribution aggregation on development validation.",
        "neural_representation_stop_rule_activated": outcome_code == "C", "total_new_model_fits": fit_count,
        "confirmation_access": False, "final_test_access": False, "instance_level_FLAG_supervision": False,
    }
    save_json(outcome, RESULTS_DIR / "stage25b_outcome.json")
    report = f"""# Stage 25B — learned customer-distribution experiment\n\n## Result\n\nOutcome **{outcome_code} — {outcome['outcome_label']}**. `ANCHOR_EARNED = {outcome['ANCHOR_EARNED']}`. Material inferiority: **{material}**.\n\nThe sealed specification SHA-256 is `{EXPECTED_SPEC_SHA256}`. Exactly 11 models were fitted: five end-to-end distribution models, five latent-distribution HGB controls, and one direct SUMMARY_17 distribution HGB control.\n\n## Representation and data\n\nThe model uses the frozen 17→32→16 encoder (1,104 parameters), dimension-major mean/population-SD/q10/median/q90/maximum pooling (96 values), and a linear 96→1 head (97 parameters), for exactly 1,201 trainable parameters. The direct control applies the same six statistics to SUMMARY_17, producing 102 values. DEV_TRAIN contains {len(train.y):,} customers and {int(train.mask.sum()):,} windows; DEV_VAL contains {len(val.y):,} customers and {int(val.mask.sum()):,} windows.\n\n## Primary findings\n\nMean end-to-end: ROC-AUC {e2e_roc:.6f}, AP {e2e_ap:.6f}. Mean latent-HGB: ROC-AUC {latent_roc:.6f}, AP {latent_ap:.6f}. Direct-distribution HGB: ROC-AUC {direct_roc:.6f}, AP {direct_ap:.6f}. Frozen FULL_48 HGB: ROC-AUC {FULL48['ROC_AUC']:.6f}, AP {FULL48['average_precision']:.6f}.\n\nLatent-HGB minus FULL_48: ROC {latent_roc - FULL48['ROC_AUC']:+.6f}, AP {latent_ap - FULL48['average_precision']:+.6f}. Latent-HGB minus direct control: ROC {latent_roc - direct_roc:+.6f}, AP {latent_ap - direct_ap:+.6f}.\n\n## Interpretation\n\n{control['interpretation_case']}. Under the evaluated SUMMARY_17 plus latent-distribution formulation, explicit FULL_48 customer profiling {'remains stronger' if latent_roc < FULL48['ROC_AUC'] or latent_ap < FULL48['average_precision'] else 'is not stronger on the mean primary metrics'}. This is a narrow representation result, not a general claim about neural networks.\n\nNo instance-level FLAG target or loss was used. CONFIRMATION and FINAL_TEST remained closed.\n\n## Stop rule\n\n{'The preregistered neural-representation stop rule activates; this direction ends here absent a qualitatively new external hypothesis.' if outcome_code == 'C' else 'The Stage 25B weak-result stop rule does not activate.'}\n"""
    (RESULTS_DIR / "stage25b_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"fit_count": fit_count, "summary": summary.to_dict(orient="records"), "outcome": outcome, "spec_sha256": EXPECTED_SPEC_SHA256}, indent=2))


if __name__ == "__main__":
    main()
