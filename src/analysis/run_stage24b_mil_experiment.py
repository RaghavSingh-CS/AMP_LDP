"""Execute the sealed Stage 24B weakly supervised MIL experiment.

The only row-level model-data loader is the original-train population, filtered
to DEV_TRAIN and DEV_VAL by the frozen Stage 17A governance manifest.  There is
no loader for CONFIRMATION or FINAL_TEST.  Loss is computed once per customer
bag; window labels are never constructed or used as targets.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
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


RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_24b"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_24b"
SPEC_PATH = RESULTS_DIR / "stage24b_execution_spec.json"
EXPECTED_SPEC_SHA256 = "15b815e32742540903977bae52d942bf7da438b11e7625f524b7a9e3cc16635f"
SOURCE_COMMIT = "e8aefe96369633612f07ae3a774fb3baded8c912"
GOVERNANCE_PATH = (
    PROJECT_ROOT / "results" / "governance" / "stage_17a" / "research_protocol_manifest.csv"
)
WINDOW_PATH = (
    PROJECT_ROOT / "data" / "windows" / "stride_30" / "usable" / "train_windows.parquet"
)
STAGE14_CONFIG = RESULTS_DIR.parent / "stage_14" / "preregistered_config.json"
STAGE24A_DIR = RESULTS_DIR.parent / "stage_24a"
BASELINE_PREDICTIONS = RESULTS_DIR.parent / "stage_17b" / "dev_val_customer_probabilities.csv"
ARMS = ("MEAN_MIL", "MAX_MIL", "GATED_ATTENTION_MIL")
SEEDS = (42, 43, 44, 45, 46)
EXPECTED_PARAMETERS = {"MEAN_MIL": 1121, "MAX_MIL": 1121, "GATED_ATTENTION_MIL": 1681}
MAX_BAG_SIZE = 34
FEATURE_COUNT = 17
EPOCHS = 50
BATCH_SIZE = 64
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2402


@dataclass
class Bags:
    role: str
    x: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    customer_ids: np.ndarray
    bag_sizes: np.ndarray
    window_ids: list[np.ndarray]
    raw_summaries: np.ndarray


class MILModel(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        self.arm = arm
        self.encoder = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        if arm == "GATED_ATTENTION_MIL":
            self.attention_v = nn.Linear(16, 16, bias=True)
            self.attention_u = nn.Linear(16, 16, bias=True)
            self.attention_w = nn.Linear(16, 1, bias=False)
        self.classifier = nn.Linear(16, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Enforce the sealed batch-local-padding semantics even though the
        # cached dataset tensor is stored at the global maximum width.  This
        # also makes appending wholly masked columns exactly invariant.
        local_width = int(mask.sum(dim=1).max().item())
        x = x[:, :local_width]
        mask = mask[:, :local_width]
        h = self.encoder(x)
        if self.arm == "MEAN_MIL":
            weights = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            attention = None
        elif self.arm == "MAX_MIL":
            pooled = h.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
            attention = None
        else:
            scores = self.attention_w(
                torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
            ).squeeze(-1)
            scores = scores.masked_fill(~mask, -torch.inf)
            attention = torch.softmax(scores, dim=1)
            attention = attention.masked_fill(~mask, 0.0)
            pooled = torch.sum(h * attention.unsqueeze(-1), dim=1)
        return self.classifier(pooled).squeeze(-1), attention


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def validate_seal() -> dict[str, Any]:
    if file_sha256(SPEC_PATH) != EXPECTED_SPEC_SHA256:
        raise ValueError("Stage 24B execution specification hash changed")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if tuple(spec["training"]["seeds"]) != SEEDS or spec["training"]["fits"] != 15:
        raise ValueError("Sealed run grid changed")
    if tuple(spec["architecture"]["pooling_arms"]) != ARMS:
        raise ValueError("Sealed pooling grid changed")
    return spec


def pre_stage_audit() -> dict[str, Any]:
    if git("branch", "--show-current") != "study2/mil":
        raise ValueError("Stage 24B must run on study2/mil")
    if git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("Stage 24B source HEAD changed")
    stage24a_hashes = {
        p.name: file_sha256(p) for p in sorted(STAGE24A_DIR.iterdir()) if p.is_file()
    }
    expected = {
        "anchor_result_criteria.json": "38d5358e3f1544340b915d256b69fabd38ec95f9465d9cde2b41d202e17c4726",
        "bag_statistics.json": "258a95741f1c89e2327b4f32dd59455212bf72554b74dc95d52670d0f53a3320",
        "instance_representation_audit.csv": "8efc9675c16eb0eb2fb6acc642661b07161cb3b5bfced30b774ddaefd7fa2fce",
        "mil_architecture_spec.json": "3ae58a5df36e01914c78348fc047582b64c354abe123d021f66520fc69b77029",
        "mil_feasibility_report.md": "7f777e48e8c52911bd26b100e99355f58228b122073b38d588b223c40b75d825",
        "stage24b_preregistration_draft.json": "b5a324d18167e82b7bdda8c53416cd735c14ec2769b604c4761048bab3ee0b07",
        "study2_governance.json": "4a065391b23fb80eae8a3d9c73096c3c1f9708048295ea73ed98f0f51637b345",
    }
    if stage24a_hashes != expected:
        raise ValueError("Stage 24A artifacts changed")
    protected_diff = git(
        "diff", "--name-only", SOURCE_COMMIT,
        "--", "results/supervised/stage_23a", "results/supervised/stage_23b", "models/supervised/stage_23b", "data"
    )
    if protected_diff:
        raise ValueError("Protected Study 1 paths changed")
    return {
        "stage": "Stage 24B pre-stage audit",
        "branch": "study2/mil",
        "source_HEAD": SOURCE_COMMIT,
        "git_clean_before_stage": True,
        "execution_spec_sha256": EXPECTED_SPEC_SHA256,
        "stage24a_artifact_hashes": stage24a_hashes,
        "stage1_protected_path_changes": [],
        "confirmation_row_level_access": False,
        "final_test_row_level_access": False,
    }


def bag_distribution(values: np.ndarray) -> dict[str, float | int]:
    q = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9], method="linear")
    return {
        "min": int(values.min()), "q10": float(q[0]), "q25": float(q[1]),
        "median": float(q[2]), "q75": float(q[3]), "q90": float(q[4]),
        "max": int(values.max()), "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
    }


def load_bags() -> tuple[Bags, Bags, dict[str, Any], np.ndarray, np.ndarray]:
    config = json.loads(STAGE14_CONFIG.read_text(encoding="utf-8"))
    if tuple(config["feature_names_in_order"]) != FEATURE_NAMES or len(FEATURE_NAMES) != 17:
        raise ValueError("SUMMARY_17 schema changed")
    governance = pd.read_csv(GOVERNANCE_PATH, dtype={ID_COLUMN: str})
    development = governance.loc[
        governance["research_role"].isin(["DEV_TRAIN", "DEV_VAL"]),
        [ID_COLUMN, "original_split", "research_role", "stage16_representation_available"],
    ].copy()
    if not development["original_split"].eq("train").all():
        raise ValueError("Development role includes a protected original split")
    manifest = pd.read_parquet(WINDOW_PATH, engine="pyarrow")
    if manifest["window_id"].duplicated().any():
        raise ValueError("Duplicate window IDs")
    role_map = development.set_index(ID_COLUMN)["research_role"]
    manifest["research_role"] = manifest[ID_COLUMN].astype(str).map(role_map)
    if manifest["research_role"].isna().any():
        raise ValueError("Window without development role")
    raw = load_window_matrix("train")
    if not np.array_equal(raw.window_ids, manifest["window_id"].astype(str).to_numpy()):
        raise ValueError("Window matrix order differs from manifest")
    summaries = behavioural_features(raw.X)
    if summaries.shape != (768923, 17) or not np.isfinite(summaries).all():
        raise ValueError("SUMMARY_17 matrix failed shape/finite checks")
    train_rows = manifest["research_role"].eq("DEV_TRAIN").to_numpy()
    mean = summaries[train_rows].mean(axis=0, dtype=np.float64)
    scale = summaries[train_rows].std(axis=0, dtype=np.float64, ddof=0)
    scale[scale == 0] = 1.0
    standardized = ((summaries.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized SUMMARY_17 is not finite")
    manifest["_row"] = np.arange(len(manifest), dtype=np.int64)

    def build(role: str) -> Bags:
        frame = manifest.loc[manifest["research_role"].eq(role)]
        groups = list(frame.groupby(ID_COLUMN, sort=False))
        n = len(groups)
        x = np.zeros((n, MAX_BAG_SIZE, FEATURE_COUNT), dtype=np.float32)
        raw_x = np.zeros_like(x)
        mask = np.zeros((n, MAX_BAG_SIZE), dtype=bool)
        y = np.empty(n, dtype=np.float32)
        ids: list[str] = []
        sizes = np.empty(n, dtype=np.int16)
        window_ids: list[np.ndarray] = []
        for i, (customer, group) in enumerate(groups):
            group = group.sort_values("window_number_within_customer", kind="stable")
            rows = group["_row"].to_numpy(dtype=np.int64)
            m = len(rows)
            if m > MAX_BAG_SIZE or group[LABEL_COLUMN].nunique() != 1:
                raise ValueError("Invalid customer bag")
            x[i, :m] = standardized[rows]
            raw_x[i, :m] = summaries[rows]
            mask[i, :m] = True
            y[i] = float(group[LABEL_COLUMN].iloc[0])
            ids.append(str(customer))
            sizes[i] = m
            window_ids.append(group["window_id"].astype(str).to_numpy())
        return Bags(
            role=role,
            x=torch.from_numpy(x), mask=torch.from_numpy(mask), y=torch.from_numpy(y),
            customer_ids=np.asarray(ids), bag_sizes=sizes, window_ids=window_ids,
            raw_summaries=raw_x,
        )

    train = build("DEV_TRAIN")
    val = build("DEV_VAL")
    if len(set(train.customer_ids) & set(val.customer_ids)):
        raise ValueError("DEV_TRAIN/DEV_VAL customer overlap")
    expected = {
        "DEV_TRAIN": (23724, 21702, 2022, 614976),
        "DEV_VAL": (5931, 5426, 505, 153947),
    }
    integrity: dict[str, Any] = {
        "stage": "Stage 24B bag integrity",
        "representation": "SUMMARY_17",
        "ordered_features": list(FEATURE_NAMES),
        "representation_dimension": 17,
        "nonfinite_values": int((~np.isfinite(standardized)).sum()),
        "duplicate_window_ids": int(manifest["window_id"].duplicated().sum()),
        "each_window_one_customer": bool(manifest.groupby("window_id")[ID_COLUMN].nunique().eq(1).all()),
        "DEV_TRAIN_DEV_VAL_customer_overlap": 0,
        "confirmation_customers_loaded": 0,
        "final_test_customers_loaded": 0,
        "instance_level_targets_created": False,
        "inputs": {
            "governance_sha256": file_sha256(GOVERNANCE_PATH),
            "window_manifest_sha256": file_sha256(WINDOW_PATH),
            "stage14_config_sha256": file_sha256(STAGE14_CONFIG),
        },
        "roles": {},
        "standardizer": {"fit_role": "DEV_TRAIN", "mean": mean.tolist(), "scale": scale.tolist()},
    }
    for bags in (train, val):
        observed = (len(bags.y), int((bags.y == 0).sum()), int((bags.y == 1).sum()), int(bags.mask.sum()))
        if observed != expected[bags.role]:
            raise ValueError(f"{bags.role} counts changed: {observed}")
        integrity["roles"][bags.role] = {
            "customers": observed[0], "normal_customers": observed[1],
            "theft_labelled_customers": observed[2], "windows": observed[3],
            "bag_size": bag_distribution(bags.bag_sizes),
        }
    return train, val, integrity, mean, scale


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def predict(
    model: MILModel, bags: Bags, override_mask: torch.Tensor | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    masks = bags.mask if override_mask is None else override_mask
    probabilities: list[np.ndarray] = []
    attention_parts: list[np.ndarray] = []
    for first in range(0, len(bags.y), BATCH_SIZE):
        last = min(first + BATCH_SIZE, len(bags.y))
        logits, attention = model(bags.x[first:last], masks[first:last])
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        if attention is not None:
            attention_parts.append(attention.cpu().numpy())
    return np.concatenate(probabilities), (
        np.concatenate(attention_parts) if attention_parts else None
    )


def padding_invariance(model: MILModel, bags: Bags) -> dict[str, Any]:
    model.eval()
    indices = np.linspace(0, len(bags.y) - 1, 8, dtype=int)
    x = bags.x[indices]
    mask = bags.mask[indices]
    extra_x = torch.cat([x, torch.zeros((len(indices), 7, FEATURE_COUNT))], dim=1)
    extra_mask = torch.cat([mask, torch.zeros((len(indices), 7), dtype=torch.bool)], dim=1)
    with torch.no_grad():
        logits, attention = model(x, mask)
        extra_logits, extra_attention = model(extra_x, extra_mask)
    logit_delta = float(torch.max(torch.abs(logits - extra_logits)).item())
    result: dict[str, Any] = {
        "bags_tested": len(indices), "extra_masked_rows": 7,
        "max_absolute_logit_difference": logit_delta,
        "tolerance": 1e-7, "passed": logit_delta <= 1e-7,
    }
    if attention is not None and extra_attention is not None:
        valid_sums = attention.sum(dim=1)
        extra_valid_sums = extra_attention[:, :MAX_BAG_SIZE].sum(dim=1)
        result.update({
            "maximum_valid_attention_sum_error": float(torch.max(torch.abs(valid_sums - 1)).item()),
            "maximum_extra_valid_attention_sum_error": float(torch.max(torch.abs(extra_valid_sums - 1)).item()),
            "maximum_padded_attention": (
                float(extra_attention[:, MAX_BAG_SIZE:].max().item())
                if extra_attention.shape[1] > MAX_BAG_SIZE
                else 0.0
            ),
        })
        result["passed"] = bool(
            result["passed"]
            and result["maximum_valid_attention_sum_error"] <= 1e-6
            and result["maximum_extra_valid_attention_sum_error"] <= 1e-6
            and result["maximum_padded_attention"] == 0.0
        )
    if not result["passed"]:
        raise ValueError("Padding-invariance test failed")
    return result


def train_one(arm: str, seed: int, train: Bags, val: Bags) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None, dict[str, Any]]:
    seed_everything(seed)
    model = MILModel(arm)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != EXPECTED_PARAMETERS[arm]:
        raise ValueError(f"{arm} parameter count {parameter_count} != {EXPECTED_PARAMETERS[arm]}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    final_loss = math.nan
    for _epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train.y), generator=generator)
        loss_sum = 0.0
        for first in range(0, len(order), BATCH_SIZE):
            idx = order[first : first + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(train.x[idx], train.mask[idx])
            loss = loss_function(logits, train.y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(idx)
        final_loss = loss_sum / len(train.y)
    runtime = time.perf_counter() - start
    probabilities, attention = predict(model, val)
    metrics = {
        "arm": arm, "seed": seed, "epochs_completed": EPOCHS,
        "parameter_count": parameter_count, "final_train_bag_BCE": final_loss,
        "runtime_seconds": runtime, "DEV_VAL_customers": len(val.y),
        "ROC_AUC": float(roc_auc_score(val.y.numpy(), probabilities)),
        "average_precision": float(average_precision_score(val.y.numpy(), probabilities)),
        "Brier_score": float(brier_score_loss(val.y.numpy(), probabilities)),
        "instance_level_FLAG_loss_used": False,
    }
    padding = padding_invariance(model, val)
    model_path = MODEL_DIR / arm.lower() / f"seed_{seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arm": arm, "seed": seed, "state_dict": model.state_dict(),
        "parameter_count": parameter_count, "execution_spec_sha256": EXPECTED_SPEC_SHA256,
        "feature_names": list(FEATURE_NAMES),
    }, model_path)
    metrics["model_path"] = str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    metrics["model_sha256"] = file_sha256(model_path)
    return metrics, probabilities, attention, padding


def summarize_primary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, group in metrics.groupby("arm", sort=False):
        row: dict[str, Any] = {"arm": arm, "seeds": len(group)}
        for metric in ("ROC_AUC", "average_precision", "Brier_score"):
            values = group[metric].to_numpy(float)
            row.update({
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std_population": float(values.std(ddof=0)),
                f"{metric}_median": float(np.median(values)),
                f"{metric}_min": float(values.min()),
                f"{metric}_max": float(values.max()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(
    y: np.ndarray, candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    roc_deltas = np.empty(BOOTSTRAPS)
    ap_deltas = np.empty(BOOTSTRAPS)
    for b in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        while np.unique(y[idx]).size < 2:
            idx = rng.integers(0, len(y), len(y))
        roc_deltas[b] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], baseline[idx])
        ap_deltas[b] = average_precision_score(y[idx], candidate[idx]) - average_precision_score(y[idx], baseline[idx])
    def item(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()), "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return {"resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "ROC_AUC_delta": item(roc_deltas), "average_precision_delta": item(ap_deltas)}


def concentration_rows(
    val: Bags, attentions: dict[int, np.ndarray], probabilities: dict[int, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, matrix in attentions.items():
        for i, m_value in enumerate(val.bag_sizes):
            m = int(m_value)
            weights = matrix[i, :m].astype(np.float64)
            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                raise ValueError("Attention weights do not normalize over valid instances")
            entropy = float(-(weights * np.log(weights.clip(1e-15))).sum())
            k = max(1, math.ceil(0.1 * m))
            ordered = np.sort(weights)[::-1]
            rows.append({
                ID_COLUMN: val.customer_ids[i], LABEL_COLUMN: int(val.y[i].item()),
                "seed": seed, "bag_size": m, "prediction": float(probabilities[seed][i]),
                "entropy": entropy,
                "normalized_entropy": 1.0 if m == 1 else entropy / math.log(m),
                "effective_number": math.exp(entropy),
                "top1_mass": float(ordered[0]),
                "top10_percent_count": k,
                "top10_percent_mass": float(ordered[:k].sum()),
            })
    return pd.DataFrame(rows)


def expected_random_jaccard(m: int, k: int) -> float:
    denominator = math.comb(m, k)
    total = 0.0
    for intersection in range(max(0, 2 * k - m), k + 1):
        probability = math.comb(k, intersection) * math.comb(m - k, k - intersection) / denominator
        union = 2 * k - intersection
        total += probability * (intersection / union if union else 1.0)
    return total


def attention_stability(
    val: Bags, attentions: dict[int, np.ndarray], predictions: dict[int, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_a, seed_b in itertools.combinations(SEEDS, 2):
        for i, m_value in enumerate(val.bag_sizes):
            m = int(m_value)
            k = max(1, math.ceil(0.1 * m))
            a = attentions[seed_a][i, :m]
            b = attentions[seed_b][i, :m]
            top_a = set(np.argpartition(a, -k)[-k:].tolist())
            top_b = set(np.argpartition(b, -k)[-k:].tolist())
            jaccard = len(top_a & top_b) / len(top_a | top_b)
            if m == 1:
                rank_correlation = 1.0
            else:
                rank_a = np.argsort(np.argsort(a)).astype(float)
                rank_b = np.argsort(np.argsort(b)).astype(float)
                rank_correlation = float(np.corrcoef(rank_a, rank_b)[0, 1])
            rows.append({
                ID_COLUMN: val.customer_ids[i], LABEL_COLUMN: int(val.y[i].item()),
                "seed_a": seed_a, "seed_b": seed_b, "bag_size": m,
                "top10_percent_count": k, "top_set_jaccard": jaccard,
                "matched_random_expected_jaccard": expected_random_jaccard(m, k),
                "attention_rank_correlation": rank_correlation,
            })
    details = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "seed_pairs": 10,
        "customer_pair_records": len(details),
        "top_set_jaccard": {},
        "attention_rank_correlation": {},
        "prediction_spearman": {},
    }
    for label, subset in [("overall", details), ("normal_FLAG_0", details.loc[details[LABEL_COLUMN].eq(0)]), ("theft_labelled_FLAG_1", details.loc[details[LABEL_COLUMN].eq(1)])]:
        summary["top_set_jaccard"][label] = describe(subset["top_set_jaccard"].to_numpy())
        summary["top_set_jaccard"][label]["matched_random_expected_mean"] = float(subset["matched_random_expected_jaccard"].mean())
        summary["attention_rank_correlation"][label] = describe(subset["attention_rank_correlation"].to_numpy())
    y = val.y.numpy().astype(int)
    for label_name, selector in [("overall", np.ones(len(y), bool)), ("normal_FLAG_0", y == 0), ("theft_labelled_FLAG_1", y == 1)]:
        correlations = []
        for a, b in itertools.combinations(SEEDS, 2):
            correlations.append(float(spearmanr(predictions[a][selector], predictions[b][selector]).statistic))
        summary["prediction_spearman"][label_name] = describe(np.asarray(correlations))
    summary["interpretation"] = "Model-relevance stability only; no theft-localization meaning."
    return details, summary


def describe(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(len(finite)), "mean": float(finite.mean()),
        "std_population": float(finite.std(ddof=0)), "q10": float(np.quantile(finite, 0.1)),
        "q25": float(np.quantile(finite, 0.25)), "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)), "q90": float(np.quantile(finite, 0.9)),
        "min": float(finite.min()), "max": float(finite.max()),
    }


def perturbation_for_seed(
    model: MILModel, seed: int, val: Bags, attention: np.ndarray, baseline: np.ndarray
) -> pd.DataFrame:
    eligible = val.bag_sizes >= 2
    top_mask = val.mask.clone()
    bottom_mask = val.mask.clone()
    removal_counts = np.zeros(len(val.y), dtype=np.int16)
    for i, m_value in enumerate(val.bag_sizes):
        m = int(m_value)
        if m < 2:
            continue
        k = min(math.ceil(0.1 * m), m - 1)
        removal_counts[i] = k
        values = attention[i, :m]
        top = np.argpartition(values, -k)[-k:]
        bottom = np.argpartition(values, k - 1)[:k]
        top_mask[i, top] = False
        bottom_mask[i, bottom] = False
    top_probability, _ = predict(model, val, top_mask)
    bottom_probability, _ = predict(model, val, bottom_mask)
    random_sum = np.zeros(len(val.y), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
    for _ in range(100):
        random_mask = val.mask.clone()
        random_scores = rng.random((len(val.y), MAX_BAG_SIZE))
        for i, m_value in enumerate(val.bag_sizes):
            m = int(m_value)
            k = int(removal_counts[i])
            if k:
                chosen = np.argpartition(random_scores[i, :m], k - 1)[:k]
                random_mask[i, chosen] = False
        random_probability, _ = predict(model, val, random_mask)
        random_sum += random_probability
    random_probability = random_sum / 100.0
    frame = pd.DataFrame({
        ID_COLUMN: val.customer_ids, LABEL_COLUMN: val.y.numpy().astype(int),
        "seed": seed, "bag_size": val.bag_sizes, "removal_count": removal_counts,
        "eligible": eligible, "baseline_probability": baseline,
        "top_removed_probability": top_probability,
        "random_removed_probability_mean_100": random_probability,
        "bottom_removed_probability": bottom_probability,
    })
    for condition, column in [
        ("top", "top_removed_probability"), ("random", "random_removed_probability_mean_100"),
        ("bottom", "bottom_removed_probability")
    ]:
        frame[f"{condition}_probability_drop"] = frame["baseline_probability"] - frame[column]
    return frame


def bootstrap_mean_difference(values: np.ndarray) -> dict[str, float | int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        idx = rng.integers(0, len(values), len(values))
        estimates[i] = values[idx].mean()
    return {
        "customers": len(values), "observed_mean": float(values.mean()),
        "bootstrap_mean": float(estimates.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def perturbation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame.loc[frame["eligible"]].copy()
    customer = eligible.groupby([ID_COLUMN, LABEL_COLUMN], as_index=False).agg(
        top_drop=("top_probability_drop", "mean"),
        random_drop=("random_probability_drop", "mean"),
        bottom_drop=("bottom_probability_drop", "mean"),
        bag_size=("bag_size", "first"), removal_count=("removal_count", "first"),
    )
    result: dict[str, Any] = {
        "metric": "baseline probability minus post-removal probability",
        "seeds_aggregated_per_customer": 5,
        "random_repeats_per_seed_customer": 100,
        "single_instance_bags_excluded": int((~frame.drop_duplicates(ID_COLUMN)["eligible"]).sum()),
        "groups": {},
    }
    for name, subset in [("overall", customer), ("normal_FLAG_0", customer.loc[customer[LABEL_COLUMN].eq(0)]), ("theft_labelled_FLAG_1", customer.loc[customer[LABEL_COLUMN].eq(1)])]:
        result["groups"][name] = {
            "TOP": describe(subset["top_drop"].to_numpy()),
            "RANDOM": describe(subset["random_drop"].to_numpy()),
            "BOTTOM": describe(subset["bottom_drop"].to_numpy()),
            "TOP_minus_RANDOM": bootstrap_mean_difference((subset["top_drop"] - subset["random_drop"]).to_numpy()),
            "TOP_minus_BOTTOM": bootstrap_mean_difference((subset["top_drop"] - subset["bottom_drop"]).to_numpy()),
        }
    result["interpretation"] = "Perturbation measures model dependence on ranked instances, not theft localization."
    return result


def feature_attention_associations(val: Bags, attentions: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    labels = val.y.numpy().astype(int)
    for seed, attention in attentions.items():
        for feature_index, feature in enumerate(FEATURE_NAMES):
            for group_name, selector in [("overall", np.ones(len(labels), bool)), ("normal_FLAG_0", labels == 0), ("theft_labelled_FLAG_1", labels == 1)]:
                weights: list[np.ndarray] = []
                values: list[np.ndarray] = []
                within: list[float] = []
                for i in np.flatnonzero(selector):
                    m = int(val.bag_sizes[i])
                    w = attention[i, :m]
                    v = val.raw_summaries[i, :m, feature_index]
                    weights.append(w)
                    values.append(v)
                    if m >= 3 and np.ptp(v) > 0:
                        rho = float(spearmanr(w, v).statistic)
                        if np.isfinite(rho):
                            within.append(rho)
                pooled = float(spearmanr(np.concatenate(weights), np.concatenate(values)).statistic)
                records.append({
                    "seed": seed, "group": group_name, "feature": feature,
                    "pooled_spearman": pooled,
                    "median_within_customer_spearman": float(np.median(within)) if within else math.nan,
                    "customers_with_defined_within_correlation": len(within),
                })
    return records


def trivial_controls(
    val: Bags, all_probabilities: dict[tuple[str, int], np.ndarray],
    attentions: dict[int, np.ndarray], concentration: pd.DataFrame,
    primary: pd.DataFrame, associations: list[dict[str, Any]], padding: list[dict[str, Any]],
) -> dict[str, Any]:
    y = val.y.numpy().astype(int)
    bag_size_records: list[dict[str, Any]] = []
    strata_records: list[dict[str, Any]] = []
    stratum_defs = [("1-10", 1, 10), ("11-27", 11, 27), ("28-34", 28, 34)]
    for arm in ARMS:
        for seed in SEEDS:
            p = all_probabilities[(arm, seed)]
            for group, selector in [("overall", np.ones(len(y), bool)), ("normal_FLAG_0", y == 0), ("theft_labelled_FLAG_1", y == 1)]:
                bag_size_records.append({
                    "arm": arm, "seed": seed, "group": group,
                    "prediction_bag_size_spearman": float(spearmanr(p[selector], val.bag_sizes[selector]).statistic),
                })
            for stratum, lower, upper in stratum_defs:
                selector = (val.bag_sizes >= lower) & (val.bag_sizes <= upper)
                row: dict[str, Any] = {
                    "arm": arm, "seed": seed, "stratum": stratum,
                    "customers": int(selector.sum()), "theft_labelled": int(y[selector].sum()),
                }
                if np.unique(y[selector]).size == 2:
                    row["ROC_AUC"] = float(roc_auc_score(y[selector], p[selector]))
                    row["average_precision"] = float(average_precision_score(y[selector], p[selector]))
                strata_records.append(row)
    attention_bag_size = []
    for seed, group in concentration.groupby("seed"):
        attention_bag_size.append({
            "seed": int(seed),
            "top10_mass_bag_size_spearman": float(spearmanr(group["top10_percent_mass"], group["bag_size"]).statistic),
            "normalized_entropy_bag_size_spearman": float(spearmanr(group["normalized_entropy"], group["bag_size"]).statistic),
        })
    association_frame = pd.DataFrame(associations)
    strongest = association_frame.iloc[association_frame["pooled_spearman"].abs().argmax()].to_dict()
    short = val.bag_sizes <= 5
    short_records = []
    for arm in ARMS:
        for seed in SEEDS:
            p = all_probabilities[(arm, seed)][short]
            row = {"arm": arm, "seed": seed, "customers": int(short.sum()), "theft_labelled": int(y[short].sum())}
            if np.unique(y[short]).size == 2:
                row["ROC_AUC"] = float(roc_auc_score(y[short], p))
                row["average_precision"] = float(average_precision_score(y[short], p))
            short_records.append(row)
    mean_summary = primary.set_index("arm")
    return {
        "bag_size": {
            "class_distribution": {
                "normal_FLAG_0": bag_distribution(val.bag_sizes[y == 0]),
                "theft_labelled_FLAG_1": bag_distribution(val.bag_sizes[y == 1]),
            },
            "prediction_correlations": bag_size_records,
            "attention_concentration_correlations": attention_bag_size,
            "performance_by_predeclared_stratum": strata_records,
        },
        "consumption_and_variability": {
            "attention_feature_associations": associations,
            "strongest_absolute_pooled_association": strongest,
            "approximately_reducible_threshold": "absolute pooled Spearman >= 0.90",
            "approximately_reducible": bool(abs(float(strongest["pooled_spearman"])) >= 0.90),
        },
        "max_pooling": {
            "MAX_minus_ATTENTION_mean_ROC_AUC": float(mean_summary.loc["MAX_MIL", "ROC_AUC_mean"] - mean_summary.loc["GATED_ATTENTION_MIL", "ROC_AUC_mean"]),
            "MAX_minus_ATTENTION_mean_average_precision": float(mean_summary.loc["MAX_MIL", "average_precision_mean"] - mean_summary.loc["GATED_ATTENTION_MIL", "average_precision_mean"]),
        },
        "short_bags_at_most_5": {
            "customers": int(short.sum()), "theft_labelled": int(y[short].sum()),
            "warning": "Very small subgroup; metrics are descriptive and unstable.",
            "metrics": short_records,
        },
        "padding_invariance": padding,
        "all_padding_tests_passed": all(item["passed"] for item in padding),
    }


def determine_outcome(
    metrics: pd.DataFrame, summary: pd.DataFrame, bootstraps: dict[str, Any],
    stability: dict[str, Any], perturbation: dict[str, Any]
) -> dict[str, Any]:
    baseline_roc = 0.7888308218953116
    baseline_ap = 0.35979459032174654
    ranked = summary.sort_values(["average_precision_mean", "ROC_AUC_mean"], ascending=False)
    selected = str(ranked.iloc[0]["arm"])
    selected_summary = ranked.iloc[0]
    runs = metrics.loc[metrics["arm"].eq(selected)]
    roc_delta = float(selected_summary["ROC_AUC_mean"] - baseline_roc)
    ap_delta = float(selected_summary["average_precision_mean"] - baseline_ap)
    positive_both = int(((runs["ROC_AUC"] > baseline_roc) & (runs["average_precision"] > baseline_ap)).sum())
    within_both = int(((runs["ROC_AUC"] >= baseline_roc - 0.01) & (runs["average_precision"] >= baseline_ap - 0.02)).sum())
    bootstrap = bootstraps[selected]
    outcome_a = bool(
        roc_delta >= 0.01 and ap_delta >= 0.02 and positive_both >= 4
        and bootstrap["ROC_AUC_delta"]["lower_95"] > 0
        and bootstrap["average_precision_delta"]["lower_95"] > 0
    )
    overall_perturbation = perturbation["groups"]["overall"]
    relevance_stable = bool(
        stability["top_set_jaccard"]["overall"]["mean"]
        > stability["top_set_jaccard"]["overall"]["matched_random_expected_mean"]
        and stability["attention_rank_correlation"]["overall"]["mean"] > 0
    )
    perturbation_supported = bool(overall_perturbation["TOP_minus_RANDOM"]["lower_95"] > 0)
    outcome_b = bool(
        selected == "GATED_ATTENTION_MIL" and roc_delta >= -0.01 and ap_delta >= -0.02
        and within_both >= 4 and relevance_stable and perturbation_supported
    )
    outcome = "A" if outcome_a else ("B" if outcome_b else "C")
    label = {
        "A": "STRONG DISCRIMINATION RESULT",
        "B": "COMPARABLE DISCRIMINATION + DISTINCT WEAK-SUPERVISION RESULT",
        "C": "WEAK RESULT",
    }[outcome]
    return {
        "outcome": outcome, "outcome_label": label,
        "selected_arm_by_frozen_rule": selected,
        "selected_mean_ROC_AUC": float(selected_summary["ROC_AUC_mean"]),
        "selected_mean_average_precision": float(selected_summary["average_precision_mean"]),
        "delta_vs_FULL_48_HGB": {"ROC_AUC": roc_delta, "average_precision": ap_delta},
        "seeds_positive_on_both_metrics": positive_both,
        "seeds_within_both_comparable_margins": within_both,
        "paired_bootstrap": bootstrap,
        "attention_relevance_stable_above_random": relevance_stable,
        "top_removal_exceeds_random_with_positive_interval": perturbation_supported,
        "outcome_A_passed": outcome_a, "outcome_B_passed": outcome_b,
        "ANCHOR_EARNED": "NO" if outcome == "C" else "YES",
        "anchor_sentence": None,
        "anchor_policy": "A final sentence is written only after inspecting which preregistered criterion passed; no sentence is manufactured for Outcome C.",
        "instance_level_FLAG_loss_used": False,
        "confirmation_accessed": False, "final_test_accessed": False,
    }


def main() -> None:
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    validate_seal()
    audit = pre_stage_audit()
    save_json(audit, RESULTS_DIR / "pre_stage_audit.json")
    train, val, integrity, mean, scale = load_bags()
    save_json(integrity, RESULTS_DIR / "bag_integrity.json")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    all_probabilities: dict[tuple[str, int], np.ndarray] = {}
    attention_by_seed: dict[int, np.ndarray] = {}
    attention_probabilities: dict[int, np.ndarray] = {}
    padding_records: list[dict[str, Any]] = []
    models: dict[int, MILModel] = {}
    for arm in ARMS:
        for seed in SEEDS:
            print(f"TRAINING {arm} seed={seed}", flush=True)
            record, probabilities, attention, padding = train_one(arm, seed, train, val)
            run_records.append(record)
            all_probabilities[(arm, seed)] = probabilities
            padding_records.append({"arm": arm, "seed": seed, **padding})
            if arm == "GATED_ATTENTION_MIL":
                if attention is None:
                    raise ValueError("Attention model returned no attention")
                attention_by_seed[seed] = attention
                attention_probabilities[seed] = probabilities
                model = MILModel(arm)
                payload = torch.load(MODEL_DIR / arm.lower() / f"seed_{seed}.pt", weights_only=True)
                model.load_state_dict(payload["state_dict"])
                models[seed] = model
            save_csv(pd.DataFrame(run_records), RESULTS_DIR / "training_runs.partial.csv")

    runs = pd.DataFrame(run_records)
    if len(runs) != 15 or not runs.groupby("arm")["seed"].apply(lambda x: set(x) == set(SEEDS)).all():
        raise ValueError("The 15-run grid is incomplete")
    save_csv(runs, RESULTS_DIR / "training_runs.csv")
    save_csv(runs[["arm", "seed", "ROC_AUC", "average_precision", "Brier_score"]], RESULTS_DIR / "primary_metrics.csv")
    summary = summarize_primary(runs)
    save_csv(summary, RESULTS_DIR / "primary_metrics_summary.csv")

    predictions = pd.DataFrame({ID_COLUMN: val.customer_ids, LABEL_COLUMN: val.y.numpy().astype(int), "bag_size": val.bag_sizes})
    for arm in ARMS:
        for seed in SEEDS:
            predictions[f"{arm}__seed_{seed}"] = all_probabilities[(arm, seed)]
    save_csv(predictions, RESULTS_DIR / "dev_val_predictions.csv")

    baseline_frame = pd.read_csv(BASELINE_PREDICTIONS, dtype={ID_COLUMN: str})
    if "model" in baseline_frame.columns:
        baseline_frame = baseline_frame.loc[
            baseline_frame["model"].eq("hist_gradient_boosting")
        ].copy()
    probability_columns = [c for c in baseline_frame.columns if "prob" in c.lower()]
    if not probability_columns:
        raise ValueError("Could not find committed baseline probability column")
    baseline_column = "hist_gradient_boosting_probability" if "hist_gradient_boosting_probability" in probability_columns else probability_columns[0]
    aligned = pd.DataFrame({ID_COLUMN: val.customer_ids}).merge(
        baseline_frame[[ID_COLUMN, baseline_column]], on=ID_COLUMN, how="left", validate="one_to_one"
    )
    if aligned[baseline_column].isna().any():
        raise ValueError("Baseline probabilities do not align to DEV_VAL")
    y = val.y.numpy().astype(int)
    baseline = aligned[baseline_column].to_numpy(float)
    bootstrap_results: dict[str, Any] = {}
    for arm in ARMS:
        ensemble = np.mean([all_probabilities[(arm, seed)] for seed in SEEDS], axis=0)
        bootstrap_results[arm] = paired_bootstrap(y, ensemble, baseline)
    save_json({"comparisons": bootstrap_results}, RESULTS_DIR / "primary_bootstrap_summary.json")

    concentration = concentration_rows(val, attention_by_seed, attention_probabilities)
    save_csv(concentration, RESULTS_DIR / "attention_concentration.csv")
    stability_details, stability_summary = attention_stability(val, attention_by_seed, attention_probabilities)
    save_csv(stability_details, RESULTS_DIR / "attention_stability_details.csv")
    save_json(stability_summary, RESULTS_DIR / "attention_stability_summary.json")

    perturbation_parts = []
    for seed in SEEDS:
        print(f"PERTURBATION attention seed={seed}", flush=True)
        perturbation_parts.append(
            perturbation_for_seed(models[seed], seed, val, attention_by_seed[seed], attention_probabilities[seed])
        )
    perturbation = pd.concat(perturbation_parts, ignore_index=True)
    save_csv(perturbation, RESULTS_DIR / "perturbation_results.csv")
    perturb_summary = perturbation_summary(perturbation)
    save_json(perturb_summary, RESULTS_DIR / "perturbation_summary.json")

    associations = feature_attention_associations(val, attention_by_seed)
    controls = trivial_controls(
        val, all_probabilities, attention_by_seed, concentration, summary,
        associations, padding_records,
    )
    save_json(controls, RESULTS_DIR / "trivial_controls.json")
    outcome = determine_outcome(runs, summary, bootstrap_results, stability_summary, perturb_summary)
    save_json(outcome, RESULTS_DIR / "stage24b_outcome.json")
    partial = RESULTS_DIR / "training_runs.partial.csv"
    if partial.exists():
        partial.unlink()
    print("STAGE24B_EXECUTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
