# AMP table and figure plan

No new analysis is authorized by this plan. Every plotted or tabulated value
must come from an aggregate artifact listed in `AMP_PAPER_EVIDENCE_INDEX.md`.

## Main tables

### Table 1 — governed cohort performance

Rows: `FULL_48`, `DIRECT`, `AMP-LDP (LATENT)`.

Column groups: DEV_VAL ROC/AP, CONFIRMATION ROC/AP, and final ROC/AP.

Exact rounded values:

| Method | DEV ROC | DEV AP | CONF ROC | CONF AP | Final ROC | Final AP |
|---|---:|---:|---:|---:|---:|---:|
| FULL_48 | 0.7888 | 0.3598 | 0.7891 | 0.3791 | 0.7979 | 0.3853 |
| DIRECT | 0.7882 | 0.3492 | 0.7870 | 0.3557 | 0.7950 | 0.3684 |
| AMP-LDP | 0.7908 | 0.3678 | 0.8118 | 0.4153 | 0.8117 | 0.4219 |

### Table 2 — final paired effects

Rows: AMP-LDP−FULL_48, AMP-LDP−DIRECT, DIRECT−FULL_48. Columns: observed ROC
delta, ROC 95% interval, observed AP delta, AP 95% interval. This remains
separate from Table 1 to keep the central table readable.

| Comparison | ROC delta [95% CI] | AP delta [95% CI] |
|---|---:|---:|
| AMP-LDP − FULL_48 | +0.0139 [+0.0005, +0.0275] | +0.0365 [+0.0080, +0.0652] |
| AMP-LDP − DIRECT | +0.0168 [+0.0063, +0.0279] | +0.0534 [+0.0274, +0.0779] |
| DIRECT − FULL_48 | −0.0029 [−0.0161, +0.0101] | −0.0169 [−0.0436, +0.0124] |

### Table 3 — representation diagnostic

Rows and values are specified in `AMP_RESULTS_FREEZE.md`: MEAN_MIL,
MAX_MIL, GATED_ATTENTION_MIL, DIRECT, and AMP-LDP on DEV_VAL. A prominent note
states that MIL versus HGB rows are diagnostic rather than perfectly controlled;
DIRECT versus AMP-LDP is the mechanistic matched comparison.

## Minimum useful figure set

### Figure 1 — label granularity and final architecture (main)

- **Purpose:** explain why the prediction unit is the customer and how AMP-LDP
  avoids instance-level theft supervision.
- **Source:** frozen method metadata only; no row-level data.
- **Panels:** (a) customer `FLAG` versus unlabeled 30-day windows; (b)
  SUMMARY_17 → 17–32–16 encoder → six-statistic latent aggregation → 96
  customer features → HGB; (c) arithmetic five-seed probability mean.
- **Message:** windows supply behaviours, but supervision and output remain
  customer-level.

### Figure 2 — information-bottleneck comparison (main)

- **Purpose:** show the methodological transition from one-vector MIL to
  distribution-preserving DIRECT and AMP-LDP.
- **Source:** Stage 24B `primary_metrics_summary.csv` and Stage 25B
  `primary_metrics_summary.csv`.
- **Axes/panels:** two aligned point panels for DEV_VAL ROC and AP; methods on
  the vertical axis. Use points, not invented uncertainty bars.
- **Message:** multi-statistic customer distributions recover information that
  mean/max/attention single-vector pooling did not retain.

### Figure 3 — cross-cohort metric stability (main)

- **Purpose:** display persistence without suggesting exact effect equality.
- **Source:** Stage 27B `final_stage_stability.json`.
- **Axes/panels:** cohort on x-axis (DEV_VAL, CONFIRMATION, final), metric on
  y-axis, one ROC panel and one AP panel, lines for FULL_48, DIRECT, AMP-LDP.
- **Message:** `AMP-LDP > FULL_48 > DIRECT` holds for both metrics in all three
  governed roles; magnitude varies.

### Figure 4 — final paired effect intervals (main or supplement if space-limited)

- **Purpose:** make uncertainty around the terminal comparisons explicit.
- **Source:** Stage 27B `final_bootstrap_comparisons.json`.
- **Axes/panels:** horizontal delta axis with zero reference; separate ROC and
  AP panels; observed effects and percentile intervals for three comparisons.
- **Message:** AMP-LDP's final gains over both controls have positive lower
  bounds; DIRECT versus FULL_48 is uncertain.

## Supplementary figures

- Full reconstruction architecture comparison.
- Stage 18 history/consumption subgroup plots already committed.
- Per-seed ROC/AP point plot using committed Stage 27B seed metrics.
- Optional feature-family and compaction plots already committed; do not create
  additional analytical variants.
