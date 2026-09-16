# AMP paper results freeze

## Reporting policy

Paper tables use four decimal places for ROC-AUC, Average Precision (AP),
effect sizes, and confidence-interval bounds. Customer and window counts are
integers. Raw committed values remain unchanged and are indexed in
`AMP_PAPER_EVIDENCE_INDEX.md`. AP is used rather than the ambiguous label
“PR-AUC” unless an historical artifact is being named.

## Two distinct FULL_48 results

These results must never be conflated:

- **Study 1 terminal result:** the Stage 23B FULL_48 HGB was refitted on
  representation-available `DEV_TRAIN + DEV_VAL` and scored `0.7925` ROC-AUC /
  `0.3702` AP on the final evaluation cohort with disclosed historical baseline
  access.
- **Study 2 comparator:** the original Stage 17B FULL_48 HGB was fitted on
  `DEV_TRAIN` only, never refitted, and scored `0.7979/0.3853` in the paired
  Stage 27B comparison. LATENT and DIRECT were also DEV_TRAIN-fitted, preserving
  symmetric training-population governance.

The Stage 23B number is Study 1 context. It is not substituted into Study 2's
paired comparison.

## Primary manuscript table

**Table 1. Customer-level discrimination across governed cohorts.**

| Method | DEV_VAL ROC | DEV_VAL AP | CONFIRMATION ROC | CONFIRMATION AP | Final ROC | Final AP |
|---|---:|---:|---:|---:|---:|---:|
| FULL_48 | 0.7888 | 0.3598 | 0.7891 | 0.3791 | 0.7979 | 0.3853 |
| DIRECT | 0.7882 | 0.3492 | 0.7870 | 0.3557 | 0.7950 | 0.3684 |
| AMP-LDP (LATENT) | **0.7908** | **0.3678** | **0.8118** | **0.4153** | **0.8117** | **0.4219** |

DEV_VAL and CONFIRMATION contain `5,931` and `6,355` customers. The final
evaluation cohort with disclosed historical baseline access contains `6,356`.
All Study 2 rows use frozen DEV_TRAIN-fitted models. AMP-LDP is the arithmetic
mean of seeds 42–46.

Paired effects belong in a second compact table because mixing cohort metrics
and uncertainty in Table 1 would obscure the central comparison.

**Table 2. Final paired customer-level effects (2,000 resamples, seed 2502).**

| Comparison | ROC delta | ROC 95% CI | AP delta | AP 95% CI |
|---|---:|---:|---:|---:|
| AMP-LDP − FULL_48 | +0.0139 | [+0.0005, +0.0275] | +0.0365 | [+0.0080, +0.0652] |
| AMP-LDP − DIRECT | +0.0168 | [+0.0063, +0.0279] | +0.0534 | [+0.0274, +0.0779] |
| DIRECT − FULL_48 | −0.0029 | [−0.0161, +0.0101] | −0.0169 | [−0.0436, +0.0124] |

## Diagnostic / ablation table

**Table 3. Why a distribution-preserving representation was needed (DEV_VAL).**

| Method | Customer representation / aggregation | Classifier form | ROC | AP |
|---|---|---|---:|---:|
| MEAN_MIL | One mean-pooled learned bag vector | Neural MIL head | 0.7399 | 0.2678 |
| MAX_MIL | One max-pooled learned bag vector | Neural MIL head | 0.7567 | 0.3146 |
| GATED_ATTENTION_MIL | One attention-weighted learned bag vector | Neural MIL head | 0.7581 | 0.2975 |
| DIRECT | Six statistics over each SUMMARY_17 dimension (102 features) | HGB | 0.7882 | 0.3492 |
| AMP-LDP (LATENT) | Six statistics over each learned latent dimension (96 features) | HGB | **0.7908** | **0.3678** |

The first three rows and final two rows are not perfectly controlled: the
classifier and representation dimensions differ. The table's defensible message
is narrower—single-vector pooling discarded useful bag-distribution information,
while multi-statistic aggregation restored performance; DIRECT then isolates the
incremental value of learned encoding within the matched distribution/HGB design.

## Historical numerical checks

- Stage 13 final window result: `0.6576929282` ROC / `0.2053047543` AP.
- Stage 13 customer-mean aggregation: `0.7023135402` ROC /
  `0.2193165970` AP. The paper may round the ROC to `0.7023`; it must not label
  the window-level number as customer-level.
- Study 1 Stage 23B final FULL_48: `0.7925125381` ROC /
  `0.3701585998` AP.
- Study 2 exact DEV_VAL, CONFIRMATION, final, effect, and CI values all match
  their authoritative committed aggregate artifacts.

No numerical discrepancy was found among the central values supplied for Stage
28. Differences are rounding only. The external STL-Net figures mentioned in
the Stage 28 instructions are not present in tracked repository documentation;
they are excluded from paper-facing quantitative comparison until a citable
source is established.

## Stability and seed robustness

The ordering `AMP-LDP > FULL_48 > DIRECT` holds for both metrics in all three
Study 2 cohorts. AMP-LDP minus FULL_48 ROC/AP effects are
`+0.0020/+0.0080`, `+0.0227/+0.0362`, and `+0.0139/+0.0365` across DEV_VAL,
CONFIRMATION, and final evaluation. AMP-LDP minus DIRECT effects are
`+0.0026/+0.0186`, `+0.0248/+0.0596`, and `+0.0168/+0.0534`.

On final evaluation, four of five individual seeds satisfied both frozen
FULL_48 parity margins. Seed ROC mean±population-SD was `0.7987±0.0037`; AP was
`0.3891±0.0080`. No preregistered seed-collapse condition triggered. The
ensemble outperformed every individual seed numerically; this is reported as an
ensemble result, not as proof that ensembling will always help.

## Terminal result

Stage 27B is **FINAL A — STRONG FINAL REPLICATION** and the terminal anchor
survives. The result supports Level 1, a strong methodological result under the
frozen Stage 27A interpretation. It remains bounded by the disclosures and
limitations in `AMP_LIMITATIONS.md`.
