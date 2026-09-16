# Stage 24B — Preregistered Weakly Supervised MIL Development Experiment

## Decision

**C — WEAK RESULT. ANCHOR_EARNED = NO.**

All 15 preregistered fits completed, but no MIL arm was broadly competitive
with the frozen `FULL_48` HGB baseline under the Stage 24A margins. The arm
selected by the sealed five-seed mean-AP/then-ROC rule was `MAX_MIL`, with mean
ROC-AUC `0.756666982953` and mean Average Precision `0.314566036048`. These are
`-0.032163838942` and `-0.045228554273` below the frozen baseline. The attention
arm learned stable, perturbation-supported model relevance, but its
discrimination was insufficient for Outcome B. No paper-anchor sentence was
earned.

Attention and perturbation results below describe model behaviour only. They do
not identify fraudulent windows, theft periods, event probabilities, or
localization accuracy.

## Governance and sealed execution

The stage began from a clean `study2/mil` branch at exact HEAD
`e8aefe96369633612f07ae3a774fb3baded8c912`. All Stage 24A hashes and protected
Study 1 paths passed the pre-stage audit. The final execution specification was
sealed before training at SHA-256
`15b815e32742540903977bae52d942bf7da438b11e7625f524b7a9e3cc16635f`.
The Stage 24B authorization explicitly fixed seeds `42–46`, replacing the
earlier draft seed set before any model was trained or scored.

Only original-training rows assigned to `DEV_TRAIN` or `DEV_VAL` were loaded.
`CONFIRMATION` and `FINAL_TEST` had no row-level loading path. No Study 1 or
Stage 24A artifact was overwritten.

An initial execution attempt stopped at the sealed padding-invariance assertion
before completing the run grid. Performance outputs were not inspected. The
forward path was corrected to slice tensors to the actual batch-local maximum
valid length, exactly enforcing the sealed batching rule, and all 15 runs were
restarted from seed 42. The final grid uses one identical corrected
implementation.

## Data and bag integrity

The exact frozen Stage 14 `SUMMARY_17` feature order was used. Feature-wise
standardization was fitted on `DEV_TRAIN` instances only and applied unchanged
to `DEV_VAL`.

| Role | Customers | Normal | Theft-labelled | Windows | Bag min / median / max | Mean bag size |
|---|---:|---:|---:|---:|---:|---:|
| `DEV_TRAIN` | 23,724 | 21,702 | 2,022 | 614,976 | 1 / 32 / 34 | 25.9221 |
| `DEV_VAL` | 5,931 | 5,426 | 505 | 153,947 | 1 / 32 / 34 | 25.9563 |

Every window mapped to exactly one customer, window IDs were unique, the
customer overlap was zero, and all representation values were finite. No new
feature, window, duplicate, truncation, or label-dependent sample was created.
No instance-level target was created.

## Frozen architecture and training

Every arm used `Linear(17,32) → ReLU → Dropout(0.1) → Linear(32,16) → ReLU`,
followed by a `Linear(16,1)` customer classifier. Mean and max models had exactly
1,121 trainable parameters. Gated attention used the frozen 16-dimensional
`V/U/w` operator and had exactly 1,681.

Training used unweighted customer-bag `BCEWithLogitsLoss`, AdamW, learning rate
`0.001`, weight decay `0.0001`, batch size 64 customers, global gradient clip
`5.0`, exactly 50 epochs, no early stopping, and seeds `42,43,44,45,46`.
Every customer appeared once per epoch. Total fit time was 1,565.1 seconds:
461.4 seconds for mean, 478.8 for max, and 624.8 for attention.

Padding was batch-local and mask-aware. Mean denominators counted valid
instances, max excluded padding, and attention softmax normalized only over
valid instances. All 15 padding tests had exact zero logit change after seven
additional masked rows; every attention sum and padded-weight assertion passed.

## Primary DEV_VAL results

| Arm | Seed | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| `MEAN_MIL` | 42 | 0.741932 | 0.267127 |
| | 43 | 0.741558 | 0.263256 |
| | 44 | 0.742599 | 0.272275 |
| | 45 | 0.738336 | 0.270146 |
| | 46 | 0.734909 | 0.266302 |
| `MAX_MIL` | 42 | 0.758909 | 0.314577 |
| | 43 | 0.754409 | 0.308012 |
| | 44 | 0.744707 | 0.291647 |
| | 45 | 0.758841 | 0.318743 |
| | 46 | 0.766468 | 0.339851 |
| `GATED_ATTENTION_MIL` | 42 | 0.758093 | 0.296875 |
| | 43 | 0.757610 | 0.297489 |
| | 44 | 0.754780 | 0.285054 |
| | 45 | 0.758052 | 0.308961 |
| | 46 | 0.761732 | 0.298931 |

| Arm | ROC mean ± population SD | Median / min / max | AP mean ± population SD | Median / min / max |
|---|---|---|---|---|
| `MEAN_MIL` | 0.739867 ± 0.002881 | 0.741558 / 0.734909 / 0.742599 | 0.267821 ± 0.003126 | 0.267127 / 0.263256 / 0.272275 |
| `MAX_MIL` | 0.756667 ± 0.007127 | 0.758841 / 0.744707 / 0.766468 | 0.314566 ± 0.015653 | 0.314577 / 0.291647 / 0.339851 |
| `GATED_ATTENTION_MIL` | 0.758053 ± 0.002211 | 0.758052 / 0.754780 / 0.761732 | 0.297462 ± 0.007598 | 0.297489 / 0.285054 / 0.308961 |

No threshold was selected and no threshold-dependent metrics were invented.
Brier scores are recorded as the single preregistered secondary metric.

The frozen `FULL_48` HGB comparator is ROC-AUC/AP
`0.788830821895/0.359794590322`. `MAX_MIL` therefore lost
`0.032163838942/0.045228554273` on the five-seed metric means. Zero of five max
seeds improved both baseline metrics, and zero met both comparable margins.
For the five-seed mean-probability ensemble, paired bootstrap delta intervals
were `[-0.035271, 0.002824]` for ROC-AUC and `[-0.054557, 0.013546]` for AP;
these uncertainty calculations do not rescue the failed predeclared mean-metric
margins.

The historical propagated-window customer-balanced HGB/Q90 comparator is
`0.741774/0.275693`. Relative to that protocol-different historical result,
`MAX_MIL` was higher by approximately `+0.014893` ROC-AUC and `+0.038873` AP,
while attention was higher by `+0.016279/+0.021769`. This is contextual evidence,
not an exact leaderboard claim.

## Attention concentration and class comparison

Across five seeds, attention was neither uniform nor extremely sparse.

| Customer label | Normalized entropy mean | Effective windows mean | Top-1 mass mean | Top-10%-windows mass mean |
|---|---:|---:|---:|---:|
| Normal (`FLAG=0`) | 0.8426 | 16.3650 | 0.2257 | 0.4038 |
| Theft-labelled (`FLAG=1`) | 0.7922 | 13.1145 | 0.2722 | 0.4633 |

Medians for normalized entropy were `0.8549` and `0.8068`; top-10% mass
medians were `0.4036` and `0.4617`, respectively. Thus theft-labelled customers
showed more concentrated model relevance descriptively. The 10% count used
`ceil(0.10 × bag size)`, so a 32-window bag removes four windows.

## Cross-seed relevance stability

Across 59,310 customer/seed-pair comparisons, mean top-set Jaccard was `0.4458`
versus matched-random expectation `0.0794`; median Jaccard was `0.3333`. Mean
attention-rank correlation was `0.5919` with median `0.6666`. Results were
stronger among theft-labelled customers: mean Jaccard `0.5044` and mean rank
correlation `0.7024`, versus `0.4403` and `0.5816` for normal customers.
Customer prediction rankings were also stable: the ten pairwise seed Spearman
correlations averaged `0.9115` overall, `0.9039` for normal, and `0.9329` for
theft-labelled customers.

This establishes reproducible model relevance, not reproducible theft timing.

## Perturbation test

All 5,930 multi-instance DEV_VAL bags were tested; the one single-window bag was
excluded. Results average five attention seeds per customer and 100 matched
random removals per seed. Positive change means the customer prediction fell
after removing instances.

| Group | TOP probability drop | RANDOM drop | BOTTOM drop | TOP−RANDOM (95% CI) | TOP−BOTTOM (95% CI) |
|---|---:|---:|---:|---:|---:|
| Overall | 0.023092 | 0.000128 | −0.001234 | 0.022965 [0.021740, 0.024186] | 0.024326 [0.023043, 0.025622] |
| Normal | 0.019485 | −0.000039 | −0.001161 | 0.019525 [0.018376, 0.020657] | 0.020646 [0.019440, 0.021840] |
| Theft-labelled | 0.061840 | 0.001921 | −0.002016 | 0.059920 [0.053489, 0.066543] | 0.063857 [0.057033, 0.070797] |

The preregistered perturbation result is positive: the model depends much more
on high-attention instances than on matched random or low-attention instances.
It still cannot establish that those instances are true theft periods.

## Bounded trivial-explanation controls

**Bag size.** Attention-model prediction had little monotonic association with
bag size (five-seed mean Spearman `0.0121` overall); mean pooling was similarly
small (`-0.0076`). Max pooling was more bag-size-dependent (`-0.1632`).
Attention top-10% mass had a modest positive size correlation (`0.243–0.258`),
while normalized entropy correlations were near zero. Average attention
ROC-AUC/AP by fixed stratum were `0.7396/0.3208` for 1–10 windows,
`0.7389/0.3363` for 11–27, and `0.7720/0.2806` for 28–34; performance varied but
did not show a single monotonic bag-size explanation.

**Consumption and variability.** No existing `SUMMARY_17` statistic approached
the predeclared reducibility threshold `|rho| >= 0.90`. The strongest individual
seed/class pooled association was attention versus `standard_deviation` among
theft-labelled customers (`rho=0.4227`, seed 43); its five-seed mean was
`0.3453`. Attention was therefore related to variability but not approximately
reducible to it. Associations with maximum and other level summaries were
smaller.

**Max mechanism.** `MAX_MIL` had mean ROC-AUC `0.001387` below attention but AP
`0.017104` above it. Sparse selection without learned attention is therefore a
credible explanation for part of the MIL signal, particularly AP; attention
did not produce a unique discrimination advantage.

**Short bags.** Only 14 DEV_VAL customers (one theft-labelled) had at most five
windows. Their metrics were extremely unstable and cannot support a subgroup
claim or tuning decision.

**Padding.** Every run had exact zero prediction change after appended masked
padding. Valid attention remained normalized and padded attention was zero.

## Outcome and interpretation

Outcome A failed because no arm improved the baseline by `+0.010` ROC-AUC and
`+0.020` AP. Outcome B failed because the selected `MAX_MIL` arm was outside
both comparable margins (`-0.010/-0.020`), despite the attention arm's stable
and perturbation-supported relevance. The materially-worse flags were also met:
the selected arm lost more than `0.020` ROC-AUC and `0.040` AP on the primary
five-seed means.

Therefore Stage 24B is **C — WEAK RESULT** and **ANCHOR_EARNED = NO**. There is
no anchor sentence. The strongest positive finding is that gated attention
learned reproducible, perturbation-supported predictive relevance without any
instance-level fraud loss. The strongest negative finding is that this did not
translate into competitive customer discrimination. The main ambiguity is
whether the gap reflects the deliberately minimal neural architecture and
window-level `SUMMARY_17` representation or a genuine advantage of Study 1's
explicit customer-level `FULL_48` longitudinal aggregation. Resolving that
would require a new, separately preregistered hypothesis—not adaptive Stage 24B
tuning.

No instance-level `FLAG` loss was used. `CONFIRMATION` was not accessed.
`FINAL_TEST` was not accessed.
