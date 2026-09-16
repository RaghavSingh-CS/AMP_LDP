# Stage 24A — Weakly Supervised MIL Feasibility and Experiment Design

## Decision

**A — GO: a clean first MIL experiment is feasible and scientifically
justified.** The available supervision supports customer-bag learning, not
instance-level theft detection. Frozen usable windows map cleanly to customers,
bags are small enough to retain in full, and a tightly controlled comparison
can distinguish useful bag learning from several important trivial mechanisms.
The central limitation is irreducible: without event timing, learned instance
relevance cannot be validated as theft localization.

No MIL model was trained or scored in Stage 24A. No row-level `CONFIRMATION` or
`FINAL_TEST` data was accessed.

## 1. Repository and governance audit

The audit began with a clean working tree on branch `study2/mil` at
`d0601c54f6e9d0f11caac2ebd27f7c86a568a863`, exactly the frozen Study 1 final
commit. Study 1 remains **FROZEN AND COMPLETE**: `FINAL_FULL_48` plus the
unweighted `HistGradientBoostingClassifier` has final ROC-AUC
`0.7925125381284772` and Average Precision `0.37015859978842647`. Those final
metrics are comparison evidence, not a Study 2 tuning set.

The Stage 17A governance manifest is unchanged at SHA-256
`dae7dba53260a3af6b8e2ddd904b5c7867c3321f27cdff5af5e4060b97c348c0`.
Membership remains mutually exclusive: 23,725 governed `DEV_TRAIN` customers
(23,724 with usable windows), 5,931 `DEV_VAL`, 6,355 `CONFIRMATION`, and 6,356
`FINAL_TEST`. Stage 24A used only the original-train processed matrix and its
usable-window manifest, then restricted them by the label-free governance
manifest to `DEV_TRAIN`/`DEV_VAL`. Protected preprocessing/window artifacts and
all Study 1 result artifacts were read-only.

The boundary is:

- **Study 1:** frozen and complete; never altered by Study 2.
- **Study 2:** weakly supervised MIL exploration; routine fit on `DEV_TRAIN`,
  evaluation on `DEV_VAL`, with `CONFIRMATION` and `FINAL_TEST` closed.

## 2. Exact supervision-granularity finding

Repository documentation and manifests support the following facts.

1. `FLAG=0` denotes a normal-labelled customer and `FLAG=1` a theft-labelled
   customer.
2. `FLAG` is one customer-level target. Window manifests copy that value from
   the customer; this is not an independently collected window outcome.
3. Dates in the data are consumption-measurement dates. `start_date` and
   `end_date` in a window manifest describe the window's measurement coverage,
   not a theft event.
4. No authoritative theft-event start/end fields or independently observed
   window theft labels exist in the inspected artifact schema or metadata.

Therefore A, “the customer is labelled fraudulent,” is observable. B, “this
temporal window is verified fraudulent,” is not observable. **The dataset does
not provide verified instance-level theft ground truth.** This finding does not
establish that propagated labels are wrong or noisy, and it never licenses a
claim that an individual window is fraudulent.

## 3. Bag feasibility and integrity

The development population contains 29,655 eligible customers and 768,923
existing usable stride-30 windows.

| Population | Customers | Windows | Min | Q10 | Q25 | Median | Q75 | Q90 | Max | Mean | Population SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All development | 29,655 | 768,923 | 1 | 10 | 14 | 32 | 34 | 34 | 34 | 25.929 | 9.957 |
| `DEV_TRAIN` | 23,724 | 614,976 | 1 | 10 | 14 | 32 | 34 | 34 | 34 | 25.922 | 9.954 |
| `DEV_VAL` | 5,931 | 153,947 | 1 | 10 | 13 | 32 | 34 | 34 | 34 | 25.956 | 9.969 |
| Normal (`FLAG=0`) | 27,128 | 708,308 | 1 | 10 | 15 | 32 | 34 | 34 | 34 | 26.110 | 9.925 |
| Theft-labelled (`FLAG=1`) | 2,527 | 60,615 | 1 | 10 | 10 | 28 | 34 | 34 | 34 | 23.987 | 10.093 |

“Very short” is predeclared as at most five usable windows: 85 customers
(`0.2866%`) meet that definition. Nine customers (`0.0303%`) have one window,
46 (`0.1551%`) have at most three, and 6,655 (`22.4414%`) have at most ten.
Among normal customers, 66/27,128 (`0.2433%`) have at most five; among
theft-labelled customers, 19/2,527 (`0.7519%`) do. The complete role-by-class
breakdown is in `bag_statistics.json`.

All 768,923 window IDs are unique. Every window maps to exactly one customer;
there are zero duplicate `(customer, window number)` and zero duplicate
`(customer, start, end)` mappings. Start/end dates parse, window indices increase
within customer, and chronological order is recoverable. A bag is constructed
solely by grouping that customer's own windows; no feature or instance from
another customer is required. With a maximum of 34 instances, variable-length
bags are straightforward and need no truncation or resampling.

## 4. Existing instance representations

| Representation | Dimensions | Development float32 matrix | Median/max bag payload | Nonfinite/missing | Chronology and interpretation |
|---|---:|---:|---:|---:|---|
| `RAW_30` | 30 | 88.00 MiB | 3.75/3.98 KiB | 0/0 | Preserves 30 ordered scaled daily values; direct trajectory interpretation |
| `SUMMARY_17` | 17 | 49.86 MiB | 2.13/2.26 KiB | 0/0 | Named level/distribution/variability/directional summaries; does not preserve the complete within-window trajectory |
| Existing `RAW_PLUS_SUMMARY_47` | 47 | 137.86 MiB | 5.88/6.24 KiB | 0/0 | Retains raw order plus summaries, but is deliberately redundant |

All three are existing frozen Stage 14 representations; the combined option
introduces no new feature engineering but is excluded from the first experiment
to avoid a representation search. Bag order remains recoverable from metadata
for every representation. `RAW_30` and `SUMMARY_17` exclude label, customer ID,
window ID, split, dates, and missingness metadata from the model input.

There is one inherited protocol limitation: the frozen scaler was historically
fit on the full original-training population before `DEV_TRAIN`/`DEV_VAL` were
created. Thus current `DEV_VAL` feature values, but not its labels, influenced
the fixed scale. This is not target leakage and is not changed in Study 2, but
it means `DEV_VAL` is not pristine with respect to unsupervised preprocessing.

**Primary recommendation: `SUMMARY_17`.** It is finite, compact, named and
interpretable, already frozen/reproducible, and a natural match to a very small
instance encoder. The choice is methodological and was made without any new
`DEV_VAL` performance. `RAW_30` remains a plausible later preregistered
sensitivity study, not part of Stage 24B.

## 5. MIL prediction problem

For customer (i), let (B_i=\{x_{i1},\ldots,x_{im_i}\}) and let
(Y_i\in\{0,1\}) be the customer `FLAG`. Each instance is encoded as
(h_{ij}=f_\theta(x_{ij})). A permutation-invariant operator produces
(z_i=g(h_{i1},\ldots,h_{im_i})), and the customer probability is
(p_i=\mathrm{sigmoid}(q(z_i))). Training minimizes binary cross-entropy over
customer bags only. There is no instance-level fraud loss and no propagated
`FLAG` supervision on individual windows.

Internal instance scores or weights are latent model quantities. They are not
ground-truth theft probabilities unless independently validated against new
event-level truth.

## 6. Exact minimal architecture and training procedure

The shared encoder is `Linear(17,32) → ReLU → Dropout(0.1) → Linear(32,16)
→ ReLU` (1,104 parameters). The shared classifier is `Linear(16,1)` (17
parameters). No normalization is used. Mean and max models contain 1,121
trainable parameters. The single gated-attention pool uses 16-dimensional
`V`, `U`, and bias-free `w` projections and adds 560 parameters, for 1,681
total.

Only pooling differs:

- mask-aware arithmetic **MEAN**;
- mask-aware elementwise **MAX**;
- one **GATED_ATTENTION** operator
  `softmax(wᵀ[tanh(Vh) ⊙ sigmoid(Uh)])`, followed by a weighted sum.

Weights use Xavier-uniform initialization and biases zero. Training is exactly
50 epochs, no early stopping, unweighted `BCEWithLogitsLoss`, AdamW with
learning rate `0.001` and weight decay `0.0001`, global gradient clipping at
`5.0`, customer batch size 64, and seeds `[17, 42, 73, 101, 202]`. Every
eligible `DEV_TRAIN` customer appears once per epoch. Fixed epochs prevent
`DEV_VAL` from becoming an early-stopping set.

Custom collation pads only to the largest bag in each batch and supplies a
boolean mask. Padding is excluded from all pools. No bag is truncated, no
window is duplicated, and there is no label-dependent sampling. The maximum
raw padded `SUMMARY_17` input is only `0.1411 MiB` per 64-customer batch; total
training memory should remain well below 1 GiB. On the audited 16-logical-CPU,
13.85-GiB environment, a conservative estimate is 2–10 minutes per 50-epoch
fit and 0.5–2.5 hours for 15 fits. PyTorch `2.13.0+cpu` is present; CUDA is not.
Specialized hardware is unnecessary.

Chronology is deliberately ignored in the first experiment. That isolates
weakly supervised bag learning and avoids a sequence-model confound. The model
may identify decision-relevant instances independent of ordering; it does not
model temporal dynamics. Transformer, LSTM, GRU, CNN, GAN, graph models,
attention variants, and architecture search are outside scope.

## 7. Minimum baseline matrix

| Arm | Action in Stage 24B | DEV_VAL ROC-AUC/AP if historical |
|---|---|---|
| Frozen Study 1 `FULL_48` HGB customer model | Reuse committed metrics/probabilities; do not retrain | `0.788831 / 0.359795` |
| Propagated-window supervised, customer-balanced HGB with Q90 customer aggregation | Reuse Stage 20B metrics; do not retrain | `0.741774 / 0.275693` |
| MIL mean | Train fresh | — |
| MIL max | Train fresh | — |
| MIL gated attention | Train fresh | — |

The final Study 1 evaluation (`0.792513 / 0.370159`) remains external comparison
evidence only; it is not a Stage 24B selection target. No broad baseline suite
is warranted.

## 8. Attention meaning and relevance analyses

Allowed interpretation: **relative contribution or relevance of an instance to
the model's customer-level representation or decision.** Forbidden without
external event validation: probability that theft occurred in the window,
confirmed theft period, theft start/end date, ground-truth anomaly probability,
or localization accuracy.

The preregistered model-behaviour analyses are entropy, normalized entropy,
effective number of attended windows `exp(entropy)`, top-1 mass, top-10% mass,
class-wise distributions, pairwise-seed top-set Jaccard, within-bag rank
correlation, and associations with the existing 17 summaries. Stability must
be compared with matched random expectations.

For perturbation, remove the top 10% of real instances, an equal number of
bottom-ranked instances, or an equal number selected randomly (100 matched
random repeats). Recompute the whole bag prediction after removal and retain at
least one instance. Single-instance bags are excluded and counted. Report
customer logit/probability changes and paired intervals. A reproducibly larger
top-removal effect can support “the model concentrates predictive evidence in
a subset of instances.” It cannot support “those instances are verified theft
periods.”

## 9. Anchor-result criteria

**Outcome A — strong discrimination** requires all of: mean ROC-AUC gain at
least `+0.010`, mean AP gain at least `+0.020`, positive gains for both metrics
in at least four of five seeds, and paired 95% customer-bootstrap intervals
above zero for both seed-ensemble deltas versus frozen `FULL_48` HGB.

**Outcome B — comparable discrimination plus distinct weak supervision**
requires mean ROC-AUC no more than `0.010` below baseline, mean AP no more than
`0.020` below, and at least four of five seeds within both margins. It also
requires reproducible concentration, top-removal exceeding random removal with
a paired interval above zero, and rank/top-set stability above matched random
expectation.

**Outcome C — weak** applies if no pool satisfies A or B, or attention is no
more stable/informative than random controls. A mean ROC-AUC loss greater than
`0.020` or AP loss greater than `0.040` is explicitly materially worse. These
margins prevent tiny rank fluctuations from becoming a success claim.

Potential one-sentence anchors are conditional, not claims: exact performance
improvement requires Outcome A and uncertainty; a bag-level supervision result
requires the fixed full protocol and seed distribution; concentration requires
predeclared mass/entropy plus stability; perturbation requires matched
top/random/bottom removal; stability requires Jaccard/rank agreement above
random. Study 2 should ultimately select one memorable statement supported by
the evidence, never manufacture one.

## 10. Controls against trivial explanations

The limited control set reports score/attention associations with mean,
maximum, standard deviation, range, and bag size; compares performance and
concentration in fixed bag-size strata `1–10`, `11–27`, and `28–34`; checks
prediction–bag-size association within each class; profiles high-attention
windows using only existing summaries; and compares mean, max, and attention
for seed stability and bag-size dependence. This exposes high-consumption,
high-variance, simple-statistic, max-as-anomaly-detector, and history-length
explanations without expanding the model grid.

Missingness fraction, masks, bag size, IDs, dates, and window positions never
enter the predictor. Very short bags are reported separately. Padding
invariance must be unit-tested by appending masked rows and confirming identical
logits and valid-instance attention. Top/random/bottom removal operates by
removing real instances, not by injecting zero pseudo-windows. These checks
address missingness/history and mask artefacts while preserving the minimal
experiment.

## 11. Exact Stage 24B proposal and future gates

Stage 24B freezes `SUMMARY_17`, the one 17→32→16 encoder, one linear customer
classifier, exactly three pools, five seeds, 50 fixed epochs, and the padded-mask
collate path. It fits 15 models on `DEV_TRAIN`, evaluates each once on `DEV_VAL`,
reports customer ROC-AUC/AP per seed and in aggregate, runs only the stated
secondary/relevance controls, and applies the predeclared anchor criteria. It
does not open `CONFIRMATION` or `FINAL_TEST` and does not select a lucky seed.

If a Stage 24B outcome earns a gate, representation, pool, architecture,
training, seed ensemble, analysis code, and claim must be frozen before a
single `CONFIRMATION` access. Failure returns work to development only for a
materially new preregistered hypothesis, never a confirmation-driven tweak.
`FINAL_TEST` stays closed until a complete Study 2 final method is frozen.

## 12. Opportunity, risk, and conclusion

The biggest scientific opportunity is a credible demonstration that
customer-level labels alone can yield competitive discrimination and a stable,
perturbation-supported concentration of predictive evidence across temporal
instances—without pretending to possess temporal fraud labels.

The biggest scientific risk is semantic overreach: attention may merely track
consumption level, variance, missing-history structure, or bag size, and no
internal analysis can turn it into verified theft timing. The inherited
full-original-train scaler also limits how pristine `DEV_VAL` is as an
unsupervised-preprocessing holdout.

Despite those limitations, the supervision, mappings, scale, compute, and
control design are adequate. The Stage 24A decision is **GO**.
