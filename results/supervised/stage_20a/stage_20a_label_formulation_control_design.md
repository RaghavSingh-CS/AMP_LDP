# Stage 20A — Label-Formulation Control Design

## 1. Research question

Can the Stage 17B direct-customer advantage be attributed to better alignment between the customer-level SGCC label and the training objective, rather than to richer customer aggregation, customer multiplicity, or the Stage 14 mean-pooling rule?

This is a design-only stage. No control was fitted or evaluated, and no new DEV_VAL performance was calculated.

## 2. Causal limitation imposed by SGCC labels

SGCC supplies a customer-level theft flag but no verified theft-event timing. A flagged customer's individual windows therefore cannot be classified as truly positive or negative. Repeating the customer flag on every window is a surrogate supervision convention, not temporal ground truth.

Consequently, the proposed controls can test whether a customer-level loss is better aligned with the available customer-level label. They cannot causally prove that particular windows are mislabeled, estimate temporal label-noise rates, or identify theft intervals.

## 3. Mechanisms to separate

The observed Stage 17B versus Stage 14 difference combines several mechanisms:

1. **Customer multiplicity:** a window-trained model lets customers with more usable windows contribute more loss terms.
2. **Prediction pooling:** Stage 14 reduces window scores to a customer score using their mean.
3. **Loss/supervision unit:** Stage 14 applies loss to proxy-labeled windows; Stage 17B applies loss once per labeled customer.
4. **Representation richness:** Stage 17B uses 48 customer behavioural features, whereas Stage 14 uses 17 features per window.

The controls below isolate the first three as far as the observed labels permit. None is presented as a fully matched causal decomposition of the nonlinear 48-feature Stage 17B detector.

The design is motivated by committed development evidence: the Stage 16 training representation contained 29,655 customers and 768,923 windows (mean 25.93, median 32, maximum 34 windows per customer), with large masses at 10 windows (21.50%) and 34 windows (45.34%). Stage 18 did not find that useful discrimination was confined to long-history customers. Stage 19 found level/distribution information conditionally important while other behavioural families retained useful signal. Together, these results make multiplicity and pooling necessary controls, but also show why another rich customer aggregation would not isolate supervision.

## 4. Candidate controls

### C1 — Customer-balanced window HGB (recommended)

- **Exact hypothesis:** Some of the direct-customer advantage is caused by unequal customer multiplicity in the window-level loss.
- **Training unit:** Usable temporal window from DEV_TRAIN.
- **Feature representation:** Exact frozen 17 Stage 14 behavioural window summaries, in frozen order; no new feature.
- **Target definition:** The customer's binary `FLAG` propagated to each of that customer's windows. This is explicitly a customer-label proxy, not a window-level truth label.
- **Fit/evaluation protocol:** Fit two otherwise identical HGB controls on DEV_TRAIN: an unweighted baseline with weight 1 per window, and a customer-balanced arm with

  `w_ij = N_windows / (N_customers * n_i)`,

  where `n_i` is customer `i`'s usable-window count in DEV_TRAIN. This gives every customer equal total weight while retaining mean window weight 1. Use the frozen HGB configuration: learning rate 0.1, 100 iterations, 15 maximum leaf nodes, minimum 100 samples per leaf, L2 regularization 1.0, early stopping disabled, and random seed 42. Produce window probabilities for DEV_VAL and use the pre-registered C2 aggregation panel.
- **Variables held constant:** DEV_TRAIN/DEV_VAL roles, rows, labels, 17-feature schema, estimator family and configuration, and customer scoring rules.
- **Variable intentionally changed:** Training loss weight contributed by customers with different window counts.
- **Inference supported:** A paired improvement over the unweighted baseline supports customer multiplicity as one mechanism. Little change argues against multiplicity being a major explanation.
- **Inference not supported:** Temporal label noise, correctness of any individual window label, or superiority of the 48-feature representation.
- **Leakage/label assumptions:** Window counts and weights are derived separately within DEV_TRAIN; customer-disjoint development roles remain fixed. No validation-derived weights or tuning are allowed.
- **Implementation complexity:** Low. A residual limitation is that HGB's minimum-leaf sample count is row-count based, so weighting cannot make the tree-building problem identical to one-row-per-customer training.

### C2 — Fixed window-prediction aggregation panel (recommended diagnostic)

- **Exact hypothesis:** Some of the direct-customer advantage is caused by the Stage 14 mean-probability pooling rule rather than the training supervision unit.
- **Training unit:** Window, using both C1 HGB arms.
- **Feature representation:** Exact frozen 17 Stage 14 behavioural window summaries.
- **Target definition:** Customer `FLAG` propagated to windows, with the same proxy-label qualification as C1.
- **Fit/evaluation protocol:** Do not fit an additional model. For each C1 arm, aggregate the identical DEV_VAL window probabilities to customers using arithmetic mean as the primary rule and median, 90th percentile, and maximum as fixed sensitivity rules. Define the 90th percentile using NumPy's linear quantile interpolation. Report every rule; do not select or promote the best-performing rule.
- **Variables held constant:** Fitted window model, window probabilities, customers, labels, and evaluation metrics.
- **Variable intentionally changed:** Deterministic window-to-customer prediction pooling rule.
- **Inference supported:** Systematic differences show sensitivity to inference-time pooling and quantify whether mean pooling masks localized high scores.
- **Inference not supported:** Label noise, loss-unit effects, or a deployable replacement selected on DEV_VAL.
- **Leakage/label assumptions:** Freezing the complete panel before scoring is essential. Choosing a winning rule from DEV_VAL would introduce model-selection adaptivity and is prohibited.
- **Implementation complexity:** Low.

### C3 — Matched linear loss-before-versus-after-pooling control (recommended mechanism control)

- **Exact hypothesis:** With the same local feature content and linear scoring family, applying the classification loss once per customer after feature pooling yields better customer discrimination than applying loss to customer-balanced proxy-labeled windows.
- **Training unit:** Two arms: windows for **W-LOGIT** and customers for **C-LOGIT**.
- **Feature representation:** Both arms start with the exact frozen 17 Stage 14 window summaries. Fit one customer-balanced mean/variance transformation on DEV_TRAIN windows using the C1 weights and apply that same transformation in both arms. C-LOGIT uses the arithmetic mean of the transformed 17-dimensional window vectors for each customer. This is a pre-specified pooling operation, not feature discovery.
- **Target definition:** W-LOGIT uses customer `FLAG` propagated to windows; C-LOGIT uses one customer `FLAG` per pooled customer row.
- **Fit/evaluation protocol:** Use fixed L2 logistic regression (`C=1`, `lbfgs`, `max_iter=300`, deterministic seed where applicable) without tuning. Fit W-LOGIT on transformed DEV_TRAIN windows with C1 customer-balanced weights. Fit C-LOGIT on one mean-transformed vector per DEV_TRAIN customer with unit customer weights. For DEV_VAL, score W-LOGIT as `sigmoid(mean(window logits))`; score C-LOGIT directly. Because a linear logit commutes with arithmetic feature averaging, the two arms use the same customer scoring form while placing the training loss before versus after pooling.
- **Variables held constant:** Development roles, original 17 local quantities, shared train-only transformation, linear model family and hyperparameters, equal total customer contribution, and customer scoring equation.
- **Variable intentionally changed:** Whether binary cross-entropy is applied to individual proxy-labeled windows or once to the pooled customer representation.
- **Inference supported:** A paired C-LOGIT advantage supports better alignment of customer-level loss with the available customer-level label under a deliberately matched linear setting.
- **Inference not supported:** Temporal label-noise causation, the size of the mechanism inside nonlinear HGB, or equivalence to the richer frozen 48-feature Stage 17B representation.
- **Leakage/label assumptions:** The shared transformation is fitted from DEV_TRAIN only. The propagated window target remains a surrogate. Coefficients are separately estimated because the loss formulations are the object of comparison.
- **Implementation complexity:** Medium; it requires explicit numerical checks that transformed-feature means and mean-logit customer scores are equivalent to tolerance.

### C4 — Fixed-K window sampling (considered; rejected as a primary control)

- **Exact hypothesis:** Equalizing the number of loss terms per customer reduces the window-model disadvantage.
- **Training unit:** A deterministic fixed number of DEV_TRAIN windows per customer.
- **Feature representation:** Exact frozen 17 Stage 14 window summaries.
- **Target definition:** Customer `FLAG` propagated to selected windows.
- **Fit/evaluation protocol:** Would fit the same frozen HGB after deterministic sampling and aggregate DEV_VAL predictions with fixed rules.
- **Variables held constant:** Feature schema, estimator, and nominal label convention.
- **Variable intentionally changed:** Customer multiplicity, but also the amount and temporal coverage of information retained.
- **Inference supported:** At most a coarse multiplicity sensitivity check.
- **Inference not supported:** A clean loss-weighting or label-noise effect.
- **Leakage/label assumptions:** Sampling position and seed can change conclusions despite the absence of temporal truth.
- **Implementation complexity:** Low, but inferential quality is inferior to exact customer balancing. It is excluded from the minimum experiment set.

## 5. Rejected or non-isolating controls

- A new HGB built from richer customer aggregates would repeat the Stage 16 confounding of representation and supervision unit.
- Multiple-instance learning, noisy-OR, attention pooling, or latent-positive-window objectives assume an unobserved instance-label mechanism—often that at least one window is positive—and add architecture choices unsupported by SGCC timing labels.
- Oracle or pseudo-positive window labels would fabricate temporal ground truth.
- Replicating each frozen 48-feature customer vector across its windows changes nominal row granularity without creating a meaningful window-supervision contrast.
- The historical Stage 14 fitted model is not a valid current DEV_VAL control because the later Stage 17A hierarchy carved DEV_VAL from the original training population. All Stage 20B controls must be refitted on DEV_TRAIN only.

## 6. Minimum defensible Stage 20B experiment set

1. Fit the unweighted and customer-balanced window HGB arms in C1 on DEV_TRAIN.
2. Evaluate both arms on identical DEV_VAL customers with the complete frozen C2 aggregation panel; arithmetic mean remains primary.
3. Fit and compare the W-LOGIT and C-LOGIT arms in C3.
4. Use ROC-AUC and Average Precision as primary customer-level metrics, with prevalence reported beside AP. Avoid threshold-dependent causal claims because the controls have different score calibration and no threshold may be tuned on DEV_VAL.
5. Use a paired customer bootstrap with 2,000 replicates and seed 42 for the pre-specified deltas: customer-balanced minus unweighted HGB under mean pooling, each fixed alternative pooling rule minus mean within an HGB arm, and C-LOGIT minus W-LOGIT.
6. Cite the frozen Stage 17B DEV_VAL result only as contextual evidence. Do not present it as fully representation-matched to these controls.

No historical Stage 14 score is to be treated as a matched baseline, and no CONFIRMATION or FINAL_TEST result is part of this set.

## 7. Stage 20B pre-registration and stopping rules

Stage 20B should separate specification freezing from evaluation:

- **Prepare:** verify Stage 17A governance and frozen hashes; freeze customer identities and roles, the exact 17-feature order, weights, model configurations, transformation, aggregation functions, metrics, bootstrap comparisons, and interpretation rules in a machine-readable manifest. This step must not calculate DEV_VAL performance.
- **Evaluate once:** assert the manifest hash, fit on DEV_TRAIN, score DEV_VAL once for the frozen comparisons, and write all specified outputs including null or adverse results.
- **No adaptation:** do not tune weights, model settings, logistic regularization, aggregators, thresholds, or comparison families after DEV_VAL scoring. Unexpected implementation failure permits correction only when documented and when no scientific choice is changed.
- **Protected roles:** CONFIRMATION and FINAL_TEST remain closed throughout Stage 20B.

## 8. Interpretation map for possible outcomes

| Observed Stage 20B pattern | Defensible interpretation |
|---|---|
| Customer balancing materially improves mean-pooled window HGB | Unequal customer multiplicity explains part of the prior gap. |
| Customer balancing has little effect | Multiplicity is unlikely to be the dominant mechanism under this control. |
| Fixed alternative pooling rules materially improve both window arms | Stage 14 mean pooling contributed to the formulation gap; the result is diagnostic, not permission to select a new rule. |
| C-LOGIT materially exceeds W-LOGIT | Customer-level loss alignment helps under matched local information and a linear scoring family. |
| W-LOGIT and C-LOGIT are similar | The evidence does not isolate loss placement as a major mechanism in the matched linear setting. |
| C1/C2/C3 explain little while Stage 17B remains much stronger contextually | Richer nonlinear customer representation is a plausible principal contributor; this remains an inference, not a complete decomposition. |
| Several controls improve | Multiple mechanisms jointly contributed; effects should not be summed as independent causal components. |
| Results are imprecise or inconsistent | The mechanism remains unresolved; no label-noise claim is justified. |

## 9. Claims permitted after Stage 20B

The strongest permitted formulation claim, if supported, is: **customer-level supervision is better aligned with the customer-level SGCC label under matched observed feature information.**

The following claim remains prohibited: **window labels are temporally noisy and caused the Stage 14 deficit.** The dataset lacks the timing ground truth needed to establish it.

All conclusions must remain limited to SGCC and to the pre-specified development hierarchy.

## 10. Governance and stage decision

Stage 20A selects C1, C2, and C3 as the minimum defensible control family and rejects C4 as redundant and less isolating. This is a methodological design decision only. It adds no performance evidence and does not change the frozen Stage 17B detector, feature specification, threshold, or governance manifest.

**CONFIRMATION NOT ACCESSED DURING STAGE 20A.**

**FINAL_TEST NOT ACCESSED DURING STAGE 20A.**
