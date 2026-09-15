# Stage 23A — Research Freeze and Final Evaluation Protocol

## Scope and pre-stage audit

Stage 23A is governance/design only. The audit began at clean commit
`6af8a2e633be571891954a509f1428d2037fea63`. No model was trained, fitted,
scored, or evaluated. No `CONFIRMATION` row-level data and no `FINAL_TEST`
features, labels, predictions, or metrics were accessed.

The committed governance and method chain passed hash/provenance inspection:

| Item | SHA-256 / result |
|---|---|
| Stage 17A governance manifest | `dae7dba53260a3af6b8e2ddd904b5c7867c3321f27cdff5af5e4060b97c348c0` |
| Historical split manifest | `6ad1132bbaef0c6f6ccc5bdc973844b805c4291612ac7170ec165f273835959c` |
| Stage 16 48-feature specification | `a722299d6207fce06180f71bf96717ce58948e127a85440e04f721c1616fcdf3` |
| Stage 17B configuration | `605eb1f482123a24a14c0929c951f400c37b3fbc2d2bb2e4d46d2ce834698046` |
| Stage 17B fitted HGB | `67ed869858185d8f71d99b3dce8b358c43793178185e5aa50e6997d11b56974b` |
| Stage 17B validation decision | `11e15f18e2521ffed8e7dbaacee469284ee356acedea1d8f34ffcf48785a68fa` |
| Stage 17C confirmation decision | `1570b83839d28d00a406e90e5b099957bd2e09da6a7ec010b1ac6f645e54b869` |
| Stage 17C confirmation metrics | `800534d0283089ff673958c9a2c1f4c38a1abbe86e7c501d998394ac317f08a1` |
| Stage 18 robustness decision | `092ffcf04f6898ec4a134718447b2985feea3476e412da81ebef8018ba00c78b` |
| Stage 19 family-ablation decision | `ef6d9830c5b7bb191fdfc6be97f1ba47381d579677dc24e322eef8751ae92ee6` |
| Stage 20B formulation-control decision | `7aa026c6b22c9e1e1a5a0fc9fd53177d0b433be5992dc725aa2014fa9df6addb` |
| Stage 21 benchmark decision | `e78d43a9462b69bc416ac1c09d1c8956a8a40a1177470a27fa0396f9101ca020` |
| Stage 22 compact-representation decision | `549386dd76fe93e73459f2e7a03bda853ba07d85d84a40859cb52d2ebeb86086` |
| Protected preprocessing scaler | `34bbc0491465fb5ba627df70ee95706e71bc92fa21c6fa8a32048048a855d0bc` |

The Stage 17A split roles, Stage 16 schema, Stage 17B model/configuration, and
protected preprocessing/split artifacts remain consistent with their committed
records. The later decisions preserve the established progression: strong
confirmation, robustness with a consumption limitation, redundant-but-useful
feature families, inconclusive causal formulation controls, boosting-family
robustness, and rejection of compact representation.

## Final evidence chain

The detailed frozen chain is in `evidence_chain.csv` and is divided as follows.

- **Historical exploratory evidence:** reconstruction FC/CNN/LSTM was weak;
  Stage 13 established usable supervised signal; Stage 14 supported behavioural
  summaries; Stage 16 supported direct customer aggregation.
- **Current development evidence:** Stage 17A reset the hierarchy; Stage 17B
  replicated the direct-customer result; Stages 19 and 21 supported useful,
  redundant representation signal across strong learners.
- **Confirmation evidence:** Stage 17C opened `CONFIRMATION` once for the exact
  frozen Stage 17B pipeline and obtained ROC-AUC `0.7890840481602820` and AP
  `0.3790578558669986`, then closed it.
- **Negative ablations:** Stage 15 rejected FC hybrid value; Stage 20B did not
  isolate multiplicity, aggregation, or loss placement as a complete causal
  explanation; Stage 22 rejected correlation-based compaction.
- **Robustness and limitations:** Stage 18 retained useful ranking across
  history strata but found consumption-dependent variation and low-consumption
  recall `0.172` at the global threshold. Evidence remains internal to SGCC,
  whose customer flag has no theft-event timing.

## Frozen final representation and classifier

`FINAL_FULL_48` is frozen. It is the exact ordered 48-feature Stage 16 direct-
customer behavioural representation. Stage 22 reduced it to 26 features but
materially reduced ROC-AUC/AP: mean paired deltas were `-0.012108` (95% CI
`[-0.019878, -0.004148]`) and `-0.021840`
(`[-0.035788, -0.007972]`). No further feature selection is permitted.

The final classifier is the unweighted
`sklearn.ensemble.HistGradientBoostingClassifier` with `learning_rate=0.1`,
`max_iter=100`, `max_leaf_nodes=15`, `min_samples_leaf=100`,
`l2_regularization=1.0`, `early_stopping=False`, and `random_state=42`.
Stage 21 found HGB, LightGBM, and their fixed mean statistically/practically
close, with no material ensemble gain; HGB is retained for simplicity and its
existing confirmation evidence.

## Final training population

The final refit uses **DEV_TRAIN + DEV_VAL only**, comprising 29,655 customers
with an established usable representation. The one original-train customer
without any usable stride-30 window is excluded and must be reported.
`CONFIRMATION` and `FINAL_TEST` are excluded.

This policy uses all routine development data while preserving the Stage 17C
confirmation result as independently reportable evidence. Consuming
`CONFIRMATION` could marginally enlarge training, but would erase its clean
confirmatory role without a compelling scientific benefit.

## Train-derived reference policy and dependency chain

Stage 23B must first verify the sealed governance, split, preprocessing, schema,
and method hashes. It then constructs all usable stride-30 windows from
`DEV_TRAIN + DEV_VAL` under the established preprocessing. Across those windows
only, it recomputes the linear-interpolation 90th percentile for `maximum`,
`standard_deviation`, `mean_absolute_consecutive_change`, and
`maximum_absolute_consecutive_change`. Each persistence feature remains the
fraction of a customer's windows strictly exceeding its corresponding frozen
final-training threshold.

All other Stage 16 definitions, names, order, chronology, early/late contrasts,
and customer/window eligibility rules remain unchanged. The final model is fit
once and the model/references are written and hashed before `FINAL_TEST` is
opened. Those same references are then applied unchanged to `FINAL_TEST`.
Neither labels nor `FINAL_TEST` contribute to reference computation.

## Threshold, metrics, and uncertainty

The secondary operating threshold is fixed at `0.20070617709942135`, the Stage
17B threshold obtained using the established positive-class maximum-F1 rule on
DEV_VAL predictions from the DEV_TRAIN-fitted model. It is not re-derived after
the combined-development refit and is never selected from `FINAL_TEST`.

Because refitting can change probability calibration, reuse of the numeric
threshold is a transparent transfer of a fully development-derived operating
point, not a guarantee of calibrated equivalence. ROC-AUC and Average Precision
are therefore primary. Precision, recall, F1, and the TN/FP/FN/TP confusion
matrix at the fixed threshold are secondary. Customer count, theft count, and
prevalence are descriptive.

ROC-AUC and AP receive customer-level nonparametric percentile 95% confidence
intervals from 2,000 valid bootstrap replicates with seed 42. A single-class
draw is discarded deterministically; sampling continues until 2,000 valid
replicates, and attempts/discards are reported.

## Historical FINAL_TEST access

The current `FINAL_TEST` population is not pristine. Stage 13 accessed it once,
before Stage 17A and before the direct-customer representation existed, to
evaluate the historical balanced window-HGB formulation. Its committed
customer ROC-AUC/AP were `0.7023135402`/`0.2193165970`.

1. Committed history shows no use of `FINAL_TEST` in Stages 14–22 to select or
   tune `FULL_48`, the Stage 17B HGB, or its threshold.
2. The old access does not make evaluation of a later, substantively different,
   pre-frozen pipeline meaningless; it does preclude a never-observed test-set
   claim and weakens the strict independence narrative.
3. The limitation is that the cohort identity and one obsolete baseline result
   were known during the broader research program, creating potential indirect
   research-level familiarity even though the final method was developed
   without its row-level data or metrics.
4. The paper must call it the **final evaluation cohort with disclosed
   historical baseline access**, never a pristine, untouched, or never-seen
   test set.
5. Stage 13 must be reported separately as historical exploratory window-
   baseline evidence. Stage 23B is the one frozen evaluation of the final
   direct-customer pipeline, with the disclosure adjacent to its results.

## Claims freeze

The row-level rationale is in `claims_matrix.csv`.

**Supported:** weak reconstruction under the evaluated formulation; usable
supervised signal; benefit of behavioural summaries; substantial direct-
customer improvement; Stage 17 development replication; strong Stage 17C
confirmation; robustness across history length; a low-consumption operating-
point limitation; and useful non-level behavioural signal.

**Partially supported:** probability aggregation contributes to the observed
formulation difference. Q90 improved AP in both window-HGB arms, but ROC-AUC
evidence was uncertain and the full pre-registered explanatory criterion failed.

**Not supported:** FC reconstruction/latent complementarity; equal performance
across consumption levels; consumption level as a complete explanation;
customer multiplicity or customer-level loss placement as the explanation;
proven temporal window-label noise; dependence on a uniquely favourable strong
boosting classifier; safe correlation-based compaction; need for GAN or SMOTE;
state of the art; or external generalization beyond SGCC.

## Frozen paper contribution

**One-sentence contribution.** On SGCC, AMP shows that interpretable customer-
level longitudinal behavioural profiling, aligned with customer-level ground
truth, provides substantially stronger and confirmed theft discrimination than
the evaluated reconstruction and window formulations, while controlled
experiments show robustness across strong learners and do not isolate a single
causal mechanism for the gain.

**Three main contributions.**

1. A leakage-governed direct-customer representation of 48 interpretable
   longitudinal behavioural quantities, systematically compared with
   reconstruction and window-to-customer formulations.
2. Replicated and independently confirmed discrimination, plus pre-specified
   subgroup robustness analysis that identifies a concrete low-consumption
   operating-point limitation.
3. Controlled learner, feature-family, formulation, and compaction experiments
   that distinguish stable findings from negative or inconclusive mechanisms.

**Central research question.** Does customer-level longitudinal behavioural
profiling, evaluated at the granularity of SGCC customer labels, improve and
robustly support electricity-theft discrimination relative to reconstruction
and window-based formulations?

**Strongest defensible conclusion.** Within SGCC, the frozen direct-customer
`FULL_48` HGB pipeline provides substantially stronger, reproducible, and
independently confirmed ranking discrimination than the evaluated reconstruction
and historical window baselines; the evidence supports the formulation's value
but not a unique causal explanation.

**Primary limitation.** Evidence comes from one SGCC dataset whose labels are
customer-level flags without theft-event timing, so neither temporal window-
label correctness nor external generalization can be established.

**Secondary limitations.** The final cohort had disclosed historical Stage 13
baseline access; DEV_VAL supported multiple controlled research decisions;
performance and fixed-threshold recall vary by consumption level; AP is
prevalence-sensitive; and the final numeric threshold may transfer imperfectly
after combined-development refitting.

**Explicit non-claims.** The paper does not claim that window labels are wrong,
that customer-level modelling fixes label noise, that any single formulation
mechanism causes the gain, that GAN/SMOTE is required, that correlation-
redundant features are safely removable, that AMP is state of the art, or that
results generalize beyond SGCC.

## Sealed final method and Stage 23B protocol

The complete unambiguous method is `final_method.json`, SHA-256
`c68512ad348d01dfc2dddf43e49301ec964d83de44746a428841edb2b1632113`.
It fixes customer prediction/training units, the exact ordered Stage 16 schema,
the HGB configuration, final-fit population, reference recomputation, threshold,
metrics, bootstrap, and random seeds.

The machine-readable final evaluation specification is
`final_evaluation_protocol.json`, SHA-256
`dbaa2035586c3e9f8f4ed2654d5a129acb2d767d14ded5b168dfc740d8943993`.
Stage 23B must make zero new methodological choices.

Stage 23B must abort before final evaluation if any sealed hash/schema/config
differs; split membership differs or overlaps final training; eligible counts
cannot be reconciled; the reference or threshold procedure cannot be executed
exactly; the protocol hash differs; the model/references are not sealed before
test opening; an unregistered methodological choice or any FINAL_TEST-dependent
choice is required; or final-test customer integrity fails. The complete abort
list is authoritative in the protocol JSON.

## Research-matrix freeze

**AMP MODEL DEVELOPMENT IS CLOSED AFTER STAGE 22.** Stage 23A is governance and
design. Stage 23B is the one frozen final-pipeline evaluation. After Stage 23B,
there is no return to model development based on `FINAL_TEST`, regardless of
whether the outcome is better, equal, or worse than development/confirmation.
Future modelling must be a separate follow-up study and cannot alter this
reported frozen experiment.

**NO NEW MODEL WAS TRAINED OR EVALUATED DURING STAGE 23A.**

**CONFIRMATION ROW-LEVEL DATA NOT ACCESSED DURING STAGE 23A.**

**FINAL_TEST DATA, PREDICTIONS, AND METRICS NOT ACCESSED DURING STAGE 23A.**
