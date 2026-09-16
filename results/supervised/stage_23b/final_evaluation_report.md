# Stage 23B — Frozen Final Evaluation

## Governance and access gate

Every pre-access assertion passed before `FINAL_TEST` features, labels, predictions,
or metrics were loaded. The verified Stage 23A protocol SHA-256 was
`dbaa2035586c3e9f8f4ed2654d5a129acb2d767d14ded5b168dfc740d8943993`. The exact final-training population was
`DEV_TRAIN + DEV_VAL`: 29,655 representation-available
customers. `CONFIRMATION` and `FINAL_TEST` were excluded. The one established
unusable original-training customer (`B08E0670C8A6933E5EC44715F7A80E9D`)
remained excluded and documented.

The four label-free persistence references were recomputed from all
768,923 usable final-training windows and frozen
before test construction. The final model was serialized before the access gate.
The access-gate artifact SHA-256 is `7d1e9d5528e5920b75fad036bc3e007394518e6634f785a80f2ec709abb689ca`.

## Frozen method

The final matrix used the exact ordered 48-feature Stage 16 representation. One
unweighted `HistGradientBoostingClassifier` was fit once with learning rate 0.1,
100 iterations, 15 maximum leaf nodes, minimum leaf size 100, L2 regularization
1.0, no early stopping, and random seed 42. The model artifact SHA-256 is
`81aee2324914ef86280d2ec56e12969ee5de5f6bcebd82e454bdd416cee9fbe1` and its size is 274,176
bytes. Fit runtime was 2.197552 seconds.

The fixed threshold was `0.20070617709942135`. It was not rederived or recalibrated.
Because the model was refit on the combined development population, calibration
may shift; all threshold-dependent metrics below are secondary/descriptive.

## Final result

The eligible final cohort contains 6,356 customers: 542
theft and 5,814 normal, for prevalence
0.085273757080. There were no exclusions and no overlap
with final training. The exact 48-column test matrix was finite and had one row
per governed eligible customer.

Primary metrics:

- ROC-AUC: **0.792512538128**; customer-bootstrap 95% percentile CI
  [0.771213657298, 0.812646883817]
  (bootstrap mean 0.792315557096).
- Average Precision: **0.370158599788**; customer-bootstrap
  95% percentile CI [0.328475324466,
  0.414294943362] (bootstrap mean
  0.371522297266).

Secondary frozen-threshold metrics: precision 0.387543252595, recall
0.413284132841, F1 0.400000000000; TN=5460, FP=354,
FN=318, TP=224.

The bootstrap used 2,000 valid customer-level replicates, seed 42, identical
replicate indices for both primary metrics, and rejected
0 single-class replicates.

## Descriptive comparison and scientific status

Stage 17B `DEV_VAL` recorded ROC-AUC/AP 0.788830821895/0.359794590322.
Stage 17C `CONFIRMATION` recorded 0.789084048160/0.379057855867. The historical
Stage 13 final-cohort window baseline recorded 0.7023135402/0.2193165970.
These are descriptive comparisons only; no superiority test, selection, tuning,
or adaptation followed.

Final status: **A — STRONG STABLE FINAL REPLICATION**. This classification is descriptive
and cannot change the frozen method.

## Historical-access disclosure and closure

This population is a **"final evaluation cohort with disclosed historical baseline
access"**, not a pristine never-seen test cohort. Stage 13 historically evaluated
an obsolete window-level baseline on it. The current FULL_48 direct-customer
method did not yet exist, and Stages 14–22 did not use this population to tune or
select the frozen current method. The historical access is a limitation requiring
disclosure, not grounds for pretending the cohort was untouched.

The final result was accepted as-is without feature engineering, model switching,
hyperparameter tuning, threshold tuning, calibration, subgroup optimization,
resampling, GAN/SMOTE, or any other adaptation.

**AMP FROZEN EXPERIMENT COMPLETE.** No additional model-development experiment
may be added to the current reported study based on this result. Future work must
be explicitly separated from the frozen AMP experiment.
