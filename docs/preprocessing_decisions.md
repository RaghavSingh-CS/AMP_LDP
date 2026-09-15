# Preprocessing Decision Log

This document records the decisions for the first cleaning stage of the SGCC
smart-meter dataset. These are implementation choices, not findings about
causality or model performance. Any later revision should be documented and
applied reproducibly.

## Decisions

| # | Decision | Reason | What was not done | Possible future revision |
|---:|---|---|---|---|
| 1 | Interpret FLAG=0 as normal and FLAG=1 as electricity theft. | This is the supplied SGCC label definition. | Labels are not inferred from consumption or relabelled. | Verify the mapping against authoritative dataset documentation and cite it in the final methodology. |
| 2 | Retain CONS_NO for traceability but exclude it from consumption features. | It uniquely identifies customers but is not a behavioral measurement. | The identifier is not encoded, transformed, or used as a numeric feature. | Use it later only to audit splits, predictions, and errors. |
| 3 | Remove customers whose entire consumption history is missing. | Such rows contain no temporal signal that customer-wise interpolation can recover. | Customer IDs are not hardcoded, and partially observed customers are not removed here. | Revisit the exclusion threshold if a defensible minimum-observation policy is established. |
| 4 | Remove 10/3/2014. | Only 15 of 42,372 source rows contain an observation, making the column effectively unusable. | No other observed date is silently dropped. | Restore it only if a more complete authoritative source becomes available. |
| 5 | Insert 9/18/2016 as an explicit all-missing date. | The header is absent even though the intended time axis is daily. Explicit insertion preserves calendar position and missingness information. | The absent date is not treated as an observed zero. | Replace it with source observations if they are recovered later. |
| 6 | Treat zero as legitimate consumption. | Zero is common and can represent a valid daily reading. | Zeros are not converted to missing values, removed, or replaced. | Reassess only with meter-status evidence distinguishing true zero from outage or error codes. |
| 7 | Retain customers whose available readings are all zero. | Their meaning is not established, and automatic removal could discard valid or informative cases. | All-zero histories are not classified as missing or filtered. | Analyze inactivity and meter-status evidence before deciding whether to flag or exclude them. |
| 8 | Preserve a separate missingness mask. | It retains where values were observed versus constructed, including the inserted date. | Imputed values are not presented as originally observed. | Supply the mask as an auxiliary model input or use it for sensitivity analysis. |
| 9 | Impute customer-wise across time. | Each customer is filled only from that customer's own temporal observations. | No cross-customer, class-level, or population statistic is used. | Compare with seasonal or model-based within-customer imputation during later experiments. |
| 10 | Use linear interpolation for internal gaps and nearest available observations for leading and trailing gaps. | This is deterministic, transparent, and provides a complete sequence for the first baseline. | Leading and trailing gaps are not trend-extrapolated; internal gaps are not filled with class averages. | Evaluate gap-length limits and seasonal interpolation, especially for long missing runs. |
| 11 | Never use FLAG during imputation. | Label-informed filling could create leakage and artificial class separation. | Normal and theft customers do not receive class-specific imputation. | This constraint should remain in later preprocessing versions. |
| 12 | Apply log1p after imputation. | Consumption is strongly right-skewed, while log1p(0)=0 preserves legitimate zeros. | Raw consumption is not standardized or min-max scaled in this stage. | Compare alternative power transforms using training data only. |
| 13 | Retain extreme positive outliers without manual clipping. | Their validity is not yet established, and removal would encode an unsupported assumption. | The value near 800,003 and other extremes are not deleted, replaced, clipped, or winsorized. | Add a documented outlier policy after source validation and train-only sensitivity testing. |
| 14 | Store transformed consumption as float32. | It is adequate for downstream ML and materially reduces memory and storage use. | Identifiers are not cast to numeric values; labels remain integer. | Use higher precision only if numerical validation demonstrates a need. |
| 15 | Do not split into training, validation, or test sets during cleaning. | Cleaning is separate from experimental design, and the source order is label-structured. | No row-order-based or random split is produced. | Define a leakage-safe, reproducible splitting policy in a later stage. |
| 16 | Do not scale consumption during cleaning. | Scaling parameters must later be fitted on training data only to avoid leakage. | No global mean, variance, minimum, or maximum is used for scaling. | Fit and persist the selected scaler after the dataset split is defined. |
| 17 | Do not handle class imbalance during cleaning. | Sampling and loss weighting are modeling decisions, not source-cleaning operations. | No over-sampling, under-sampling, synthetic generation, or class weighting is applied. | Compare imbalance strategies within the training and evaluation design. |
| 18 | Keep preprocessing deterministic and the raw source immutable. | Reproducibility and auditability require stable rules and source preservation. | The raw CSV is never overwritten; preprocessing has no stochastic operation. | Add versioned configurations, checksums, and data-lineage metadata as the pipeline grows. |

## Stage boundary

This stage ends after calendar repair, removal of wholly unobserved customers,
mask creation, customer-wise imputation, log1p transformation, validation, and
Parquet export. It does not perform splitting, scaling, windowing, class
balancing, feature engineering, model training, evaluation, or deployment.

## Stage 2 — Customer Split and Scaling

Stage 2 uses the validated cleaned Parquet and matching missingness mask from
Stage 1. The decisions below define the baseline customer split and scaling
methodology; they do not imply that this is the only valid experimental design.

| # | Decision | Reason | What was not done | Possible future revision |
|---:|---|---|---|---|
| 1 | Use cleaned_data.parquet and missing_mask.parquet as the Stage 2 inputs. | Stage 1 established the validated, imputed, log1p-transformed feature table and its observation mask. | The raw CSV is not cleaned again or used directly for splitting. | Version the input artifact hashes in a future formal data registry. |
| 2 | Split at the customer row level. | Each customer is the independent assignment unit and must not leak between subsets. | Dates and individual readings are not independently assigned to splits. | Revisit only if the research question changes to temporal forecasting with an explicitly time-based design. |
| 3 | Target 70% train, 15% validation, and 15% test. | This provides substantial training data and two held-out subsets. | No tuning or evaluation is performed during this stage. | Compare alternative ratios if later sample-size analysis supports a change. |
| 4 | Stratify customer assignment using FLAG. | Stratification preserves the minority theft proportion across all subsets. | FLAG is not used to calculate feature transformations. | Use grouped or repeated stratification only if future metadata introduces additional grouping constraints. |
| 5 | Use RANDOM_STATE=42. | A fixed seed makes the assignment reproducible. | The source row order is not treated as a split rule. | Manage seeds through a versioned configuration if multiple experiments are introduced. |
| 6 | Shuffle through deterministic stratified splitting rather than use original row order. | The source order is label-structured and would produce biased subsets. | No contiguous or order-based split is produced. | This decision should remain unless an explicitly temporal split is justified. |
| 7 | Assign every customer to exactly one split. | Mutual exclusivity prevents customer leakage. | No customer is duplicated across train, validation, or test. | This rule should remain for customer-level evaluation. |
| 8 | Treat split_manifest.parquet as the authoritative assignment record. | A permanent ID-to-split mapping remains reproducible even if other tables are reordered. | Future stages should not resplit customers independently. | Add experiment/version identifiers if multiple approved manifests are created. |
| 9 | Split masks using the exact data assignments and ordering. | Each mask row must describe the corresponding customer's feature row. | The mask is not randomly split on its own. | Preserve this alignment rule for all later window or sequence artifacts. |
| 10 | Apply StandardScaler after the Stage 1 log1p transformation. | The baseline centers and scales each already-skew-reduced date feature. | MinMaxScaler, RobustScaler, per-customer normalization, and global scalar normalization are not used. | Compare alternative train-fitted scalers as controlled experiments. |
| 11 | Fit StandardScaler only on training consumption features. | Train-only fitting prevents validation and test distribution leakage. | Validation/test rows, labels, identifiers, and masks do not contribute to fitted statistics. | This constraint must remain for any replacement scaler. |
| 12 | Transform validation and test data with the same training-fitted scaler. | All subsets must share one reproducible feature space. | Separate validation or test scalers are not fitted. | No revision is expected unless the entire preprocessing design changes. |
| 13 | Retain CONS_NO for traceability but exclude it from scaling and model features. | IDs are useful for audit and alignment but carry no intended consumption meaning. | IDs are not numerically encoded or passed to StandardScaler. | Continue using IDs only for lineage, joins, and error analysis. |
| 14 | Exclude FLAG from scaling. | FLAG is a target label, not a consumption feature. | Label values do not influence scaler means or variances. | This exclusion should remain. |
| 15 | Exclude the missingness mask from scaling. | Mask values encode observation status and must remain binary. | Mask columns are not standardized or included in fitted statistics. | A later model may accept the unchanged mask as a separate input. |
| 16 | Scale feature-wise across training customers for each date column. | StandardScaler therefore learns one training mean and standard deviation for every chronological date feature. | Scaling is not performed per customer, per class, or with one global scalar. | Evaluate a different representation only as a documented baseline change. |
| 17 | Store scaled consumption as float32. | This preserves Stage 1's memory-efficient ML representation. | IDs and labels are not cast to floating-point features. | Use float64 only if numerical evidence shows float32 is insufficient. |
| 18 | Persist the fitted scaler with joblib. | The exact preprocessing statistics are required for validation, test, inference, and deployment consistency. | The scaler is not reconstructed from held-out data. | Add artifact metadata and versioning as deployment work begins. |
| 19 | Do not correct class imbalance during Stage 2. | Sampling and loss weighting belong to later model-training experiments. | No SMOTE, over-sampling, under-sampling, or class weights are applied. | Compare imbalance strategies only within the training design. |
| 20 | Do not create windows or sequences during Stage 2. | Window length and stride are separate temporal-model decisions. | No sliding-window or sequence-generation artifacts are produced. | Define these parameters in the next preprocessing stage. |
| 21 | Do not train any model during Stage 2. | This stage ends with deterministic split and scaling artifacts. | No autoencoder, GAN, classifier, anomaly scorer, or hyperparameter tuning is implemented. | Training begins only after later pipeline decisions are approved. |
| 22 | Do not allow validation or test statistics to influence training preprocessing. | Held-out independence is required for credible evaluation. | No held-out mean, variance, range, label, or mask information is used in fitting. | This methodological boundary should remain across all future preprocessing. |

Zero-variance training date features, if present, are retained. StandardScaler
maps such features using its documented behavior; Stage 2 reports them rather
than deleting them automatically.

### Stage 2 boundary

Stage 2 ends after creating customer-disjoint scaled train, validation, and test
Parquets, exactly aligned masks, the authoritative split manifest, and the
serialized train-fitted scaler. It performs no windowing, balancing, feature
selection, model training, anomaly scoring, or evaluation.

## Stage 3 — Temporal Windowing

Stage 3 reads the scaled and customer-disjoint Stage 2 Parquets. It creates
reference-only temporal window manifests; it does not duplicate consumption
sequences or train a model.

| # | Decision | Reason | What was not done | Possible future revision |
|---:|---|---|---|---|
| 1 | Read the scaled Stage 2 train, validation, test, and aligned mask Parquets. | Stage 2 fixes customer assignments and train-only scaling for controlled temporal experiments. | Stage 1 data is not rescaled, and customers are not reassigned. | Version input hashes through a formal artifact registry if the pipeline expands. |
| 2 | Fix the initial baseline window length at 30 positions. | A consistent approximately monthly sequence length supports controlled stride comparisons. | Different window lengths are not compared in this stage. | Evaluate alternative lengths separately while holding customer splits fixed. |
| 3 | Support stride configurations 30, 15, and 7. | These represent non-overlapping, 50%-overlapping, and weekly-shifted 30-position windows. | Unsupported strides are rejected. | Change allowed values explicitly if later experiments justify it. |
| 4 | Make no assumption that one stride is inherently superior. | Overlap changes redundancy and temporal resolution, so performance must be compared empirically. | Stride is not selected using labels or preliminary model results. | Compare approved strides using the same downstream method and splits. |
| 5 | Execute stride configurations sequentially. | One-stride-at-a-time processing limits RAM and disk use. | Dense outputs for all stride variants are not generated simultaneously. | Consider parallelism only after resource profiling. |
| 6 | Execute stride 30 for the initial Stage 3 artifact. | It is the requested low-duplication baseline. | Strides 15 and 7 are not generated during the initial run. | Generate them later with unchanged Stage 2 inputs. |
| 7 | Reuse the exact Stage 2 customer assignments. | Fixed assignments make stride comparisons controlled and prevent customer leakage. | Stage 3 does not perform a new split. | Assignments must remain fixed for baseline comparisons. |
| 8 | Never resplit customers during window generation. | A customer must remain in one held-out role. | No random or order-based assignment occurs. | No revision is expected for customer-level evaluation. |
| 9 | Generate windows independently within each customer. | Positions from different customers must never form one sequence. | No window crosses a customer boundary. | Keep this rule for all later sequence artifacts. |
| 10 | Drop incomplete trailing windows instead of padding them. | The baseline uses only full 30-position sequences. | Final partial sequences are not padded or extended. | Compare masked padding later only as a documented experiment. |
| 11 | Inherit FLAG from the customer as weak supervision. | FLAG is customer-level; theft event dates are unavailable. | Windows are not treated as verified theft events or relabelled from patterns. | Use event-level labels if a future authoritative source provides them. |
| 12 | Make no claim that every window from a FLAG=1 customer contains theft. | Positive customers may contain normal intervals, making positive window labels noisy. | Theft dates are not invented or inferred. | Consider weakly supervised methods during later modeling. |
| 13 | Preserve the aligned mask as separate auxiliary information. | It records which positions were observed versus imputed. | Mask values are not combined with scaled consumption. | Evaluate the mask as a separate later model input. |
| 14 | Do not pass missingness into a model during Stage 3. | This stage creates manifests only. | No mask-aware architecture or training is implemented. | Decide mask usage alongside model design. |
| 15 | Do not filter windows by missingness. | A defensible threshold requires empirical analysis. | Highly missing, fully missing, and zero-looking windows are retained. | Evaluate thresholds using the retained metadata. |
| 16 | Retain missing_count and missing_fraction for every window. | These fields support later diagnostics without losing provenance. | Missingness is not inferred from scaled consumption. | Add gap-run summaries if later analysis needs them. |
| 17 | Allow windows to span the removed 2014-10-03 date. | The baseline treats the 1,034 features as one positional sequence despite the known discontinuity. | No break, padding date, or special exclusion is introduced. | Compare calendar-contiguous handling separately. |
| 18 | Store manifests rather than dense 30-value arrays. | Existing split files plus IDs and indices reconstruct windows while avoiding duplication. | Consumption and mask sequences are not embedded per manifest row. | Materialize batches lazily in later training code. |
| 19 | Recover windows using split, CONS_NO, start index, and inclusive end index. | These references deterministically identify the Stage 2 slice; actual boundary dates are also retained. | Generic day numbers do not replace actual feature dates. | Add a dataset accessor during model implementation. |
| 20 | Perform no class balancing or model training. | Window creation is separate from sampling, architecture, optimization, and evaluation. | No sampling, Autoencoder, GAN, classifier, anomaly score, or threshold is created. | Address these choices only in approved later stages. |

### Stage 3 boundary

Stage 3 ends after generating and independently validating the stride-30 train,
validation, and test window manifests plus their configuration and missingness
summary. Strides 15 and 7 remain planned, not executed. No dense window tensor,
class balancing, model architecture, training, or evaluation is produced.

### Stage 3B — Missingness Quality Analysis

Stage 3B is descriptive analysis only. It does not filter or rewrite the
canonical stride-30 manifests.

| Decision | Reason | What was not done | Possible future revision |
|---|---|---|---|
| Pause model implementation to analyze missingness quality. | Stage 3 showed that 340,994 windows (23.67%) have all 30 source positions originally missing. Imputed numerical values do not create directly observed consumption information. | No Autoencoder, GAN, classifier, anomaly score, training, or evaluation was started. | Resume model design only after a filtering policy is reviewed and approved. |
| Compare candidate retention thresholds empirically. | Rules 0%, <=10%, <=25%, <=50%, <=75%, and <100% have different window, customer, and class effects. | No threshold was selected from intuition alone. | Use the evidence tables and later sensitivity analysis to approve a rule. |
| Interpret thresholds explicitly using missing counts out of 30. | Integer counts avoid floating-point ambiguity; for example, <=10% permits at most 3 missing positions. | Threshold comparisons do not rely on approximate float equality. | Keep count-based semantics if other window lengths are evaluated. |
| Preserve all Stage 3 manifests unchanged. | Stage 3B must remain reversible analysis rather than a data mutation. | No filtered train, validation, or test manifest was created. | Generate versioned filtered artifacts only after an explicit decision. |
| Examine effects at both window and customer levels. | Large window retention can hide customers who lose every usable window. | The analysis does not rely on window percentages alone. | Retain customer-coverage checks in every future filtering experiment. |
| Check class-composition distortion explicitly. | Theft-customer windows have higher missingness on average, so filtering reduces their representation disproportionately. | No balancing or label-specific filtering is applied. | Evaluate downstream sampling only after a common quality rule is approved. |
| Analyze temporal position and consecutive fully-missing runs. | Missingness is uneven across dates and often occurs in long blocks. | No date position or long-run pattern is automatically excluded. | Consider temporal-quality features or policies as controlled experiments. |
| Treat removing only fully missing windows as a proposed minimum rule pending review. | A 30/30-missing window contains zero observed readings; this proposal retains 76.33% of windows and causes one theft-labelled customer to lose all windows. | This proposal is not canonical and has not been applied. | Review customer coverage and class effects before accepting or revising it. |
| Keep more aggressive thresholds unresolved. | The analysis does not establish an objectively optimal aggressive cutoff. | No <=75%, <=50%, <=25%, <=10%, or zero-missing-only policy is adopted. | Compare model sensitivity only after the minimum quality policy is agreed. |

The complete evidence and proposed recommendation are in
results/missingness_analysis/report.md. Model training remains intentionally
paused pending review of the filtering decision.

### Stage 3C — Window Quality Filtering

Stage 3C adopts and applies the minimum baseline eligibility rule to the
unfiltered stride-30 manifests. It creates versioned usable subsets while
leaving the canonical Stage 1, Stage 2, Stage 3, and Stage 3B artifacts
unchanged.

| Decision | Reason | What was not done | Possible future revision |
|---|---|---|---|
| Retain a window when `missing_fraction < 1.0`, equivalently `missing_count < 30`. | This requires at least one directly observed position in every eligible 30-position sequence. A 30/30-missing window has imputed numerical values but no direct consumption signal. | No stricter cutoff is applied. Partially missing windows remain eligible. | Compare stricter common thresholds only as documented sensitivity experiments. |
| Apply the same rule to train, validation, and test manifests. | A common data-quality definition avoids label- or split-specific selection rules. | No label-specific filtering, resplitting, rebalancing, over-sampling, or under-sampling is performed. | Any later sampling policy belongs to model training and must use training data only. |
| Preserve each usable manifest as an exact, order-preserving subset of its canonical Stage 3 manifest. | This retains window IDs, customer IDs, labels, split assignments, temporal boundaries, and missingness metadata without transformation. | No consumption values, masks, dates, labels, identifiers, or manifest fields are rewritten. | Add formal artifact version metadata if multiple filtering policies are retained. |
| Accept the measured class and customer-coverage effect of the minimum rule for the initial baseline. | The rule removes 340,994 of 1,440,478 windows (23.67%) and retains 1,099,484 (76.33%). One theft-labelled train customer loses all 34 windows; no normal customer loses all windows. | The lost customer is not moved to another split, given synthetic windows, or recovered using a lower-quality exception. | Report customer coverage in evaluation and test a documented alternative only if the loss materially affects conclusions. |
| Treat missingness as associated with the supplied customer label and therefore as a potential shortcut signal. | Theft-labelled windows are removed disproportionately: theft share changes from 8.53% before filtering to 7.91% after filtering. Aggressive quality thresholds could further distort class composition. | Missingness is not interpreted as theft evidence, and the baseline is not allowed to learn directly from missingness metadata. | Evaluate missingness-aware variants separately and disclose their shortcut-risk controls. |
| Use only the scaled consumption sequence as the baseline model input. | This isolates the temporal consumption baseline and avoids direct use of missingness or identity metadata. | The missingness mask, `missing_fraction`, `FLAG`, and `CONS_NO` are not baseline model inputs. `FLAG` remains the evaluation label; `CONS_NO` remains traceability metadata. | Test mask-aware architectures only as a separately named sensitivity study. |
| Keep model construction outside Stage 3C. | This stage establishes a reproducible eligible-window population only. | No dense tensor, data loader, Autoencoder, GAN, classifier, anomaly score, training, tuning, or evaluation is created. | Begin modeling in a separately approved stage using the usable manifests. |

The retained population contains 1,012,539 normal-labelled windows and 86,945
theft-labelled windows. The removed-only-fully-missing rule is deliberately a
minimum eligibility policy, not a claim that every retained partially missing
window is equally reliable. Future missingness-threshold sensitivity analyses
must keep the customer split fixed and report window retention, customer
coverage, and class composition together.

## Stage 4 — Baseline Autoencoder

Stage 4 implements the first anomaly-detection model pipeline. At implementation
time, full training was intentionally **not** executed by Codex. The user
subsequently executed the run manually on CPU. It used 708,308 normal training
windows and 152,246 normal validation windows, ran 35 of at most 50 epochs, and
triggered early stopping. The best qualifying checkpoint is epoch 30 with
normal-validation MSE `0.043589054377373454`. The test set remained untouched,
and no anomaly threshold was selected during Stage 4.

| Decision | Adopted baseline and reason |
|---|---|
| Model | A fully connected Autoencoder is the first anomaly-detection baseline. It learns to reconstruct normal consumption behavior without treating `FLAG` as a network feature. |
| Architecture | `30 → 16 → 8 → 4 → 8 → 16 → 30`, with latent dimension 4. Hidden layers use ReLU. The final output is linear because StandardScaled inputs can be negative or positive. |
| Training population | Usable stride-30 `FLAG=0` train windows only. Theft-labelled train windows are not used to update model parameters. The theft-labelled train customer with zero usable windows is not recovered, replaced, or resplit. |
| Validation population | Usable stride-30 `FLAG=0` validation windows only for validation loss, early stopping, and best-checkpoint selection. Theft-labelled validation windows are used only after training to inspect reconstruction-error distributions. |
| Model input | Exactly 30 scaled consumption values. `FLAG`, `CONS_NO`, the missingness mask, `missing_count`, `missing_fraction`, dates, and window IDs are excluded from network input and remain labels or metadata. |
| Loss | `nn.MSELoss()`. Mean squared error directly optimizes reconstruction and penalizes larger reconstruction mistakes more strongly. MAE, Huber, BCE, and weighted MSE are not used in this baseline. |
| Optimizer | Adam with learning rate `1e-3` and standard defaults. No learning-rate search is performed. |
| Batch and epochs | Default batch size 256 and maximum 50 epochs; both are CLI-configurable. The default DataLoader worker count is zero to avoid Windows process duplication and excess memory use. |
| Early stopping | Monitor normal validation MSE with patience 5 and minimum delta `1e-5`. Save the state-dictionary checkpoint corresponding to the best qualifying validation MSE. |
| Reproducibility | Seed 42 for Python, NumPy, PyTorch, and CUDA when available. Deterministic modes are requested where practical, although exact determinism can depend on device, backend, and operations. |
| Window configuration | Window length 30 and stride 30. Strides 15 and 7 have not been executed. |
| Anomaly-score candidate | Per-window reconstruction MSE. No anomaly threshold is selected during Stage 4, and no `MSE > threshold` classifier is implemented as a final decision. |
| Test-set policy | Test data is excluded from training, model selection, validation-distribution analysis, and threshold decisions. It remains reserved for later final evaluation. |
| Weak supervision | `FLAG=1` is a customer-level label; an individual window from a theft-labelled customer is not guaranteed to contain theft behavior. Validation error separation alone is not final performance evidence. |
| Memory policy | Load one processed split into a compact float32 NumPy matrix, vectorize customer-ID mapping once, retain compact integer manifest references, and slice 30 values lazily in `Dataset.__getitem__`. Train and validation are held only when training requires both; test is never loaded. |

The real run saves a state-dictionary checkpoint and model configuration under
`models/autoencoder/`, plus training history, summary, training curve,
validation reconstruction errors, and an error-distribution plot under
`results/autoencoder/stride_30/`. The saved validation-error table contains
metadata and per-window MSE only, not reconstructed 30-value arrays.

Future controlled comparisons may include different latent dimensions, MAE,
Huber loss, temporal CNN or recurrent Autoencoders, strides 15 or 7, and
missingness-aware variants. None of these comparisons has been executed. GANs,
ensembles, threshold selection, and test evaluation remain outside Stage 4.

## Stage 5 — Validation-Based Anomaly Threshold

Stage 5 uses the completed Autoencoder's per-window validation reconstruction
MSE as the anomaly-score candidate. All metrics below are validation metrics
under weak customer-level labels; they are not confirmed theft-event results.

| Decision | Reason | What was NOT done | Possible future revision |
|---|---|---|---|
| Use validation reconstruction MSE as the anomaly score and keep test data untouched. | Threshold selection must be completed without contaminating the reserved final test evaluation. | No test Parquet, test reconstruction score, test label, or test metric was read or calculated. | Apply the frozen threshold once in a separately approved final test stage. |
| Compare normal-only p90, p95, p97.5, and p99 thresholds with efficient supervised validation candidates. | Percentiles expose explicit normal false-positive tradeoffs; best-F1 and best-balanced-accuracy candidates show what label-optimized selection would imply. | No millions-point brute-force grid, accuracy-only optimization, or test-tuned threshold was used. | Add an explicit utility investigation-cost model only if a defensible business cost ratio becomes available. |
| Adopt threshold `0.16436323150992393`, the normal-validation 95th percentile. | It controls normal-window FPR at 5.0005% without fitting the cutoff directly to noisy theft-window labels. It avoids the best-F1 candidate's impractical 79.30% FPR while retaining more theft-labelled windows than p99. | The mathematically highest validation F1 was not adopted blindly, and the threshold is not claimed to be globally optimal. | Reassess the operating point using deployment capacity or explicit false-positive/false-negative costs. |
| Record selected-threshold validation metrics: TP 1,450, TN 144,633, FP 7,613, FN 11,845, precision 0.159991, recall 0.109064, F1 0.129707, specificity 0.949995, FPR 0.050005, and predicted anomaly rate 5.4748%. | These values make the chosen baseline's detection/false-alarm tradeoff explicit under the available labels. | Accuracy is not treated as the primary metric, and the values are not presented as final test performance. | Report confidence intervals or customer-aware resampling in later robustness work. |
| Treat the best-F1 and best-balanced-accuracy thresholds as diagnostics, not deployment choices. | Best F1 is only 0.164975 and flags 80.21% of validation windows; best balanced accuracy is 0.556662 with similarly high false-positive volume. | No arbitrary business cost ratio was invented to justify either cutoff. | Revisit supervised selection if event-level labels or investigation-cost evidence becomes available. |
| Report validation ROC-AUC 0.564310 and PR-AUC 0.110242 as weakly supervised ranking summaries. | Threshold-independent summaries confirm that window-level score separation is limited. | These AUCs are not interpreted as event-level or test performance. | Compare later models using the same untouched test protocol after all validation decisions are frozen. |
| Explicitly retain the weak-supervision limitation. | `FLAG=1` is customer-level, so an individual theft-labelled window may contain normal behavior; 44.39% of theft-labelled windows score at or below the normal median. | Every `FLAG=1` window is not treated as a confirmed anomalous event. | Use event timestamps or finer labels if an authoritative source becomes available. |
| Keep customer aggregation diagnostic only. | Mean/max/p95 customer ROC-AUC values are 0.5751/0.5658/0.5781 and PR-AUC values are 0.1428/0.1248/0.1550. P95 improves separation modestly over window ROC-AUC 0.5643 and PR-AUC 0.1102, but remains weak. | No separate customer-level final threshold was selected. | Evaluate customer-level decision rules in a dedicated validation design. |
| Record that reconstruction MSE is not strongly driven by missing fraction in this validation set. | Overall Pearson/Spearman correlations are -0.0140/-0.0417; within-label correlations are also small and negative. | The model, Stage 3C filter, or selected threshold was not changed solely from this diagnostic. | Repeat with mask-aware and stricter-missingness sensitivity variants. |
| Freeze the selected threshold in `models/autoencoder/ae_stride30_threshold.json`. | A versioned artifact prevents later test results from changing the validation-selected operating point. | No anomaly threshold is recomputed from test data. | Version a replacement only after a new approved validation experiment. |

Lower thresholds increase theft-labelled recall but also raise normal false
alarms; higher thresholds reduce false alarms while missing more theft-labelled
windows. The selected p95 threshold is a defensible first baseline rather than
a claim of optimality. Stage 5 ends before Autoencoder test evaluation, GAN
work, alternative strides, ensembles, dashboards, or deployment.

## Stage 6 — Frozen Test Evaluation

Stage 6 evaluates the stride-30 fully connected Autoencoder once on the reserved
test customers. The Stage 4 checkpoint and Stage 5 threshold were hashed before
and after evaluation and remained unchanged.

| Decision | Reason | What was NOT done | Possible future revision |
|---|---|---|---|
| Freeze the Stage 5 threshold before reading test outcomes and load it authoritatively from `models/autoencoder/ae_stride30_threshold.json`. | The test split must measure generalization of the already selected baseline rather than influence its operating point. The exact rule is `reconstruction_mse > 0.16436323150992393`. | The threshold was not recomputed, adjusted, rounded independently, or optimized against test labels. | Future models must freeze their validation decisions before using the same test protocol. |
| Evaluate all 165,020 usable test windows exactly once: 151,985 normal and 13,035 theft-labelled. | This preserves the Stage 2 customer split, Stage 3C eligibility policy, Stage 4 model input, and complete eligible test population. | No customer was recovered, removed, resplit, or reprocessed; no full reconstruction vectors were saved. | Keep this population fixed for controlled future baseline comparisons. |
| Record frozen window-level results: TP 1,578, TN 144,478, FP 7,507, FN 11,457, precision 0.173693, recall 0.121059, F1 0.142676, specificity 0.950607, FPR 0.049393, balanced accuracy 0.535833, accuracy 0.885081, and predicted anomaly rate 5.5054%. | These metrics provide the final operating-point record for the stride-30 FC Autoencoder under the frozen validation threshold. | Accuracy is not treated as the primary metric, and the confusion matrix is not used for retrospective tuning. | Compare future models using precision, recall, F1, FPR, ranking metrics, and operational tradeoffs together. |
| Record test ROC-AUC 0.561688 and PR-AUC 0.114411. | Threshold-independent ranking remains only modestly above random/prevalence baselines and supports a conservative interpretation. | These values are not presented as confirmed theft-event performance. | Later models must improve ranking while preserving a defensible false-positive rate. |
| Interpret validation-to-test behavior as broadly stable. | Absolute test-minus-validation differences are small: precision 0.013702, recall 0.011995, F1 0.012969, specificity/FPR 0.000612, ROC-AUC 0.002623, PR-AUC 0.004169, and anomaly-rate percentage 0.030617. | The apparent small test improvements in precision, recall, F1, and PR-AUC were not used to alter the model. | Use repeated or external validation only in a separately approved robustness study. |
| Retain customer aggregation as diagnostic only. | Test customer p95 aggregation gives ROC-AUC 0.599251 and PR-AUC 0.168575; mean/max ROC-AUC values are 0.591093/0.587050. Mean anomalous-window fractions are 0.0482 for normal and 0.1102 for theft-labelled customers. | No fraction cutoff or final customer-level classifier was created. | Develop a customer decision rule only through a new validation design, never from these test outcomes. |
| Record that test reconstruction MSE remains weakly related to missing fraction. | Overall Pearson/Spearman correlations are -0.0153/-0.0418; normal values are -0.0133/-0.0399 and theft-labelled values are -0.0365/-0.0627. | The model, filter, imputation, or threshold was not changed from this diagnostic. | Repeat the diagnostic for later missingness-aware sensitivity models. |
| Preserve the customer-level weak-supervision caveat. | `FLAG=1` identifies a theft-labelled customer, not a confirmed theft event in every window. Window TP/FN terminology is therefore an approximation under weak labels. | No causal claim or event-level detection claim is made from high-error examples. | Use event-timestamp labels if an authoritative dataset becomes available. |
| Classify the frozen baseline as weak. | ROC-AUC 0.561688, PR-AUC 0.114411, F1 0.142676, and recall 0.121059 show limited separation despite stable generalization and controlled FPR near 5%. | The baseline is not described as moderate or strong, and its stability is not confused with high predictive power. | Require future stride/model/GAN experiments to beat this frozen baseline under the same evaluation principles. |

No Autoencoder retraining, architecture modification, threshold adjustment,
test-driven model selection, stride comparison, GAN, ensemble, dashboard, or
deployment work was performed. The final machine-readable performance record
is `results/autoencoder/stride_30/test_evaluation/baseline_final_summary.json`.

## Stage 7 — Stride-15 Autoencoder Ablation

Stage 7 evaluates stride 15 as the first temporal-overlap ablation. Its purpose
is to determine whether overlapping 30-day windows improve anomaly separation
relative to the frozen stride-30 fully connected Autoencoder. The controlled
change is only `stride: 30 → 15`. The customer split, Stage 1 transformations,
train-fitted scaler, window length 30, `missing_count < 30` eligibility rule,
architecture `30 → 16 → 8 → 4 → 8 → 16 → 30`, ReLU/linear activations, MSE,
Adam, learning rate 0.001, batch size 256, maximum 50 epochs, patience 5,
minimum delta 1e-5, seed 42, normal-only training and early stopping,
normal-validation P95 threshold methodology, and frozen-test evaluation
methodology remain fixed.

The programmatic window calculation is
`floor((1,034 - 30) / 15) + 1 = 67` windows per customer. The final window ends
at feature index 1,019, leaving 14 trailing chronological positions unused.
The original Stage 2 customer assignments are reused exactly, with no overlap
between train, validation, and test customers.

| Split | Customers | Raw windows | Fully missing | Usable | Usable normal | Usable theft-labelled | Retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 29,656 | 1,986,952 | 466,091 | 1,520,861 | 1,400,965 | 119,896 | 76.5424% |
| Validation | 6,355 | 425,785 | 98,373 | 327,412 | 301,102 | 26,310 | 76.8961% |
| Test | 6,356 | 425,852 | 99,426 | 326,426 | 300,631 | 25,795 | 76.6525% |
| Overall | 42,367 | 2,838,589 | 663,890 | 2,174,699 | 2,002,698 | 172,001 | 76.6120% |

Before filtering, the raw window population contains 2,596,585 normal and
242,004 theft-labelled windows (8.5255% theft-labelled). After applying only
the fixed fully-missing-window rule, the theft-labelled share is 7.9092%.
One theft-labelled training customer (`B08E0670C8A6933E5EC44715F7A80E9D`)
loses all 67 windows; no normal, validation, or test customer loses all usable
windows. This is diagnostic evidence only and does not reopen the fixed
missingness-policy decision.

Shared preprocessing, Dataset, and training code now accepts strides 30 and 15.
Stride-specific paths isolate the future checkpoint at
`models/autoencoder/ae_stride15_baseline.pt`, its config at
`models/autoencoder/ae_stride15_baseline_config.json`, and training outputs at
`results/autoencoder/stride_15/`; the frozen stride-30 paths are unchanged.
The training entry point continues to exclude test data and does not select a
threshold. After manual training, the same normal-validation 95th-percentile
threshold policy and the same one-time frozen-test methodology are to be used;
no numeric stride-15 threshold or performance result is recorded at this stage.

A normal-only 256-window training smoke test produced finite `(256, 30)` input
and output tensors, finite MSE, and a successful forward/backward optimizer
step. It loaded no test data and wrote no artifact. The compact train split
matrix occupied 117.0 MiB and the sampled integer manifest references occupied
2 KiB. The real run retains lazy NumPy slicing, batched loading, and
`num_workers=0` on Windows; it does not construct a dense all-window tensor.

Full stride-15 training is intentionally executed manually by the user:

    python -m src.training.train_autoencoder --stride 15

## Stage 8 — Stride-15 Evaluation and Stride Ablation

Stage 8 verifies the manually completed stride-15 training run, freezes the
normal-validation P95 threshold before test access, evaluates the reserved test
split exactly once, and compares saved stride-15 results against the frozen
stride-30 baseline. No model was retrained. The stride-15 checkpoint SHA-256 is
`7df22d08a04aacba45fbab701dab517bf431892b61ea4fcb7d68f7d753cc6547`.

### Training verification

The checkpoint and configuration retain architecture
`30 → 16 → 8 → 4 → 8 → 16 → 30`, latent dimension 4, ReLU hidden layers,
linear output, MSE, Adam, learning rate 0.001, batch size 256, patience 5,
minimum delta 1e-5, and seed 42. Training used 1,400,965 usable `FLAG=0`
training windows only; early stopping used 301,102 usable `FLAG=0` validation
windows only. The CPU run executed 36 epochs, stopped early after epoch 36,
selected epoch 31, achieved best normal-validation MSE
`0.04118696386392505`, and ran for 1,251.49 seconds. The saved summary records
`test_set_used: false` and `anomaly_threshold_selected: false`; all 327,412
saved validation reconstruction errors are finite and nonnegative.

| Validation population | Count | Mean | Median | Std | Min | P25 | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 301,102 | 0.041187 | 0.017682 | 0.095739 | 0.000051 | 0.005784 | 0.017682 | 0.039526 | 0.089932 | 0.152459 | 0.425142 | 8.414934 |
| Theft-labelled | 26,310 | 0.073601 | 0.020584 | 0.197337 | 0.000048 | 0.009521 | 0.020584 | 0.052271 | 0.164702 | 0.321152 | 0.874865 | 6.058082 |

The theft/normal validation median ratio is 1.1641 and mean ratio is 1.7870.
Theft-labelled proportions exceeding normal P90/P95/P97.5/P99 are
16.4082%/10.7678%/6.7199%/3.6754%. Validation ROC-AUC is 0.563470 and PR-AUC
is 0.109945. Because `FLAG` is customer-level, these are weakly supervised
window-label results rather than confirmed theft-event results.

### Frozen threshold and test

The authoritative threshold is the normal-validation 95th percentile,
`0.1524585030972959`, saved in
`models/autoencoder/ae_stride15_threshold.json` with SHA-256
`6166e63c36599e9eb17efb8f8d17df5cab146d2293cf1af93b312ee772ab8630`.
Best-F1 and best-balanced-accuracy candidates were diagnostic only. At P95,
validation TP/TN/FP/FN are 2,833/286,046/15,056/23,477; precision 0.158365,
recall 0.107678, F1 0.128193, specificity 0.949997, FPR 0.050003, balanced
accuracy 0.528837, and anomaly rate 5.4638%. The checkpoint and threshold hashes
were recorded before loading test data and remained unchanged afterward.

The one-time frozen test covers 326,426 windows: 300,631 normal and 25,795
theft-labelled. TP/TN/FP/FN are 3,094/285,739/14,892/22,701; precision
0.172023, recall 0.119946, F1 0.141340, specificity 0.950464, FPR 0.049536,
balanced accuracy 0.535205, accuracy 0.884835, anomaly rate 5.5100%, ROC-AUC
0.561077, and PR-AUC 0.113595. Validation-to-test behavior is broadly stable;
the frozen threshold was not changed after test outcomes were observed.

Test customer mean/max/P95 MSE aggregation gives ROC-AUC
0.587743/0.580410/0.600611 and PR-AUC 0.155437/0.130475/0.171796. Mean
anomalous-window fractions are 0.04791 for normal and 0.10876 for
theft-labelled customers. These remain diagnostics; no customer decision
threshold was created.

Test missing-fraction Pearson/Spearman correlations with reconstruction MSE are
-0.0330/-0.0597 overall, -0.0328/-0.0577 for normal, and -0.0482/-0.0807 for
theft-labelled windows. They remain weak, although modestly more negative than
stride 30's overall -0.0153/-0.0418.

### Controlled stride comparison

The customer split, preprocessing, StandardScaler, window length, usable rule,
architecture, loss, optimizer, learning rate, batch size, early stopping, seed,
P95 threshold method, and frozen-test method are fixed; only stride changes.
Stride 15 lowers best validation MSE from 0.043589 to 0.041187 but does not
improve window-level detection.

| Frozen test metric | Stride 30 | Stride 15 | Difference 15−30 |
|---|---:|---:|---:|
| Precision | 0.173693 | 0.172023 | -0.001670 |
| Recall | 0.121059 | 0.119946 | -0.001113 |
| F1 | 0.142676 | 0.141340 | -0.001336 |
| Specificity | 0.950607 | 0.950464 | -0.000143 |
| FPR | 0.049393 | 0.049536 | +0.000143 |
| Balanced accuracy | 0.535833 | 0.535205 | -0.000628 |
| ROC-AUC | 0.561688 | 0.561077 | -0.000611 |
| PR-AUC | 0.114411 | 0.113595 | -0.000816 |
| Anomaly rate | 5.5054% | 5.5100% | +0.0046 pp |

Customer P95 ROC-AUC/PR-AUC changes from 0.599251/0.168575 to
0.600611/0.171796, a small improvement, while mean and maximum aggregation are
slightly worse. Overall, stride 15 is effectively equivalent and slightly
worse at the window level; it does not meaningfully beat stride 30.

Stride 7 is not recommended next for this fully connected baseline. Stride 15
increases windows/customer from 34 to 67 and usable training volume from
708,308 to 1,520,861, materially increasing CPU time without improving core
detection. Stride 7 would create 144 windows/customer—about 2.15 times the
stride-15 volume. It remains unexecuted and can be reconsidered only with a
model or hypothesis that can plausibly exploit denser overlap.

## Stage 9 — 1D CNN Autoencoder

Stage 9 introduces a compact 1D CNN Autoencoder as the next architecture
experiment. The stride-15 FC ablation reduced reconstruction loss but did not
improve frozen-test anomaly discrimination and approximately doubled training
cost. Therefore the next controlled axis is architecture rather than additional
window overlap. Window length returns to 30 and stride returns to 30 for lower
cost and direct comparison with the frozen FC baseline.

### Controlled design

The Stage 1–3 preprocessing, customer split, train-fitted StandardScaler,
30-position window length, stride 30 manifests, `missing_count < 30` usable rule,
normal-only training, MSE loss, Adam optimizer, learning rate 0.001, batch size
256, maximum 50 epochs, patience 5, minimum delta 1e-5, seed 42, and zero
Windows DataLoader workers remain fixed. Only the model architecture changes
from fully connected global mixing to convolutional local receptive fields and
shared filters.

The model consumes `(batch, 1, 30)`, where the single channel is scaled
consumption. The existing model-agnostic `WindowDataset` continues to return
`(batch, 30)` and the CNN training pipeline performs the channel reshape. No
missingness mask is added as a channel. `FLAG`, `CONS_NO`, missing counts and
fractions, dates, and window IDs remain excluded from model input.

### Exact architecture

| Stage | Operation | Output per example |
|---|---|---|
| Input | One scaled consumption channel | `(1, 30)` |
| Encoder | Conv1d 1→8, kernel 3, stride 1, padding 1; ReLU | `(8, 30)` |
| Encoder | MaxPool1d kernel 2, stride 2 | `(8, 15)` |
| Encoder | Conv1d 8→16, kernel 3, stride 1, padding 1; ReLU | `(16, 15)` |
| Encoder | MaxPool1d kernel 3, stride 3 | `(16, 5)` |
| Bottleneck | Flatten 80; Linear 80→8; ReLU | `(8)` |
| Decoder | Linear 8→80; ReLU; reshape | `(16, 5)` |
| Decoder | ConvTranspose1d 16→8, kernel 3, stride 3; ReLU | `(8, 15)` |
| Output | ConvTranspose1d 8→1, kernel 2, stride 2; linear output | `(1, 30)` |

No decoder adjustment, cropping, padding, or output padding is required:
`(5−1)×3+3=15` and `(15−1)×2+2=30`. The model has 2,209 trainable parameters.
Its latent dimension is 8 rather than the FC baseline's 4 because convolutional
feature compression to `(16,5)` occurs before the dense bottleneck. This is the
initial CNN baseline, not a latent-dimension sweep.

Convolutional filters provide a mechanism for local temporal pattern learning
and weight sharing. They may represent local spikes, drops, flat regions,
short-term transitions, or recurring motifs, but no such learned semantics are
claimed before later filter inspection.

### Training and evaluation boundary

Training uses the 708,308 usable stride-30 `FLAG=0` train windows derived from
the Stage 3C filter summary. Early stopping and best-checkpoint selection use
only the 152,246 usable `FLAG=0` validation windows. Theft-labelled validation
windows are scored only after training using the reloaded best checkpoint. The
training script saves history, summary, curve, and per-window validation MSE,
but it does not choose a threshold or load test data.

The CNN threshold method is the 95th percentile of normal-validation
reconstruction MSE. It is frozen before test evaluation under the same FC
baseline methodology. Best-F1 remains diagnostic and does not replace P95 as
the authoritative threshold.

A synthetic `(256,1,30)` smoke batch encoded to `(256,8)` and reconstructed to
`(256,1,30)`. A real normal-only train batch loaded as `(256,30)`, reshaped to
`(256,1,30)`, produced finite input, output, MSE, and gradients, and completed
one Adam step. The train split matrix occupied 117.0 MiB; one float32 input or
output batch occupied 30 KiB. No full epoch, checkpoint, validation scoring,
threshold selection, or test access occurred. CPU runtime will be the primary
bottleneck because convolutional operations repeat across approximately 2,767
training batches per epoch, but memory remains modest under lazy window slicing.

Full CNN training was executed manually by the user:

    python -m src.training.train_cnn_autoencoder --stride 30

The CPU run used 708,308 normal training windows and 152,246 normal validation
windows, executed 48 epochs, stopped early, and retained epoch 43 with normal
validation MSE `0.03020924524280455`. Runtime was 1,945.98 seconds. The raw
epoch-47 validation MSE was slightly lower but did not exceed the configured
`1e-5` minimum-improvement requirement; epoch 43 is therefore the authoritative
checkpoint. The checkpoint SHA-256 is
`26f38e47ae30f1d51a1ec9ecb738588ecd67dbf0b522d92e3168d4f3c53a31ae`.
No threshold was selected and test data was not used during training.

## Stage 10 — CNN Threshold, Frozen Test, and FC Comparison

Stage 10 verified the saved CNN training lineage, analyzed the 165,541 saved
validation errors, froze a validation-only P95 threshold, evaluated the
reserved test split under that fixed rule, and compared the CNN with the frozen
stride-30 FC baseline. No CNN or FC retraining occurred.

### Validation threshold

Normal/theft-labelled validation mean MSE values are `0.030209`/`0.048112` and
medians are `0.013406`/`0.015110`. Validation ROC-AUC is `0.555699` and PR-AUC
is `0.101596`, showing substantial score overlap under customer-level weak
labels. The authoritative normal-validation P95 threshold is
`0.10475244373083115`, saved at
`models/cnn_autoencoder/cnn_ae_stride30_threshold.json` with SHA-256
`c027aa9eb6e8137247c6ebd0ff6a3aafe0a48f9ab99b04423abdb4df199c6ed5`.
At this threshold, validation TP/TN/FP/FN are 1,241/144,633/7,613/12,054;
precision is `0.140163`, recall `0.093343`, F1 `0.112059`, specificity
`0.949995`, and anomaly rate `5.3485%`. Supervised best-F1 candidates were
diagnostic only and did not alter P95.

### Frozen test evaluation

The frozen checkpoint and threshold hashes were recorded before test access and
remained unchanged. The final test population contains 165,020 windows: 151,985
normal and 13,035 theft-labelled. The fixed rule produced TP/TN/FP/FN of
1,361/144,460/7,525/11,674; precision `0.153162`, recall `0.104411`, F1
`0.124173`, specificity `0.950489`, FPR `0.049511`, balanced accuracy
`0.527450`, ROC-AUC `0.555689`, PR-AUC `0.104788`, and anomaly rate `5.3848%`.
Validation-to-test behavior is broadly stable.

Customer p95-MSE aggregation has test ROC-AUC `0.579486` and PR-AUC
`0.144033`; it remains diagnostic and no customer classifier was fitted. The
overall reconstruction-MSE/missing-fraction Spearman correlation is `-0.052406`
(`-0.050107` normal and `-0.078384` theft-labelled), so missingness remains a
weak correlate rather than a basis for changing preprocessing.

The initial test command completed scoring but encountered a cleanup-only API
error before writing artifacts because `ProcessedSplitData` has no `close()`
method. The corrected invocation wrote the frozen results and then encountered
a report-formatting key mismatch after metrics and score artifacts had already
been saved. The Markdown report was finalized from those saved metrics without
another inference pass. These execution errors did not alter the checkpoint,
threshold, predictions, or reported metrics.

### FC versus CNN

The comparison holds the split, preprocessing, 30-day stride-30 windows,
normal-only training, MSE/Adam settings, P95 threshold method, and frozen-test
policy fixed. The CNN reduced best normal-validation MSE by `30.70%`, but used
2,209 versus 1,362 parameters (`1.62×`) and 1,945.98 versus 600.96 CPU seconds
(`3.24×`). Its window-level discrimination is consistently weaker:

| Frozen test metric | FC | CNN | CNN−FC |
|---|---:|---:|---:|
| Precision | 0.173693 | 0.153162 | -0.020531 |
| Recall | 0.121059 | 0.104411 | -0.016647 |
| F1 | 0.142676 | 0.124173 | -0.018503 |
| Specificity | 0.950607 | 0.950489 | -0.000118 |
| ROC-AUC | 0.561688 | 0.555689 | -0.005999 |
| PR-AUC | 0.114411 | 0.104788 | -0.009623 |

Lower reconstruction loss did not translate into better theft-label ranking or
P95 operating-point performance. The fully connected Autoencoder therefore
remains the preferred frozen baseline. Detailed artifacts are under
`results/cnn_autoencoder/stride_30/` and
`results/model_comparison/fc_vs_cnn/`.

## Stage 11 — LSTM Autoencoder

### Decision and controlled purpose

Implement an LSTM Autoencoder as the third controlled stride-30 architecture
experiment. The FC Autoencoder remains stronger than the CNN on frozen anomaly
detection even though the CNN lowered normal-validation reconstruction MSE by
30.70%. Reconstruction quality alone therefore does not establish anomaly
separation. The LSTM tests whether explicit recurrent, order-dependent sequence
memory improves discrimination.

The customer split, Stage 1 preprocessing, train-fitted StandardScaler,
30-position window length, stride 30 manifests, `missing_count < 30` usable
rule, normal-only training, MSE loss, Adam optimizer, learning rate 0.001,
batch size 256, maximum 50 epochs, patience 5, minimum delta `1e-5`, seed 42,
and zero DataLoader workers remain fixed. Only the architecture changes.
Windows are not regenerated or rescaled.

### Exact architecture

The model consumes one scaled consumption value per timestep. The existing
architecture-agnostic `WindowDataset` returns `(B, 30)` and the LSTM training
pipeline reshapes it to `(B, 30, 1)`. `CONS_NO`, `FLAG`, missingness fields,
dates, and window IDs remain excluded from model input.

| Stage | Operation | Output per example |
|---|---|---|
| Input | 30 ordered timesteps, one value each | `(30, 1)` |
| Encoder | One-layer LSTM, input size 1, hidden size 16, `batch_first=True` | final hidden `(16)` |
| Latent | Linear `16 → 8`, ReLU | `(8)` |
| Decoder input | Repeat the latent vector for all 30 timesteps | `(30, 8)` |
| Decoder initial hidden | Linear `8 → 16`, ReLU | `(1, B, 16)` |
| Decoder initial cell | Deterministic zeros | `(1, B, 16)` |
| Decoder | One-layer LSTM, input size 8, hidden size 16, `batch_first=True` | `(30, 16)` |
| Output | Per-timestep Linear `16 → 1`, no output activation | `(30, 1)` |

No teacher forcing is used. The latent vector supplies every decoder timestep,
while its learned projection initializes decoder hidden state; decoder cell
state is always zero. The model has 3,177 trainable parameters. Latent dimension
8 matches the CNN bottleneck as a reasonable architecture-scale control and is
not the result of a sweep.

The FC model primarily treats the window as a flat vector, the CNN adds local
shared temporal filters, and the LSTM adds recurrent state and order-dependent
memory. This may represent sequential evolution, persistent trends, gradual
changes, or dependencies of later readings on earlier readings. No individual
hidden-state semantics are claimed without a separate investigation.

### Training and evaluation boundary

Full training will use all 708,308 usable stride-30 `FLAG=0` training windows.
Only the 152,246 usable `FLAG=0` validation windows may drive early stopping,
best-epoch selection, and checkpoint selection. Theft-labelled validation
windows are scored only after training with the reloaded best checkpoint.
The training script will save the checkpoint and config under
`models/lstm_autoencoder/`, with history, summary, curve, validation errors,
and validation-error plot under `results/lstm_autoencoder/stride_30/`.

The training stage does not select an anomaly threshold and never loads test
data. A later stage must freeze the 95th percentile of normal-validation LSTM
reconstruction error before one frozen test evaluation. The eventual controlled
comparison is FC versus CNN versus LSTM under the same evaluation metrics;
anomaly-detection performance, not reconstruction MSE alone, determines which
architecture is carried forward.

### Smoke-test and resource evidence

A synthetic `(256, 30, 1)` batch encoded to `(256, 8)` and reconstructed to
`(256, 30, 1)` with finite values. One real normal-only training batch loaded
as `(256, 30)`, reshaped to `(256, 30, 1)`, produced a finite MSE, finite
gradients for all 14 parameter tensors, and completed one Adam step. Its labels
were exclusively `FLAG=0`. Only the train split was loaded; validation and test
were not accessed. No epoch, checkpoint, threshold, or result artifact was
created by the smoke test.

The train split matrix occupies 117.0 MiB and one float32 256-window input batch
occupies 30 KiB before or after the shape-only view. There are approximately
2,767 training batches per epoch. Lazy window slicing avoids materializing a
duplicated dense window matrix. CPU memory is modest, but LSTM recurrence is
sequential across 30 timesteps and is expected to be slower per batch than the
FC and CNN baselines; batch size remains fixed rather than being increased.

Full LSTM training was executed manually by the user:

    python -m src.training.train_lstm_autoencoder --stride 30

The CPU run used 708,308 normal training windows and 152,246 normal validation
windows. It executed all predefined 50 epochs; early stopping did not trigger.
Epoch 50 is the best checkpoint with validation MSE `0.037662720912064844` and
final training MSE `0.0368094745261073`; runtime was 2,674.76 seconds. The
maximum remains fixed at 50 despite continued validation improvement, preserving
the pre-registered controlled architecture comparison. No threshold was selected
and test data was not used during training.

## Stage 12 — LSTM Frozen Evaluation and Architecture Comparison

Stage 12 verified the manual LSTM run, used only saved validation errors to
freeze the normal-validation P95 threshold, then completed frozen test inference
and a saved-artifact-only FC/CNN/LSTM comparison. No model was retrained or
extended, and no test result influenced the threshold.

### LSTM validation threshold

The LSTM validation normal/theft-labelled mean MSE values are `0.037663` /
`0.061816`, with medians `0.016616` / `0.019248`; score ranges overlap under
customer-level weak labels. Validation ROC-AUC is `0.559887` and PR-AUC is
`0.103233`. The frozen normal-validation P95 is `0.13370828330516815`, saved at
`models/lstm_autoencoder/lstm_ae_stride30_threshold.json`. Its SHA-256 is
`3205cef563253b89a9dcfca9f5df6758c847c1ad2110c7dd58f0fa6ffae3682b`; the
checkpoint SHA-256 is
`1e9ad7b9f778b240b9e2f26b64e68fe94241c99d16df81d319e618ed13927fa2`.
At the frozen P95, validation precision/recall/F1 are `0.138509` / `0.092065` /
`0.110609`, specificity/FPR are `0.949995` / `0.050005`, and the anomaly rate
is `5.3383%`. Best-F1 and best-balanced-accuracy candidates are diagnostic only.

### Frozen test and diagnostics

On 165,020 frozen test windows (151,985 normal and 13,035 theft-labelled), the
LSTM produced TP/TN/FP/FN `1,255/144,577/7,408/11,780`; precision `0.144869`,
recall `0.096279`, F1 `0.115679`, specificity `0.951258`, FPR `0.048742`,
balanced accuracy `0.523769`, ROC-AUC `0.557689`, PR-AUC `0.104403`, and anomaly
rate `5.2497%`. Validation-to-test behavior is broadly stable.

Customer mean/max/P95 MSE ROC-AUC values are `0.575256` / `0.564239` /
`0.580588`; corresponding PR-AUC values are `0.138774` / `0.120780` /
`0.148500`. No customer classifier or customer threshold was created. Overall
MSE/missing-fraction Pearson/Spearman correlation is `-0.012434` / `-0.047798`,
consistent with weak missingness association in FC/CNN analyses.

### Controlled architecture decision

All three models use the same split, preprocessing/scaler, 30-day stride-30
usable windows, normal-only training, MSE/Adam settings, learning rate 0.001,
batch 256, maximum 50 epochs, seed 42, normal-validation P95 selection, and
frozen-test methodology. Architecture is the primary changed factor.

| Model | Parameters | Best validation MSE | CPU seconds | Test ROC-AUC | Test PR-AUC | Test F1 | Customer P95 ROC-AUC / PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| FC | 1,362 | 0.043589 | 600.96 | **0.561688** | **0.114411** | **0.142676** | **0.599251 / 0.168575** |
| CNN | 2,209 | **0.030209** | 1,945.98 | 0.555689 | 0.104788 | 0.124173 | 0.579486 / 0.144033 |
| LSTM | 3,177 | 0.037663 | 2,674.76 | 0.557689 | 0.104403 | 0.115679 | 0.580588 / 0.148500 |

CNN reconstructs normal windows best but worsens detection. LSTM reconstruction
lies between FC and CNN, but it also does not improve window-level or
customer-level discrimination over FC and costs `4.45×` FC CPU runtime. All
three models are weak under customer-level `FLAG` supervision. FC remains the
selected reconstruction-model baseline because it leads every principal frozen
test and customer-ranking metric while being smallest and fastest. Detailed
artifacts are under `results/lstm_autoencoder/stride_30/` and
`results/model_comparison/fc_vs_cnn_vs_lstm/`.

The next research stage should investigate one new pre-registered modeling
hypothesis with the frozen FC baseline retained for comparison; this stage does
not implement a GAN, ensemble, longer sequences, or hyperparameter search.

## Stage 13 — Leakage-Safe Supervised Signal Diagnostic

Stage 13 preserves the Stage 1–3 preprocessing, authoritative customer split,
Stage 3C usable-window rule, trained reconstruction models, and all prior
thresholds and outputs. It asks whether the existing 30-value scaled consumption
windows contain supervised signal under customer-level `FLAG` labels. The only
model features are the 30 ordered consumption values. IDs, target, split fields,
window/date/index metadata, missingness information, and prior model outputs are
excluded. Logistic model standardization is fitted on training windows only.

The exact customer population is 42,367: 38,755 normal and 3,612 theft-labelled
(8.525503%). Train/validation/test customer counts are 29,656/6,355/6,356, with
zero overlap. Their usable stride-30 window counts are 768,923/165,541/165,020.
One theft-labelled training customer has no usable window after the previously
adopted Stage 3C rule. `FLAG` remains customer-level and is propagated to each
window; it does not confirm theft within a particular window.

Five fixed baselines use seed 42: dummy prior, unweighted/balanced logistic
regression, and unweighted/balanced HistGradientBoosting. No resampling,
synthetic augmentation, neural model, or hyperparameter search is used.
Threshold-dependent validation metrics use a per-model validation-F1-maximizing
threshold. Customer diagnostics aggregate usable-window probabilities by mean
and freeze a separate validation-F1-maximizing threshold.

Validation-only selection uses highest window ROC-AUC then average precision and
chooses balanced HistGradientBoosting. Its validation window ROC-AUC/average
precision/F1 are `0.662847`/`0.203618`/`0.243040`; mean-probability customer
values are `0.712044`/`0.222600`/`0.275208`. The model hash and window/customer
thresholds `0.608718870015863`/`0.5251327815829091` were frozen before test
inference. Aggregate test counts had been inspected solely for the requested
split audit; no test feature matrix or per-window label was loaded before the
freeze.

Following the one-time frozen-test policy established in Stages 6, 8, 10, and
12, only the selected supervised model was evaluated on test. Window
ROC-AUC/average precision/F1 are `0.657693`/`0.205305`/`0.257316`; customer-mean
values are `0.702314`/`0.219317`/`0.280340`. Class weighting changes tree window
ROC-AUC by only `+0.001154` and does not materially create the signal.

The diagnostic supports a controlled supervised or hybrid research track: the
selected validation window ROC-AUC is `0.098536` above the FC reconstruction
validation ROC-AUC. Absolute precision/recall remain modest, and weak labels
limit window-level interpretation. The recommended next stage is a
pre-registered label/window-formulation and compact consumption-feature study,
potentially followed by a validation-only test of whether the frozen FC score
adds incremental supervised value. Stage 13 does not claim causal detection or
deployment readiness. Full evidence is under `results/supervised/stage_13/`.

## Stage 14 — Behavioural Representation Analysis

Stage 14 chooses representation analysis instead of another neural network or
GAN because Stage 13 already established moderate supervised signal while the
earlier FC/CNN/LSTM sequence showed that added reconstruction complexity did not
improve discrimination. From Stage 14 onward, test is closed for development.
This stage uses train and validation only.

Before validation scoring, Stage 14 pre-registered 17 deterministic features
derived only from each existing 30-value scaled consumption window: `mean`,
`median`, `minimum`, `maximum`, `q10`, `q90`, `standard_deviation`, `range`,
`interquartile_range`, `mean_absolute_deviation`, `first_to_last_change`,
`absolute_first_to_last_change`, `mean_absolute_consecutive_change`,
`maximum_absolute_consecutive_change`, `linear_slope`,
`fraction_below_minus_one`, and `second_half_minus_first_half_mean`.

The compared representations are raw 30 values, summaries only, and their
47-column concatenation. Each uses the fixed Stage 13 logistic regression and
HistGradientBoosting configurations with unweighted and balanced variants. IDs,
labels, split/date/window metadata, missingness information, earlier predictions
and reconstruction outputs, and all test information are excluded. Logistic
standardization is fitted on training windows only. All engineered train and
validation values are finite, no feature is constant, and no clipping or new
transformation is introduced.

All raw-only validation metrics reproduce Stage 13 within `1e-12`. Summary
representations improve customer ROC-AUC and average precision across both model
families and both weight settings. Unweighted HGB improves from raw-only
`0.717956`/`0.240969` to summary-only `0.738697`/`0.261320`; combined is slightly
lower at `0.737707`/`0.260590`. Balanced HGB shows a similar raw-to-summary gain
from `0.712044`/`0.222600` to `0.733311`/`0.242302`. Logistic gains are also
consistent. Therefore summary-only unweighted HGB becomes the validation-frozen
supervised baseline for the next research stage.

Validation permutation importance for the selected tree is led by `maximum`,
`mean_absolute_consecutive_change`, `median`, `minimum`, and
`maximum_absolute_consecutive_change`. The apparent signal is primarily level
and local variability/jumps; trend, half-window shift, and relative-low fraction
are weak conditional contributors. Correlation among summaries and possible
customer/load confounding prevent causal theft interpretations.

Stage 14 did not load test features, labels, or predictions and did not use the
historical Stage 13 test result for selection. The recommended Stage 15 question
is whether frozen FC autoencoder-derived signals add incremental validation
information beyond summary-only HGB. Stage 15 is not executed. Full evidence is
under `results/supervised/stage_14/`.

## Stage 15 — Frozen-FC Hybrid Ablation

Stage 15 tests the selected stride-30 FC autoencoder as a complementary
representation rather than treating reconstruction as the final anomaly
detector. The unchanged checkpoint is
`models/autoencoder/ae_stride30_baseline.pt`, SHA-256
`421f1e2cdf96da1bd3231cd0baa750746d89757700d63ac428f9167f69237fa4`.
It is the 1,362-parameter `30→16→8→4→8→16→30` model trained on 708,308
normal training windows and selected at epoch 30 using normal-validation MSE.
The latter creates inherited validation reuse for all FC-derived Stage 15
features; results are diagnostic and may be optimistic. No FC retraining or
adaptation occurs, and the validation-derived P95 threshold is not a feature.

Before scoring, ten FC-derived features were pre-registered: reconstruction
MSE, MAE, maximum absolute error, early-half MSE, late-half MSE,
late-minus-early MSE, and the four frozen ReLU bottleneck coordinates. The
controlled unweighted HGB ablation compares exact Stage 14 summaries,
reconstruction-only, all FC features, summaries plus reconstruction, and
summaries plus all FC features. Extraction is deterministic CPU inference with
gradients disabled and no label input. All 165,541 validation window IDs,
customer IDs, labels, and computed MSE values align with saved Stage 4 output;
maximum MSE difference is exactly zero. Stage 14 summary-only metrics reproduce
within `1e-12`.

Customer ROC-AUC/AP/F1 are `0.738697`/`0.261320`/`0.303249` for summary-only,
`0.663819`/`0.180220`/`0.238210` for reconstruction-only,
`0.714750`/`0.240460`/`0.288013` for all FC features, and
`0.738454`/`0.263043`/`0.301744` for the complete hybrid. Thus full-hybrid
deltas are `-0.000243` ROC-AUC, `+0.001723` AP, and `-0.001506` F1. A seeded
2,000-replicate paired customer bootstrap gives 95% intervals
`[-0.001825, 0.001353]` for ROC-AUC delta and
`[-0.000485, 0.003987]` for AP delta.

FC features have small conditional importance in the hybrid and substantial
redundancy with Stage 14 summaries: reconstruction MSE, maximum error, and MAE
have Spearman correlations around `0.946–0.962` with variability/change
summaries, while latent coordinate 3 correlates `0.972` with mean. The outcome
is **C — no complementary value**. The isolated microscopic AP increase does
not justify frozen neural inference, especially with flat/worse ROC-AUC and F1
and inherited validation reuse. The FC branch is rejected from the current AMP
detector; Stage 14 summary-only unweighted HGB remains frozen.

For paper relevance, FC/CNN/LSTM experiments remain controlled evidence that
reconstruction anomaly modelling was investigated and rejected for lack of
incremental detection value. The recommended Stage 16 question is whether a
direct customer-level model using compact temporal aggregates of the frozen 17
window summaries improves over mean aggregation of window probabilities. Stage
16 is not executed. **TEST NOT ACCESSED DURING STAGE 15.** Full evidence is
under `results/supervised/stage_15/`.
