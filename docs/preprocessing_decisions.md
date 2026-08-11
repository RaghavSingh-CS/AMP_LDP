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
