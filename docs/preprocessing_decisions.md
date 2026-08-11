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

