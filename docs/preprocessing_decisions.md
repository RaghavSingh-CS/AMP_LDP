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
