# AMP manuscript claim matrix

This matrix is the paper-facing guardrail. Wording may be made shorter, but not
stronger, without new evidence outside the closed studies.

| Claim | Evidence | Allowed wording | Forbidden overreach |
|---|---|---|---|
| Supervised customer signal exists | Stage 13 customer-mean ROC/AP `0.7023/0.2193`, above reconstruction results | “Customer labels contain usable discrimination signal under the evaluated SGCC protocol.” | Verified theft-event detection; deployment performance |
| Reconstruction anomaly is weak | FC/CNN/LSTM final window ROC near `0.56` and AP near `0.10–0.11` | “Reconstruction anomaly scoring was weak under the evaluated formulations.” | All anomaly detection is ineffective |
| Customer aggregation matters | Stage 16 materially improved over historical window scoring; Stage 24B single-vector MIL remained below FULL_48 | “Customer-level aggregation and retained distribution information matter in this setting.” | One isolated causal mechanism explains every gain |
| Behavioural summaries help | Stage 14 summary-only HGB improved over raw-only | “Behavioural summaries improved historical validation discrimination.” | The 17 summaries are universally optimal |
| Single-vector MIL is insufficient under the evaluated setup | Mean/max/attention MIL remained materially below FULL_48 on DEV_VAL | “The evaluated single-vector MIL pools were insufficient.” | Attention or MIL generally cannot work |
| Distribution preservation improves representation | DIRECT and LATENT materially exceed single-vector MIL, with design caveat | “Multi-statistic customer aggregation recovered information lost by one-vector pooling.” | A perfectly controlled causal comparison between every table row |
| Learned encoding adds value over DIRECT | Positive AMP-LDP−DIRECT effects in DEV_VAL, CONFIRMATION, and final; final paired CIs above zero | “Learning window representations before identical distribution aggregation improved customer-level discrimination.” | Learned windows localize theft or represent causal theft behaviour |
| Effect persists across governed cohorts | Both metrics positive against DIRECT in all three roles; final frozen evaluation supports the effect | “The qualitative learned-encoding advantage persisted across development, confirmation, and governed final evaluation.” | External validation or universal generalization |
| AMP-LDP compares favourably with FULL_48 | Parity in development; higher confirmation/final point metrics; final paired CIs positive | “AMP-LDP was competitive with or improved on the frozen FULL_48 comparator under the Study 2 protocol.” | Substitute Study 1's refitted FULL_48 result into the paired Study 2 comparison |
| FULL_48 is not the same model in both studies | Stage 23B refit used DEV_TRAIN+DEV_VAL; Study 2 comparator remained Stage 17B DEV_TRAIN-fitted | State the training population whenever quoting a final FULL_48 result | Present `0.7925/0.3702` and `0.7979/0.3853` as conflicting estimates of one identical checkpoint |
| Robustness has limits | Stage 18 consumption strata varied; four of five final seeds met both parity margins | “Results were seed-stable by preregistered rules, with subgroup limitations.” | Equal performance for all customer subgroups |
| Temporal localization is not established | SGCC `FLAG` has no verified theft timing; no instance-level theft loss or validation | “Windows are unlabeled instances used to build a customer prediction.” | Theft-window detection, theft-period localization, individual-window theft probability |
| External generalization is not established | One public SGCC source; no external utility cohort | “Evidence is internal to the governed SGCC cohorts.” | Generalizes to other utilities, countries, meters, or deployment settings |
| Literature superiority is not established | Heterogeneous SGCC protocols; no protocol-equivalence audit | “Published SGCC values provide context only.” | State of the art or direct superiority based on incomparable splits/prevalence |
| Final evaluation has a governance limitation | Historical Stage 13 access is disclosed; Study 2 was later frozen without its row-level access | Use the exact Stage 27A disclosure and governed-cohort terminology | Pristine, untouched, unseen, never-seen test set |
| No causal interpretation | All results are observational predictive comparisons | “The results support a representational interpretation.” | Learning identifies causes or mechanisms of theft behaviour |

## Frozen anchor

> Learning window representations before distribution-preserving customer aggregation adds reproducible customer-level discrimination beyond identical direct aggregation across development, confirmation, and the governed final evaluation.
