# AMP-LDP literature comparison

## Reading rule

The table compares research designs, not leaderboard positions. “SGCC” does not imply common customer membership, preprocessing, balancing, split, prevalence, tuning budget, or metric implementation. Exact point estimates are included only where the primary article exposed them clearly; otherwise the metric family and protocol distinction are reported.

## Recent journal articles (2023–2026)

| Ref | Dataset and label granularity | Method | Supervision | Reported metrics / protocol | Principal limitation for comparison | Relevance to AMP-LDP |
|---|---|---|---|---|---|---|
| R01 (2023) | SGCC; one label per customer history | Supervised contrastive encoder with LTTB augmentation | Customer-level supervised contrastive + classification loss | AUC and recall on a paper-specific SGCC partition | No matched direct-distribution control; split is not AMP-governed | Confirms value of learned customer representations |
| R02 (2023) | SGCC; 42,372 histories and 3,615 theft customers stated | Cost-sensitive GCAE + LSTM, salp-swarm tuning | Customer-level supervised; class weighting | 80/20 split; precision 99.45%, F1 95.93%, accuracy 92.25%, AUC 71.13% as reported | Strong tuning and unusual divergence among metrics; not protocol-equivalent | Reconstruction/encoding and imbalance comparator |
| R03 (2023) | CNGC/SGCC; customer theft label | AlexNet classifier with imbalance treatment | Customer-level supervised | Accuracy, precision, recall, F1 and AUC on article-specific split | Full-history classifier; sampling and tuning differ | Deep full-history baseline family |
| R04 (2024) | SGCC daily histories; customer label | Dual-time feature fusion and deep classifier | Customer-level supervised | Accuracy 0.947, precision 0.932, recall 0.964, F1 0.948, AUC 0.986 under paper protocol | No AP and no governed AMP split; fusion differs from window-distribution question | Multi-scale temporal representation context |
| R05 (2024) | SGCC customer histories | Dynamic generative residual GCN | Customer-level supervised | Classification metrics under article preprocessing/split | Graph construction and generative augmentation confound representation comparison | Graph-temporal alternative |
| R06 (2024) | SGCC customer histories | Data-driven deep-learning pipeline with imbalance handling | Customer-level supervised | Accuracy/precision/recall/F1/AUC family under paper-specific partition | Optimization/preprocessing not matched; headline scores not portable | Recent supervised SGCC context |
| R07 (2024) | SGCC customer histories | Multi-objective evolutionary hybrid deep learning | Customer-level supervised | Classification metrics under internally optimized protocol | Search budget and objectives differ; no frozen external cohort | Optimization-heavy supervised alternative |
| R08 (2024) | Source and new-region theft datasets | Two-stage generalizable detector | Supervised domain/region transfer | Cross-region classification assessment | Different scientific target and datasets; AMP has no external-region test | Defines external-generalization evidence AMP lacks |
| R09 (2024) | CNGC/SGCC; customer label; three split ratios | K-means SMOTE + MLP-GRU | Customer-level supervised with synthetic balancing | Precision, recall, F1, accuracy, MCC, ROC-AUC, PR-AUC; 50/50, 75/25 and 90/10 splits; best displayed AUC 0.95 | Multiple split ratios and synthetic balancing; no frozen cohort identities | Illustrates prevalence/split sensitivity and PR reporting |
| R10 (2024) | Non-smart-grid billing/consumption data; customer target | Deep supervised theft predictor | Customer-level supervised | Classification metrics under non-SGCC protocol | Different metering environment and feature availability | Shows setting-specificity of ETD claims |
| R11 (2025) | SGCC customer histories | Hybrid deep classifier | Customer-level supervised | Reports accuracy up to 94.7% plus classification metrics under article protocol | No protocol match to AMP; point metrics cannot be transferred | Recent SGCC deep-learning context |
| R12 (2025) | Temporal electricity sequences; customer target | Transfer-learned CNN-LSTM and BiGRU-BiLSTM | Customer-level supervised transfer learning | Precision/recall/F1/accuracy/ROC-AUC; reported AUC up to 0.89 | Transfer source and target construction differ from AMP | Temporal representation-transfer context |
| R13 (2026) | Electricity-consumption theft data; customer target | Bayesian-optimized TCN with explainability | Customer-level supervised | Reported accuracy 98.04% under its own protocol | Protocol and cohort not AMP-equivalent; accuracy is prevalence-sensitive | Temporal convolution and explainability context |
| R14 (2024) | SGCC and Smart* household data | Self-decision ant-colony clustering | Unsupervised clustering followed by labeled assessment | Dataset-specific classification metrics | Two datasets and clustering objective differ from customer probability modelling | Non-neural representation/control family |
| R15 (2024) | Consumption patterns with constructed/injected attacks | Hybrid CNN with generative treatment | Supervised on constructed attack labels | Accuracy, precision, recall, F1 and ROC analysis | Synthetic attacks are not equivalent to inspected SGCC customer labels | Warns against merging natural-label and synthetic-attack results |

## Foundational and methodological support

| Ref | Role in report | Specific use |
|---|---|---|
| R16–R19 | Earlier electricity-theft baselines | Establish engineered, CNN, CNN-LSTM, wide/deep, and optimization-based customer-history detection families |
| R20 | Domain review | Supports the NTL taxonomy and warns that datasets/protocols vary |
| R21 | Weak-supervision foundation | Defines bags, instance ambiguity, and MIL assumptions |
| R22 | Temporal representation foundation | Situates deep learning for time-series classification |
| R23 | Evaluation foundation | Supports precision–recall/AP emphasis for imbalanced outcomes |
| R24 | Imbalance foundation | Distinguishes resampling, algorithm-level, and evaluation-level responses to class imbalance |

## Comparison conclusion

The recent literature demonstrates broad success with customer-supervised temporal models, but it does not provide a protocol-equivalent leaderboard for AMP. The defensible comparison is structural: AMP-LDP retains the customer as the supervised unit, treats windows as unlabeled instances, preserves their latent distribution, and isolates learned encoding with an identical-statistics DIRECT control. Its reported numerical comparison remains internal to the governed AMP cohorts.
