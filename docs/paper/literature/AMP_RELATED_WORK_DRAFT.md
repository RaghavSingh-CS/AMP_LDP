# Related work draft

## Electricity-theft detection from smart-meter histories

Data-driven non-technical-loss detection has progressed from engineered customer profiles and conventional supervised learners to temporal deep networks. Messinis and Hatziargyriou organize the broader NTL detection landscape and emphasize the importance of data and evaluation design [R20]. On utility smart-meter records, Buzau et al. demonstrate the continued strength of supervised engineered features [R18]. SGCC studies introduced wide-and-deep temporal views [R17], CNN-LSTM modelling [R19], and genetic-algorithm-assisted learning [R16]. These works motivate customer-level temporal modelling, but their results also illustrate why paper-specific preprocessing and splits must remain attached to reported metrics.

Recent journal work expands the model family substantially. Liu et al. use supervised contrastive learning to compare customer representations [R01]; Pamir et al. combine a convolutional autoencoder, cost-sensitive LSTM classification, and metaheuristic tuning [R02]; and Khan et al. use an AlexNet-derived detector [R03]. Other recent studies use dual-time feature fusion [R04], dynamic graph convolution [R05], deep data-driven pipelines [R06], multi-objective evolutionary search [R07], and explicit cross-region transfer [R08]. MLP-GRU modelling with synthetic balancing highlights sensitivity to train/test ratios and metric choice [R09], while non-smart-grid prediction shows that feature and acquisition settings can differ materially [R10]. More recent hybrid and transfer models continue this direction [R11, R12], and temporal convolution with explainability adds a model-audit perspective [R13]. Clustering [R14] and constructed-attack CNN protocols [R15] broaden the methodological landscape but are not directly interchangeable with inspected customer-label experiments.

## Label granularity, weak supervision, and temporal representation

The SGCC FLAG used by AMP is a customer-level label; it does not identify theft days or theft windows. Multiple-instance learning provides the relevant conceptual distinction between a labeled bag and ambiguous instances [R21]. Applying a customer label to every window would create propagated weak labels, not verified temporal ground truth. AMP therefore uses windows as unlabeled instances and applies the supervised loss only after customer aggregation.

Deep time-series research shows that learned temporal representations can capture local and long-range structure using convolutional, recurrent, and hybrid architectures [R22]. Electricity-theft studies likewise learn from entire histories or transformed temporal views [R01, R04, R05, R12, R17, R19]. The narrower AMP question is not whether temporal encoders are useful in general, but whether learned window encodings add information after the within-customer distribution is preserved. That question requires a matched aggregation control, because improvements could otherwise arise from the distribution summaries alone.

## Distribution-preserving customer profiling

AMP-LDP encodes each 30-day window and aggregates six statistics—mean, stabilized population standard deviation, 10th percentile, median, 90th percentile, and maximum—per latent dimension. The DIRECT control computes the same statistics on the handcrafted window summaries. This design separates the contribution of learned encoding from the contribution of distribution preservation. Within the verified bibliography, no article was found that combines customer-only labels, unlabeled windows, matched direct-versus-latent distribution aggregation, and frozen replication across three governed cohorts. This observation is limited to the reviewed set and is not a universal priority claim.

## Imbalance and evaluation

Electricity-theft labels are typically rare, and recent studies address imbalance through class weighting, undersampling, SMOTE variants, or generative procedures [R02, R03, R05, R09, R15, R19]. He and Garcia distinguish data-level and algorithm-level imbalance remedies and the need for suitable assessment [R24]. Saito and Rehmsmeier show why precision–recall analysis is more informative than ROC alone for imbalanced binary outcomes [R23]. AMP therefore reports both ROC-AUC and Average Precision at the customer level and uses paired customer bootstrap intervals for comparisons. Accuracy is not treated as a primary metric.

## Positioning of AMP-LDP

AMP-LDP is positioned as a label-granularity-aware customer classifier, not a theft-window detector. Its evidence concerns customer-level discrimination on one public SGCC-derived dataset across governed development, confirmation, and final-evaluation cohorts. The final cohort has disclosed historical baseline access. Consequently, the report may discuss reproducibility across those governed cohorts and improvement over the matched DIRECT control, but must not claim temporal localization, external utility generalization, causal identification, or state-of-the-art superiority.

## Citation completeness

All 24 Stage 29 references are cited in this draft: R01–R15 as recent domain literature, R16–R20 as earlier domain evidence, and R21–R24 as methodological support.
