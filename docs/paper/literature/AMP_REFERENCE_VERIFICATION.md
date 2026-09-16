# AMP-LDP reference verification register

## Verification method

Stage 29 used public bibliographic and publisher information only. No AMP cohort rows, prediction files, or model artifacts were opened. Each admitted record passed the following gates:

1. Crossref DOI lookup returned a journal article with matching title, authors, journal, and year.
2. The DOI resolved to a publisher or journal landing page.
3. Scopus or Web of Science coverage was substantiated at journal level through a publisher indexing page or a Scopus-derived SCImago journal record.
4. The item was not a conference paper, book, chapter, preprint, editorial, or non-peer-reviewed page.
5. Protocol details included below are limited to what the article landing page/full text documents; “not established” is used instead of inference.

Verification date: **2026-09-16**.

## Journal indexing evidence

| Journal(s) | Admitted refs | Substantiation | Status |
|---|---:|---|---|
| EURASIP Journal on Wireless Communications and Networking | R01 | Springer journal landing page and Scopus-derived journal coverage record | Scopus substantiated |
| Energy Science & Engineering | R02 | [Wiley abstracting/indexing information](https://onlinelibrary.wiley.com/page/journal/20500505/homepage/productinformation.html) lists Web of Science and SCIE | WoS substantiated |
| Frontiers in Energy Research | R03, R09 | [Frontiers journal facts](https://www.frontiersin.org/journals/energy-research/about) lists Scopus | Scopus substantiated; WoS status is not relied upon |
| Energies | R04, R19 | [MDPI indexing page](https://www.mdpi.com/journal/energies/indexing) lists Scopus and SCIE | Scopus and WoS substantiated |
| Sensors | R15 | [MDPI indexing page](https://www.mdpi.com/journal/sensors/indexing) lists Scopus and SCIE | Scopus and WoS substantiated |
| Scientific Reports | R11 | [Nature journal information](https://www.nature.com/srep/about) lists Scopus and Web of Science | Scopus and WoS substantiated |
| PLOS ONE | R23 | [PLOS journal information](https://journals.plos.org/plosone/s/journal-information) lists Scopus and Web of Science | Scopus and WoS substantiated |
| IEEE Access; IEEE Transactions on Industrial Informatics; IEEE Transactions on Smart Grid; IEEE Transactions on Knowledge and Data Engineering | R05, R17, R18, R24 | IEEE journal landing pages plus Scopus-derived SCImago journal records | Scopus substantiated |
| Expert Systems with Applications; Applied Energy; Heliyon; Results in Engineering; Engineering Applications of Artificial Intelligence; Electric Power Systems Research; Pattern Recognition | R06–R08, R10, R12, R14, R16, R20, R21 | Elsevier DOI landing pages plus Scopus-derived SCImago journal records; the records identify each as a journal with current or historical Scopus coverage | Scopus substantiated |
| AIP Advances | R13 | AIP journal landing page plus Scopus-derived journal coverage record | Scopus substantiated |
| Data Mining and Knowledge Discovery | R22 | Springer journal landing page plus Scopus-derived journal coverage record | Scopus substantiated |

The publisher pages above are preferred evidence where they explicitly expose indexing. SCImago is used only as a journal-level Scopus-coverage substantiation source; it is not used to verify article metadata or experimental results.

## Article-level scope and protocol register

| Ref | Dataset / label granularity documented | Method and evaluation information verified | AMP-LDP relevance | Verification status |
|---|---|---|---|---|
| R01 | SGCC; customer histories with customer-level normal/theft targets | Supervised contrastive encoder with LTTB augmentation; AUC and recall; customer is prediction unit | Learned temporal representation under customer supervision | Verified |
| R02 | SGCC; 42,372 customer histories, 3,615 theft labels stated; 80/20 split | Cost-sensitive GCAE-LSTM with salp-swarm tuning; accuracy, precision, recall, F1, AUC | Reconstruction/encoding, imbalance, and protocol sensitivity | Verified |
| R03 | CNGC/SGCC consumption histories; customer label | AlexNet-based supervised classification; undersampling and tuned classifier; accuracy, precision, recall, F1, AUC | Deep full-history baseline family | Verified |
| R04 | SGCC daily customer consumption with customer label | Dual-time feature fusion; reported accuracy 0.947, precision 0.932, recall 0.964, F1 0.948, AUC 0.986 under the paper’s split | Multi-scale temporal representation; not protocol-matched to AMP | Verified |
| R05 | SGCC customer histories and labels | Dynamic generative residual GCN; classification metrics under paper-specific preprocessing/split | Graph-temporal supervised alternative | Verified |
| R06 | SGCC customer histories and labels | AI/deep-learning pipeline with imbalance handling; classification metrics under paper-specific partitioning | Recent SGCC supervised benchmark context | Verified |
| R07 | SGCC customer histories and labels | Multi-objective evolutionary hybrid deep learning; customer classification metrics | Optimization-heavy supervised alternative | Verified |
| R08 | Regional electricity-theft data; source-to-new-region transfer setting | Two-stage generalizable approach; region-shift evaluation and classification metrics | Highlights external-region generalization not tested by AMP | Verified |
| R09 | CNGC/SGCC customer histories; customer label; three train/test ratios | K-means SMOTE plus MLP-GRU; precision, recall, F1, accuracy, MCC, ROC-AUC and PR-AUC | Explicit imbalance and split-sensitivity example | Verified |
| R10 | Non-smart-grid billing/consumption setting; customer theft target | Deep supervised theft prediction; article-specific metrics and split | Broadens application setting but is not SGCC-comparable | Verified |
| R11 | SGCC customer histories and customer label | Deep hybrid classifier; reports accuracy and other classification metrics under its own protocol | Recent SGCC deep-learning context | Verified |
| R12 | Temporal electricity sequences; customer theft target | Transfer learning with CNN-LSTM and BiGRU-BiLSTM; reported ROC-AUC up to 0.89 in the article protocol | Transfer-based temporal representation | Verified |
| R13 | Electricity-consumption theft classification; customer target | Bayesian-optimized TCN with explainability; reported accuracy 98.04% under its own protocol | Temporal convolution and explainability context | Verified |
| R14 | SGCC and Smart* experiments; customer/household evaluation labels | Self-decision ant-colony clustering; classification assessment on two datasets | Non-neural clustering control family | Verified |
| R15 | Consumption-pattern data with injected/constructed attack patterns | Hybrid CNN and generative data treatment; accuracy, precision, recall, F1 and ROC analysis | Shows synthetic-attack protocols differ from inspected SGCC labels | Verified |
| R16 | SGCC consumption histories and customer targets | Genetic-algorithm feature/parameter optimization with supervised classification metrics | Pre-2023 SGCC optimization baseline | Verified |
| R17 | SGCC customer histories and labels | Wide-and-deep CNN combining one- and two-dimensional views; AUC-centered evaluation | Foundational SGCC temporal representation | Verified |
| R18 | Utility smart-meter data with inspected non-technical-loss targets | Supervised learning and engineered features; customer-level evaluation | Established engineered-profile comparator context | Verified |
| R19 | SGCC customer histories and labels | CNN-LSTM with SMOTE; accuracy, precision, recall, F1 and AUC | Foundational deep temporal SGCC baseline | Verified |
| R20 | Multiple NTL datasets/protocols reviewed | Survey of NTL detection families and evaluation limitations | Domain taxonomy and comparability caution | Verified |
| R21 | Cross-domain MIL literature | Survey of bag assumptions, label ambiguity, and aggregation | Formal weak-supervision/MIL framing | Verified |
| R22 | Time-series classification literature | Review of deep temporal architectures and evaluation | Representation-learning context | Verified |
| R23 | Simulated and empirical imbalanced-class examples | ROC versus precision–recall analysis | Justifies AP/PR emphasis at low prevalence | Verified |
| R24 | Cross-domain imbalanced-learning literature | Survey of data-, algorithm-, and evaluation-level remedies | Imbalance methodology boundary | Verified |

## Unresolved or excluded candidates

| Candidate | Reason not admitted | Required resolution |
|---|---|---|
| “STL-Net” numerical comparison previously mentioned in project notes | The original journal publication and its exact evaluation protocol were not located in tracked repository documentation or independently verified during Stage 29 | Locate the original peer-reviewed journal article, resolve its DOI, confirm indexing, and inspect the exact dataset/split/metric protocol before citing any number |
| “Energy Theft Detection Using Gradient Boosting Theft Detector With Feature Engineering-Based Preprocessing” (Punmiya and Choe) | The candidate DOI supplied by a secondary trace resolved to no Crossref record; bibliographic identity was therefore not admitted | Establish the correct DOI from the publisher record and repeat all admission checks |
| Search-result-only articles without accessible protocol detail | Metadata alone was insufficient for the requested dataset/label/protocol audit | Inspect the publisher full text or accepted manuscript and verify indexing before admission |

No unresolved candidate appears in the BibTeX file or in paper-facing quantitative comparisons.
