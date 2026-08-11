# Stage 3B — Missingness Quality Analysis

## Scope

This analysis evaluates the canonical stride-30 manifests without filtering or
rewriting them. A threshold of 0.25 means retain windows with
missing_fraction <= 0.25. Comparisons use integer counts out of 30 to avoid
floating-point boundary ambiguity. The special <1.00 rule retains every window
except those with 30/30 originally missing positions.

## Threshold comparison

| Rule | Retained windows | Retained % | Theft % before | Theft % after | Customers losing all | Theft customers losing all |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 898,396 | 62.37% | 8.526% | 7.925% | 57 | 8 |
| 0.10 | 1,019,380 | 70.77% | 8.526% | 7.831% | 26 | 5 |
| 0.25 | 1,050,855 | 72.95% | 8.526% | 7.789% | 22 | 4 |
| 0.50 | 1,086,896 | 75.45% | 8.526% | 7.846% | 3 | 2 |
| 0.75 | 1,093,814 | 75.93% | 8.526% | 7.875% | 3 | 2 |
| <1.00 (remove fully missing only) | 1,099,484 | 76.33% | 8.526% | 7.908% | 1 | 1 |

## Fully missing windows

- Fully missing normal-customer windows: 305,131
  (23.16% of normal windows).
- Fully missing theft-customer windows: 35,863
  (29.20% of theft-customer windows).
- Customers with at least one fully missing window:
  23,110.
- Customers whose all 34 windows are fully missing:
  1.
- Longest consecutive fully missing run: 34
  windows.

## Missingness by customer class

| FLAG | Mean | Median | P90 | Fully missing | Zero missing |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2542 | 0.0000 | 1.0000 | 23.16% | 62.78% |
| 1 | 0.3180 | 0.0000 | 1.0000 | 29.20% | 57.97% |

FLAG is customer-level weak supervision. The class comparison therefore does
not establish whether any particular positive window contains theft.

## Temporal concentration

The five window positions with the highest average missingness are:

| Position | Dates | Mean missing | Fully missing | Zero missing |
|---:|---|---:|---:|---:|
| 2 | 3/2/2014 to 3/31/2014 | 0.4322 | 37.23% | 32.54% |
| 23 | 11/23/2015 to 12/22/2015 | 0.4262 | 41.69% | 53.56% |
| 3 | 4/1/2014 to 4/30/2014 | 0.4148 | 37.40% | 38.29% |
| 22 | 10/24/2015 to 11/22/2015 | 0.4141 | 40.67% | 54.95% |
| 1 | 1/31/2014 to 3/1/2014 | 0.4140 | 37.98% | 39.91% |

## Longest fully missing runs

| CONS_NO | FLAG | Split | Fully missing windows | Maximum consecutive run |
|---|---:|---|---:|---:|
| B08E0670C8A6933E5EC44715F7A80E9D | 1 | train | 34 | 34 |
| 040A61A4680D6D770CCAF0213944B193 | 0 | val | 33 | 33 |
| 0E5FF6F545323C7382522290DCEB0DF7 | 0 | train | 33 | 33 |
| 145A9E0323B3443A046B427EBF7AF7A7 | 1 | train | 33 | 33 |
| 2E175D02BE79895286938D217E36A7A1 | 0 | train | 33 | 33 |
| 3258DA14AC82C15A98B4A2A5FEB280FB | 0 | test | 33 | 33 |
| 38779BE2A88ABCEDBD45879004D1F6FC | 0 | val | 33 | 33 |
| 6280D9197D402FA5F612C6785AEA36FA | 0 | train | 33 | 33 |
| 65B7C4532F9B092EBB992E80689FD64F | 0 | train | 33 | 33 |
| 6ECDD200AAFC9CFBB5BA4516CAAA9864 | 0 | val | 33 | 33 |

## Evidence-based recommendation

### A. Minimum obvious rule

**Proposed for review:** remove only fully originally missing windows
(missing_fraction < 1.0). Such windows contain no directly observed consumption
information even though Stage 1 imputation supplied numerical values. This
rule would retain 1,099,484 windows
(76.33%) and remove
340,994. It would cause
1 customers—including
1 theft-labelled
customers—to lose all windows. That customer-level effect must be reviewed
before the rule becomes canonical.

### B. More aggressive rules

Thresholds of <=0.75, <=0.50, <=0.25, <=0.10, and 0% increasingly reduce
dependence on imputation, but they discard more windows and may remove every
window for additional customers. Their actual retention, class composition,
and customer loss are shown above and in the CSV outputs.

The descriptive results do **not** establish an objectively optimal aggressive
threshold. No canonical filtering decision is made here. The minimum rule is a
methodologically clear proposal because 30/30-missing windows contain zero
observed readings; more aggressive filtering should be selected only after
reviewing customer coverage, class distortion, and later sensitivity analysis.
