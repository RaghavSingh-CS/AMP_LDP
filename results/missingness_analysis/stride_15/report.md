# Stage 3B — Missingness Quality Analysis

## Scope

This analysis evaluates the canonical stride_15 manifests without filtering or
rewriting them. A threshold of 0.25 means retain windows with
missing_fraction <= 0.25. Comparisons use integer counts out of 30 to avoid
floating-point boundary ambiguity. The special <1.00 rule retains every window
except those with 30/30 originally missing positions.

## Threshold comparison

| Rule | Retained windows | Retained % | Theft % before | Theft % after | Customers losing all | Theft customers losing all |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 1,777,632 | 62.62% | 8.526% | 7.947% | 51 | 8 |
| 0.10 | 2,016,319 | 71.03% | 8.526% | 7.816% | 25 | 5 |
| 0.25 | 2,076,117 | 73.14% | 8.526% | 7.792% | 21 | 4 |
| 0.50 | 2,128,869 | 75.00% | 8.526% | 7.831% | 3 | 2 |
| 0.75 | 2,144,729 | 75.56% | 8.526% | 7.859% | 3 | 2 |
| <1.00 (remove fully missing only) | 2,174,699 | 76.61% | 8.526% | 7.909% | 1 | 1 |

## Fully missing windows

- Fully missing normal-customer windows: 593,887
  (22.87% of normal windows).
- Fully missing theft-customer windows: 70,003
  (28.93% of theft-customer windows).
- Customers with at least one fully missing window:
  23,219.
- Customers whose all 67 windows are fully missing:
  1.
- Longest consecutive fully missing run: 67
  windows.

## Missingness by customer class

| FLAG | Mean | Median | P90 | Fully missing | Zero missing |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2550 | 0.0000 | 1.0000 | 22.87% | 63.02% |
| 1 | 0.3195 | 0.0000 | 1.0000 | 28.93% | 58.38% |

FLAG is customer-level weak supervision. The class comparison therefore does
not establish whether any particular positive window contains theft.

## Temporal concentration

The five window positions with the highest average missingness are:

| Position | Dates | Mean missing | Fully missing | Zero missing |
|---:|---|---:|---:|---:|
| 3 | 2/15/2014 to 3/16/2014 | 0.4402 | 37.49% | 34.73% |
| 4 | 3/2/2014 to 3/31/2014 | 0.4322 | 37.23% | 32.54% |
| 46 | 11/23/2015 to 12/22/2015 | 0.4262 | 41.69% | 53.56% |
| 5 | 3/17/2014 to 4/15/2014 | 0.4245 | 37.55% | 33.11% |
| 45 | 11/8/2015 to 12/7/2015 | 0.4188 | 41.10% | 54.63% |

## Longest fully missing runs

| CONS_NO | FLAG | Split | Fully missing windows | Maximum consecutive run |
|---|---:|---|---:|---:|
| B08E0670C8A6933E5EC44715F7A80E9D | 1 | train | 67 | 67 |
| 040A61A4680D6D770CCAF0213944B193 | 0 | val | 65 | 65 |
| 0E5FF6F545323C7382522290DCEB0DF7 | 0 | train | 65 | 65 |
| 145A9E0323B3443A046B427EBF7AF7A7 | 1 | train | 65 | 65 |
| 2E175D02BE79895286938D217E36A7A1 | 0 | train | 65 | 65 |
| 3258DA14AC82C15A98B4A2A5FEB280FB | 0 | test | 65 | 65 |
| 38779BE2A88ABCEDBD45879004D1F6FC | 0 | val | 65 | 65 |
| 6280D9197D402FA5F612C6785AEA36FA | 0 | train | 65 | 65 |
| 65B7C4532F9B092EBB992E80689FD64F | 0 | train | 65 | 65 |
| 6ECDD200AAFC9CFBB5BA4516CAAA9864 | 0 | val | 65 | 65 |

## Fixed baseline quality rule

The adopted rule remains `missing_count < 30`: remove only fully originally
missing windows. This diagnostic does not select or reconsider a threshold.
Retention, class composition, and customer coverage are reported above. More
aggressive thresholds remain diagnostic only and are not candidates for this
controlled ablation.
