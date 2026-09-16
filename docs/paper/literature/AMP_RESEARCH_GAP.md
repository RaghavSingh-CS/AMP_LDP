# AMP-LDP research gap

## Evidence-bounded gap

Recent electricity-theft research in the verified set is dominated by supervised customer-history classifiers, including contrastive, recurrent, convolutional, graph, transfer-learning, and optimization-heavy pipelines [R01–R13]. Foundational SGCC work similarly maps a complete customer history, or transformed full-history views, to a customer theft label [R17, R19]. These studies establish that temporal consumption representations can support customer discrimination, but their preprocessing, class balancing, split construction, model selection, prevalence, and reported metrics vary materially [R02, R04, R09, R17, R20]. A shared dataset name therefore does not create a protocol-matched comparison.

Multiple-instance learning supplies a formal language for bags with ambiguous instance labels [R21], while time-series representation learning supplies mechanisms for encoding temporal segments [R22]. Within the **24 verified journal articles reviewed here**, however, no paper was found that simultaneously:

1. treats temporal windows as unlabeled instances under an inspected customer-level theft label;
2. learns a window representation without imposing a theft loss on individual windows;
3. preserves the within-customer distribution of learned window representations through matched summary statistics;
4. compares that learned representation against the *same* distribution statistics applied directly to handcrafted window features; and
5. freezes the resulting customer-level model and tests the qualitative effect across separate development, confirmation, and governed final-evaluation cohorts.

This is a **bounded finding about the verified set**, not a universal priority or “first method” claim.

## Narrow research question

> Under customer-level SGCC theft labels and no verified theft-event timestamps, does learning representations of unlabeled consumption windows before distribution-preserving customer aggregation improve customer-level discrimination beyond applying the same aggregation directly to handcrafted window features?

## How AMP-LDP addresses the gap

AMP-LDP separates label granularity from temporal representation. Thirty-day windows are measurement segments, not verified theft periods. A 17-dimensional window summary is encoded to 16 latent dimensions; six distribution statistics per latent dimension form a 96-dimensional customer representation; a histogram gradient-boosting classifier produces customer probabilities; and five preregistered seeds are averaged. The matched DIRECT control applies the same six statistics to the 17 handcrafted dimensions, isolating learned encoding from distribution aggregation. FULL_48 supplies a strong engineered customer-profile comparator.

The completed AMP evidence supports a reproducible *within-dataset, governed-cohort* result, not external generalization, theft-period localization, causal identification, or state-of-the-art status.

## Remaining literature and evidence gap

The verified literature does not remove the need for external utility validation, event-level theft annotations, or protocol-equivalent benchmarking. Future literature work should verify additional journal studies only when their original records and evaluation protocols are accessible; adding headline scores without split and prevalence context would weaken rather than strengthen the comparison.
