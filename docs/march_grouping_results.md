# March grouping and support pilot — 2026-10-13

Task 4B COMPLETE as a feasibility audit, NOT a final uncertainty report.
No prediction metrics computed in pilot replicates. No filtering, relabeling,
weight fitting, Future Test access or final bootstrap confidence interval.

## Graph scope and counts
All 4,438 original March cohort records are vertices; 4,332 retained records alone
enter support counts. The other 106 are bridge-only vertices. Edges use exact
disarmed-content SHA equality OR standard TLSH <=10 including length. All
9,845,703 March pairs considered; 28,516 qualifying edges. TNULL=0.

| Quantity | Observed |
| --- | ---: |
| All components | 2,236 |
| Active components with retained samples | 2,234 |
| Excluded-only components | 2 |
| Additional retained-component merges due to excluded bridges | 0 |
| Multi-label components, all / active retained | 28 / 28 |
| Largest component including excluded vertices | 185 |
| Largest retained component | 166 (3.832% of retained samples) |
| Active retained group median / p90 / p95 / p99 | 1 / 2 / 3.35 / 26.67 |
| Active singleton components | 1,907 |

The largest retained component, M000010, contains sillyp2p/small (174 total, 166
retained). The largest original component, M000002, has 185 total / 131 retained.
No single retained component dominates the entire March dataset, but group sizes
are highly uneven. Full discrete distributions, members and edges are saved.

## Effective group concentration — not independent sample size
Define `G_size = (sum_g n_g)^2 / sum_g n_g^2`, where n_g is retained group size.
Observed G_size=174.5143 versus 2,234 nominal active groups. Equal-sized groups give
their actual count; a few large groups reduce this index. This is a descriptive
concentration measure only, not a variance estimate or proof of 174 independent
units. Per-class analogues use each group's sample count from that class.

Even larger nominal class-group counts can concentrate: agent has 47 samples / 11
groups but size-effective groups 2.918; dinwod has 187 / 29 but 3.694. These are
not reasons to alter evaluation classes or remove samples.

## Class support and 500 pilot replicates
The original supported set remains 47 classes; the four absent classes are not
counted as bootstrap support failures. Draw G=2,234 whole active groups with
replacement per replicate, seed 42. All members and mixed-label relationships stay
together. Sample size varies with group multiplicities. Save all 500 replicates;
zero discards, retries or denominator changes. No method scores or intervals.

| Family | Retained samples | Groups | Pilot replicates missing family |
| --- | ---: | ---: | ---: |
| blocker | 3 | 2 | 81 |
| cryptinject | 6 | 3 | 21 |
| ditertag | 3 | 3 | 22 |
| fasong | 7 | 3 | 28 |
| fuerboos | 2 | 2 | 70 |
| gupboot | 4 | 2 | 69 |
| mocrt | 3 | 3 | 22 |
| pluto | 11 | 3 | 29 |
| stormser | 9 | 3 | 34 |
| vb | 2 | 2 | 67 |

304/500 replicates (60.8%) miss at least one originally supported class; 196 retain
all 47. Number of missing classes -> replicate count:
0 ->196, 1 ->198, 2 ->78, 3 ->21, 4 ->6, 5 ->1.

For a class represented in m of G groups, its exact omission probability under
this specified empirical resampling scheme is `(1-m/G)^G`. For m=2 it is 13.5214%;
for m=3 it is 4.9687%. These probabilities describe bootstrap draws, not the chance
of a family disappearing in a real future month. Class-missing events are dependent
when groups contain multiple families; do not add these probabilities to estimate
the any-missing rate. The 60.8% rate is a finite 500-draw pilot estimate.

## Feasibility interpretation
- Paired whole-group resampling is computationally feasible with one common draw
  vector for all methods; no technical need to split mixed-label groups.
- Broad supported-class F1 / balanced-accuracy uncertainty is not ready for this
  naive resampling design. Keeping only the 196 complete-class draws conditions on
  support and changes the bootstrap distribution. Averaging over fewer classes
  changes the estimand. Neither action is allowed.
- Fixed-51 F1 can be numerically defined with the established zero convention,
  but an interval heavily affected by rare-class omission would need explicit
  methodological justification. A defined number alone does not establish validity.
- Accuracy/NLL/Brier remain computable for every nonempty draw. Conditional paired
  intervals for these can be proposed, with limits on group independence and size
  imbalance. ECE is also computable but nonlinear/bin-based; keep its uncertainty
  exploratory rather than an additional confirmatory endpoint.

TLSH components do not prove independence/exchangeability. Direct edges are
exhaustive only within March; no outside-month bridge nodes are considered.
Disarming, TLSH false negatives and transitive closure can distort similarity.
The grouping threshold was not selected using model performance. Pilot outcomes
do not justify tuning it for narrower intervals or more favorable significance.

## Artifacts and tests
Run: data/derived/march_group_feasibility/20261013_125459_149338/.
members.csv, groups.csv, edges.csv, class_support.csv, pilot_replicates.csv,
pilot_counts.npz (all group multiplicities/counts/missing masks), report.json,
checksums.json. Primary retained selection matches the March evaluation by ID/label.
Report SHA-256: `70cbcac4d34819291593a719599a11ff3b3637ab0b067db3060b539c6a3f523d`.
Entry point scripts/march_group_feasibility.py; seven new tests cover bridge nodes,
exact/TNULL links, label-blind graph, month rejection, budget, mixed-label draws,
no dropped replicates, deterministic seed and CSV/NPZ integration.
Final full suite: 109/109 PASS (0.713s).
