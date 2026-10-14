# February grouping/support feasibility — 2026-10-13

Run: data/derived/february_group_audit/20261013_090516_568369/.
Engineering feasibility PASS; TLSH statistical independence is not established.
Fixed audit threshold 10; primary dataset threshold remains 0.

- 3,825 vertices, 3,799 retained samples, 26 bridge-only vertices.
- 2,032 components, all containing retained samples. Excluded bridges cause no
  additional merge relative to the retained-only graph in this dataset.
- Retained group size median 1, p90 2, p95 3, p99 24.69, maximum 144 (3.79%).
  No giant component exceeding the predeclared ideal-fold gate ceil(3799/5)=760.
- 21 components contain multiple retained family labels. Labels never split groups.
- S has 31 classes / 3,646 samples (95.97% of retained February); all classes with
  >=20 samples also meet >=5 groups. No outputs or retained samples removed.
- Fold sizes 763, 752, 761, 763, 760; group counts 404, 397, 424, 418, 389.
  All S classes have positive fitting AND scoring support in every fold.
- Zero ID, component and content-hash overlap in every fit/scoring partition.
  All exhaustive detected edges remain within a component, including bridge paths.

Label imbalance persists: pluto has 22 samples / 6 groups and scoring counts
4,2,1,14,1; stormser has 24 / 5 and counts 13,1,7,1,2; agent has 21 / 9 and
counts 2,3,2,2,12. Do not interpret five fold scores as five equally informative,
independent temporal replications. OOF per-class loss pools samples, not fold means.
Scalar temperature fitting does not require every one of the 51 outputs to have
labels in each fold. February has no allaple/tescrypt samples; non-S rare classes
remain in calibration and model outputs but not in class-balanced quality S.

All pairs inside February were considered; this rules out search approximation
misses at threshold 10, not TLSH false negatives. Transitive membership does not
imply every pair has distance <=10. Bridges outside February, original armed bytes,
other thresholds and statistically independent family lineages were not audited.
Proceed with the approved quality-only experiment using operational groups and
these limitations; do not call the threshold statistically validated.

Tests: 7 new audit tests PASS, including excluded bridges, mixed labels, exact/TNULL,
threshold boundary, deterministic partition, overlap rejection and CSV integration.
