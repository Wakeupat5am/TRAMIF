# February feasibility audit — conditional approval, 2026-10-13

The owner approved quality-only softmax(-L), coefficient 1, drift/uncertainty
coefficients 0, existing models and primary policy unchanged. Cross-fitting is
conditional on this audit. This supersedes only the approved portions of the older
reliability proposal. No coefficient grid, encoder retraining or future access.

## Scope fixed before computing the graph
Use all 3,825 original February cohort records as vertices. Join exact disarmed
content hashes or standard TLSH distance <=10, including length, exhaustively.
Connected components are label-blind. Only 3,799 primary-retained samples can
enter folds and quality scoring. The other 26 may connect components but never
re-enter calibration. No March vertices or outside-February bridges.

Use 5 folds shared across views; S is exactly classes with >=20 retained samples
and >=5 active components. Keep all 51 classifier outputs. TNULL cannot create a
TLSH edge; exact hashes still can. Missing/invalid inventory is an input error.

## Predeclared engineering checks and conservative review gate
- Complete, unique retained sample coverage, one fold per component.
- No sample, component, exact-content or detected threshold edge crosses fit/score.
- At least five active components; every scoring and fitting partition nonempty.
- S nonempty and each class in S represented in every fitting partition.
- If the largest retained component exceeds ceil(N/5), STOP for imbalance review:
  such a component cannot fit within an ideal fold. This is a conservative
  engineering review gate, not a claim that this size is a statistical theorem.
- Report per-class counts in all folds, absent classes, group-size distribution,
  mixed labels, support coverage and fold imbalance even if hard checks pass.

The single deterministic assignment sorts groups by largest fraction of a class
they contain, then size, then stable ID; seeds the first five into separate folds;
minimizes squared normalized per-class and total-size imbalance for remaining
groups. It uses February labels only for stratification, never model losses or
March labels. No repeated seeds, threshold search or performance-based selection.
This heuristic does not prove the best possible partition.

PASS means computational feasibility, not proven independence or suitability of
TLSH threshold 10. Serious concentration or unusable fold distributions found in
the detailed audit require explicit review before dependent work. No silent
changes to threshold, support or number of folds. Five within-February folds are
not five temporal episodes. Missing non-S classes do not invalidate scalar
temperature fitting but are disclosed; no claims for absent February classes.

No bootstrap, quality estimation or temperature fitting happens in this script.
