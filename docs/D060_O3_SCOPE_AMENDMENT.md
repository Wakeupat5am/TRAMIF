# D060 — Owner adoption of D059, with O3 modification

2026-10-18. Recorded **before updating the statistical specification/hash**.
Authority: owner's D060 execution instruction in this conversation.
The immutable D059 proposal/snapshot remains preserved under
`data/derived/d059_statistical_specification/20261018_052308_445235/`.
Previous specification SHA256:
`c84b35bb2191b72189c0261b533e5b8113375bd28a8c8aca774f18877c5fe795`.

O1,O2,O4 APPROVED: original-data estimates and five-seed descriptive spread;
eight contrasts; exact metrics/edge rules/practical targets. No CIs,p-values or
significance claims. Coverage has not been validated.

**O3 APPROVED WITH MODIFICATION:** retain both retention10/20 sensitivity AND
file-size-stratified descriptive analysis under V1. The latter is required in the
reduced study, rather than deferred as optional in D059. File-size cutoffs must
come only from the frozen TRAIN cohort, never Future outcomes. Neither analysis
may select new hyperparameters,thresholds,weights or preferred methods.
U remains diagnostic-only,D descriptive; disclose unperformed capacity/packing/
broad baseline/end-to-end controls. No claim of full original-framework validation.

Implementation conventions for the approved size analysis: freeze pooled TRAIN
byte-size quartiles using NumPy float64 linear quantiles at .25/.5/.75. Equal-size
ties stay together: bins (-inf,q25],(q25,q50],(q50,q75],(q75,inf). Repeated cutoffs
can make bins empty; report NOT_ESTIMABLE/count0 rather than change cutoffs. For
each month/bin publish size range,sample/class counts and each method/seed's
supported and fixed51 metrics, plus F1 on the original whole-month support to
show composition effects. Empty bins do not invalidate nonempty original months
or get averaged into a replacement primary estimand. No causal adjustment claim.
These are deterministic reporting conventions; primary estimands are unchanged.

D060 authorizes bounded synthetic integration, artifact packaging and local
backup/restore. It does not authorize opening Future, training, fitting or retuning.
Any later protocol change needs a new recorded decision/version.
