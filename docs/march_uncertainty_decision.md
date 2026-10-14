# March paired uncertainty — historical proposal, superseded

Current status (2026-10-13): the owner approved D012, with seed **42**, 2,000
whole-group draws and Accuracy/NLL/Brier intervals only. Execution is complete;
see [results](march_paired_bootstrap_results.md) and DECISION_LOG.md D013.
The proposal below is retained as decision history, not active configuration.
Its proposed seed 20261013 and request for new draws were superseded; the executed
seed-42 sequence includes the earlier support-pilot prefix, explicitly disclosed.

## Updated recommendation after the approved feasibility audit (2026-10-13)
This update supersedes the pre-audit proposal below. Metrics and support-only
pilot are now complete, but formal bootstrap remains UNAPPROVED. Evidence:
docs/march_quality_only_results.md and docs/march_grouping_results.md.
There are 2,234 active groups and 304/500 draws miss original supported classes.
No interval or significance claim was computed from these pilot draws.

Recommended next bounded uncertainty protocol, for OWNER APPROVAL:
1. Freeze the audited March graph and the exact 4,332-sample, five-method artifact.
   Keep weights, temperatures, thresholds and class mapping unchanged. Do not use
   bootstrap outcomes to choose a new grouping, metric, model or weight vector.
2. Use 2,000 new paired whole-group draws, seed 20261013, with G active groups
   sampled with replacement. This is a proposed final Monte Carlo budget, NOT
   currently authorized. Keep all mixed-label members and group multiplicities;
   use identical draws across methods. Preserve variable resampled sample counts.
3. For quality-minus-uniform, report 95% percentile intervals for Accuracy, NLL
   and Brier as CONDITIONAL development uncertainty. All are computable for every
   draw; missing-class draws remain included. Accuracy uses the replicate sample
   count, not fixed 4,332. NLL/Brier are sample means with the same multiplicities.
   No refitting February temperatures or quality weights inside these intervals;
   therefore these do not capture training/calibration/weight-estimation variance.
4. Keep supported-47 Macro-F1, fixed-51 Macro-F1 and balanced accuracy as point
   estimates WITHOUT formal bootstrap intervals in this round. This is explicit
   noncompletion of class-metric uncertainty, not dropping those metrics from the
   report. Do not substitute the February 31-class quality support for March's 47.
   A later class-preserving design must respect mixed-label groups and acknowledge
   the 2-3-group classes; simply increasing replicate count cannot create support.
5. Keep ECE-15 as a point estimate in the recommended minimal protocol. A separate
   exploratory interval is optional only if explicitly approved; its binning and
   nonsmooth statistic make it less suitable for a strong comparative claim here.
6. Save all replicate deltas, multiplicities, class-support flags, distribution
   summaries and histograms. Inspect extreme/multimodal distributions and report
   instability; no removing unfavorable draws. No p-values or global superiority
   claim based on selected intervals. Supported Macro-F1 remains reported as the
   classification endpoint; easier uncertainty for other metrics does not replace it.

Intervals would describe the empirical March cluster population under an unverified
between-component independence approximation. They are not final future-test
evidence. Nominal cluster count 2,234 and size-concentration count 174.5 do not
certify independence or statistical validity.

Methodological reference: Cameron & Miller (2015), Sections III.B and VI.C,
[A Practitioner's Guide to Cluster-Robust Inference](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf).
It supports resampling whole clusters, inspecting bootstrap pathologies and not
naively using only surviving draws. It studies regression inference; it does NOT
validate TLSH grouping, our F1 functional, or the proposed TRAMIF intervals. The
recommendation above is a conservative project-specific proposal based on the audit.

The alternative is to retain point estimates only and defer all formal uncertainty.
Either option needs owner choice; no final-bootstrap implementation in this batch.

## Historical pre-audit proposal (retained for decision traceability)

Quality weights are frozen and February provenance verified. New March comparison
has not run. This document does not authorize a new grouping or bootstrap policy.

## Why a decision is needed
The approved threshold <=10 was scoped to February feasibility. A valid paired
uncertainty calculation also needs resampling units for March. Reusing February
groups does not describe March dependence; treating each March file as independent
ignores known similarity. Choosing a March threshold or silently switching to an
IID file bootstrap would add an unapproved methodology choice.

Earlier primary filtering removes certain earlier-related samples at threshold 0;
it does not prove that the remaining March samples are independent at threshold 10.
Hash equality alone likewise does not establish independence of similar binaries.

## Recommended bounded proposal (owner approval needed)
1. Audit March operational components using content-hash equality OR standard TLSH
   <=10, with all original March cohort records as graph vertices. Excluded March
   records may bridge retained records but never re-enter scoring. Use the frozen
   primary retained March set, class mapping, checkpoints, inference temperatures
   and weights. No February/March cross-period bridge expansion in this proposal;
   disclose that limitation. This changes resampling units, NOT primary filtering.
2. Report component sizes, per-class group support and multi-label groups. Stop
   for review if a giant component or concentration makes uncertainty unusable;
   no threshold search to improve confidence intervals or results.
3. Compare the approved five methods on the identical 4,332 retained March samples.
   Preserve all seven requested point metrics, even unfavorable ones.
4. Conditional diagnostic paired cluster bootstrap: 500 replicates, seed 42;
   sample the number of active components with replacement, include all retained
   members of each selected component with multiplicity. Use identical draws for
   quality-only and uniform. Save every replicate and the sampled group counts.
   Report quality minus uniform deltas; for loss metrics negative is improvement.
5. Fix the original March supported-class set (47) and fixed mapping (51) BEFORE
   resampling. Never silently average macro-F1 over whichever classes appear.
   Accuracy/NLL/Brier/ECE remain defined when a replicate misses a class. The
   fixed-51 F1 convention remains zero for 0/0, explicitly disclosed.
   Flag missing original-supported classes in every replicate. If any replicate
   lacks an original supported class, do NOT publish a bootstrap interval for
   supported-class macro-F1 or balanced accuracy pending a further decision.
   Preserve all replicates; no retry/discard or conditional-on-valid CI.
6. For eligible diagnostic intervals, use 2.5/97.5 percentiles. These are conditional
   on fixed models, grouping and March's empirical group population, not training
   uncertainty, future generalization or a proof of independent components. Small
   groups-of-groups support can make percentile intervals unstable. Do not infer
   certain superiority or choose new weights from the intervals.

This conservative missing-class rule may leave class-metric intervals unavailable.
That is preferable to quietly changing the estimand. A class-preserving scheme
that also preserves mixed-label groups needs separate design; naive within-class
resampling would split those groups and is not proposed here.

Alternative: approve point estimates only for now, explicitly leaving Task 4's
paired uncertainty incomplete. No bootstrap is then run. Either decision leaves
the frozen February weights and Future Test protection unchanged.
