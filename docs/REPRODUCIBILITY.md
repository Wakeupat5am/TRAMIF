# Historical reproduction and artifact contract

## D062 — Final saved-logit reproduction (2026-10-18)

Owner explicitly authorized final scoring after D061. Completed run:
`data/derived/final_future_scoring_v1/20261018_141835_165405/`;receipt SHA `53c3c885851136549eacb8756ccd3c44debec3fea5e24bc722efb6a730840bc6`.
Read docs/FINAL_FUTURE_RESULTS.md and config/source snapshots before reproduction.
No training,inference,calibration or weighting fit is needed to reproduce metrics.
Targeted tests (37PASS):
```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_final_future_scoring tests.test_d060_integration tests.test_final_point_statistics -q
```
Deliberate CPU reproduction only;creates a fresh run without replacing originals:
```powershell
.\.venv\Scripts\python.exe -B -m scripts.score_final_future --owner-approved-final-scoring
```
CLI pins exact D061 receipt/D060 bundle and verifies source/config/input chains.
Independent metric readback is automatic;compare full-precision result fields and
predictions,not timestamps/elapsed time or ZIP container hashes between runs.
Do not rerun inference or select favorable outputs. Preserve negative practical
result. Old seed42-only evaluator is not the final seven-arm/five-seed consumer.
No final CIs/p-values;five-seed SD is descriptive. Off-device D061/D062 backup is
not established. Historical reproduction notes below describe earlier checkpoints.

2026-10-17. Use repository root and the existing verified environment. No install,
CNN training or Future operation is required to inspect current historical outputs.
The repo contains uncommitted/untracked work; Git is not an artifact backup.

## Inspect first, do not rerun completed inference

`automation/continuation_state.json` identifies completed runs, hashes, blockers and
the exact next owner decision. Original checkpoints, Q, D and temperatures are
immutable. Historical E1/E2 reports plus raw logits/probabilities already exist;
compare to the hashes in historical_research_results.md before recomputation.

Focused CPU synthetic verification (no real inference):

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_historical_episode_evaluation tests.test_qd_selection_protocol tests.test_e1_outcome_preparation tests.test_frozen_future_point -v
```

Result at this checkpoint:20/20 PASS. Tests do not establish research validity.
The Future guard test deliberately raises SystemExit before any file read.

## Exact historical execution commands (already completed)

For a deliberate fresh reproduction only, commands create NEW run directories:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.evaluate_historical_episode --case E1
.\.venv\Scripts\python.exe -B -m scripts.evaluate_historical_episode --case E2 --e1-selection data/derived/historical_episode_evaluation/E1/20261017_033742_471099/beta_selection.json --e1-sha256 a36ec42417923921739f455814d8a41e41a9937fb8789ce014794236802820c2 --execute-inference
```

E1 reuses the pinned saved C logits. E2 command performs CPU C inference; do not run
it merely to inspect saved results. Registered E1 selection, not an arbitrary
latest directory, controls E2. No resume/retry, remapping, sample intersection or
temperature fit is performed. New reproduction IDs differ but numerical/config
bindings should match; compare full precision before claiming exact equivalence.

Every report saves input/source hashes. Output checksums exclude their own file.
Existing primary MMD has a documented bad self-entry; use its pinned report and
supplemental verification, never silently repair the original. Package freezes
all numerical values; displayed rounded tables are not execution inputs.

## Stored artifacts and schema

- E1/E2 report.json: seven point metrics, confusion matrices, class supports,
  per-class results, beta, temperatures/provenance and phase roles.
- predictions.npz: six methods, canonical IDs/classes/labels, normalized full-K
  log probabilities and predictions. E2 also preserves per-view raw logits.
- forecast_scores.json: S_eval/missing S, observed losses, all fixed forecasts'
  signed/absolute residuals; E2 selector is None and cannot choose a new beta.
- beta_selection.json: E1-only candidate scores/tie evidence and locked-protocol hash.
- historical_research_summary CSVs: display/export data with full precision.
- historical_metrics.vl.json: standard plotting specification, SPEC_ONLY;
  no rendering library installed and no figure falsely labeled publication-ready.

Manifest SHA identifiers and disarmed-content hashes differ. Reproduction must use
the same disarmed bytes/preprocessing, not substitute original/feature-vector data.
Python/NumPy/Torch/source versions and hashes remain in predecessor receipts;
hash mismatch is a STOP, not an instruction to edit data to pass.

## Future remains closed

Local curated archive and restored-metric replay PASS:
`data/derived/research_snapshot/20261017_034656_224911/`.
282 files, SHA256 manifest, CRC verification and restored E1/E2 metrics are saved.
Archive SHA256:`690bbbe550ccd00cbc27718ff668456e7466dad84d74d590db6cddf198a569d4`.
This local snapshot precedes the final documentation/checkpoint addendum; its own
manifest defines exact included versions. It is not an off-device backup and
does not contain raw malware/image datasets. Archived code was not executed.

PRE_FUTURE_READINESS_REPORT.md lists the incomplete full-framework gates. The
dormant point evaluator needs a future receipt/preparation pipeline not constructed
here. Do not invoke its owner-approval flag until actual approval; a guard flag
does not itself provide consent. No future manifest, predictions or results are
included in this historical reproduction path.
# D052 engineering reproduction addendum

Run from C:\Users\hoang\Desktop\TRAMIF:
`& .\automation\reproduce_pre_future_engineering.ps1 -LocalSnapshot`.
Runs only25 new targeted synthetic/CPU tests, records individual test IDs and
source/config hashes, optionally verifies a curated ZIP restore in a temporary
directory. No training, real-data postprocessing, empirical solver fit, real
bootstrap or Future data access. New receipts are under ignored
data/derived/pre_future_engineering/verification/. This does not replace raw
image/binary backup or prove off-device recovery. Use the prior historical snapshot
for existing checkpoints/results; no new checkpoints exist in this batch.
Exact training/post commands and prerequisites:docs/FINAL_MULTISEED_EXECUTION.md.
