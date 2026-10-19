# D060 — Pre-Future integration and combined freeze

2026-10-18. **Engineering PASS; Future remains closed.** No model training,
inference,temperature/weight fitting,beta retuning or real Future evaluation.

## Approved scope and specification

O1/O2/O4 approved. O3 modified to REQUIRE retention10/20 sensitivity and
TRAIN-quartile file-size descriptive analysis. Amendment recorded first in
`D060_O3_SCOPE_AMENDMENT.md`; original D059 snapshot and original V1 preserved.
Current authoritative `FINAL_STATISTICAL_PROTOCOL.md` SHA256:
`3a2d575659b95155bc21a0f0b3f91789392891b84b9b62d6c8596a9158d74feb`.
Same two primary outcomes,five seeds,K51,eight contrasts,beta1,gamma0.
No CIs/p-values/significance; U diagnostic-only,D descriptive.

## What was verified

| Item | Result | Evidence/limit |
|---|---|---|
|Five seeds × seven arms|PASS SYNTHETIC INTEGRATION|48synthetic samples,six synthetic months,full51outputs;actual registered checkpoint/T/weights metadata |
|ID/class/label/provenance guards|PASS|Reject shuffled IDs/classes,mismatched labels/seed/checkpoint/T,nonfinite logits,missing seeds/months |
|Probability fusion|PASS|Own-seed calibration;uniform arithmetic probability average;frozen Q/QD/learned mixture;zero learned weights supported |
|Aggregation|PASS|Per-seed Mean/Worst first,then five-seed mean/SD(ddof1)/range;8paired differences;original monthly support |
|Retention0/10/20 logic|PASS SYNTHETIC|Same direct-earlier policy as historical code;excluded bridges remain references;content equality,TNULL,standard TLSH length component,label independence,nested subsets |
|File-size logic|PASS|Cutoffs fit ONLY from18,061frozen TRAIN sizes;empty bins explicit,all methods/seeds same subsets,no weight refit |
|Independent readback|PASS|630synthetic monthly confusion matrices;stdlib F1/mean/worst/SD/paired deltas agree to1e-12 |
|Combined model/signal freeze|PASS|15checkpoint bindings;5own-seed sets;D054/D058 receipts reused;byte hashes checked for packaging without loading checkpoints |
|Historical retention reference addendum|PASS|26,324original historical inventory rows incl excluded;no pairs/graph recomputed |
|Local restore/reproduction|PASS|1,872files incl15checkpoints;all restored hashes match;synthetic replay from restored code equals original |

**44 distinct targeted tests PASS:**17evaluator/retention/size,13primary-statistics,
9package/backup and5reference-binding. Initial38-test suite PASS;after correcting
the package adapter,9affected tests PASS(including new schema regression);then
5reference tests PASS. No whole-repository count inferred.
The adapter initially expected summary status PASS instead of D054's actual
ARTIFACT_VERIFIED(root receipt is PASS). Fixed adapter only;failure artifact retained
in the first `combined_freeze_v1` run. No legacy evidence or expected hash was changed.

Size quartiles frozen in bytes: q25=86414,q50=254672,q75=721840.
Bins right-closed;ties stay together. Reports include per-bin support/counts,
supported/fixed51 metrics and F1 on original month support. Empty bins are explicitly
NOT_ESTIMABLE. These are descriptive strata,not causal/packing adjustment. No real
historical B/C size analysis or real Future stratification was run in D060.

## Immutable evidence

Base combined package:
`data/derived/combined_freeze_v1/20261018_080413_102200/package.json`.
Package SHA256:`4ccd7be2b6f622ffd7903a9aeafe27988ad9f6e905502f235f4f395d26279862`.
Receipt SHA256:`738c4d9cba4e28fab197b3e53d412b6bfa5d6085f7004a873e1e4797089d8c7f`.
Includes1,868explicit asset bindings,canonical TRAIN size manifest,preprocessing,
source/spec hashes,per-seed checkpoint/T/Q/QD/learned bindings. Immutable D054/D058
evidence reused;no repeated checkpoint deserialization,optimizer or signal audit.

Integration/readback:
`data/derived/d060_integration_verification/20261018_080451_805081/`.
Receipt SHA256:`de9f64d7481d0b9079f9f6e828e672f3d39da0b3674eba360bf2f7d09cb3e8f4`.

Historical reference addendum:
`data/derived/d060_retention_reference/20261018_080848_504344/`.
Receipt SHA256:`cdc96d04d0e4aa8ee0dbd8bad8a867d90b66808c65d5dcf60d0f76dd38da9d3c`.
It binds the base package to original train/validation TLSH inventories/configs/
summary checksums,checks IDs/timestamps/families against historical canonical
metadata,and includes its own zip/restore. The base package was not overwritten
to add previously unlisted reference files. Both components are required in the
final bundle index,not a choice between alternative experimental candidates.

Main backup:
`data/derived/combined_freeze_backup/20261018_080540_341571/`.
Archive bytes954706206(910.48MiB);source bytes1309625596.
Archive SHA256:`1f0ff9a3c189a139fe750d08ae4cb124ddcaf42654e2a48fd461413b4756e5fd`.
Receipt SHA256:`a7d5083fb77afa9a6396be1282e55d6e8ecc62ac54604410225c284ce8ba5216`.

## Reproduction boundary and backup coverage

Restored historical arrays/checkpoints/metadata/source snapshots and frozen weights
match their byte hashes. Synthetic evaluation replay matches exactly in the same
Python/NumPy/Torch environment. This is not a fresh-machine dependency installation,
CNN training reproduction or evidence of real-data prediction quality. No registered
checkpoint was run during restore. Existing saved historical arrays are copied,
not regenerated or reevaluated.

**Not protected by this local backup:** loss of the same machine/disk;complete raw
malware archive;complete generated image datasets;dependency wheels/venv/CUDA driver
installation. No off-device copy has been verified in D060. Copy all bundle-listed
archives plus bundle index to independent storage and verify SHA256 for disaster
recovery;do not delete originals based on a local-only archive.

## Remaining scientific limits and exact Future conditions

No additional methodology choice blocks the approved reduced study. Limits remain:
descriptive/nonstandardized MMD,negative U evidence and unavailable QU/QDU;overlapping
historical episodes;unadjusted packing/collection effects;unequal encoder capacities;
five shared-data fits and a fixed six-month horizon;no validated inferential coverage.
Retain negative E2 results,no full-framework/SOTA/causal/efficiency superiority claim.

Before real Future scoring:
1. Owner explicitly authorizes protected April–September input preparation and,
   separately or in the same staged instruction,evaluation after input verification.
2. Use the final bundle index/base+reference addendum hashes;verify no source/config
   mismatch. All15models/own-seed T and Q/QD/learned weights stay frozen.
3. Authorized preparation must create immutable complete known-family manifests,
   content/TLSH/retention0/10/20 decisions with witnesses and coverage,consistent
   images and five-seed logits receipts;all methods share IDs/class order per set.
   Include all earlier historical/within-Future records as references even excluded.
   Future ingest/inference integration is not proved by synthetic tests here.
4. STOP on missing/corrupt inputs,wrong cutoff/provenance or empty primary month;
   no silent dropping/retry to improve results. Report nonestimable sensitivity
   subsets/empty size bins explicitly. Publish all methods,seeds,months,contrasts,
   strata and negative results without tuning.

D060 does not authorize any of those Future data operations. The next concrete
step is owner-authorized protected input preparation using this frozen bundle.
