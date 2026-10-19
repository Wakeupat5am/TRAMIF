# Frozen embedding artifacts — PASS

2026-10-16. Scope: verify owner-created files; no forward, training, fitting, MMD
or Future Test. Three targeted tests PASS; nine extraction runs and27 arrays PASS.

| Case | K | TRAIN reference | A | B |
| --- | ---: | ---: | ---: | ---: |
| E1 | 32 | 3266 | 1137 | 1137 |
| E2 | 42 | 4197 | 1734 | 1734 |
| Final | 51 | 5354 | 1768 | 1768 |

Each selection has three float32 matrices: SBSMI2048D, raw-byte/entropy512D.
File checksums and matrix payload hashes match; all values finite. Samples and
their order/family/timestamp/mapping equal hash-pinned G3-B selections. plan()
checks chronology and canonical historical identities; no C/Future file used.
Each case has disjoint TRAIN-reference/A/B IDs. Checkpoint identity, endpoint, K
and saved encoder-state hash agree across its three stages. Current checkpoint,
model/source/config/selection hashes match extraction provenance. Receipts record
state unchanged and three-view identity/content alignment; repeated IDs across
different cases have matching content hashes.

Existing per-image/report receipt hashes are reused, not re-read as a full image
audit. Array integrity is independently checked, but no independent inference was
run to reconstruct embeddings. Checksums do not establish statistical independence,
bandwidth adequacy, intrinsic representation superiority or predictive Drift value.

Artifacts: `data/derived/frozen_embedding_verification/20261016_121558_412739/`
contains report.json with all nine extraction receipt paths/hashes, ordered-ID
hashes, input/source hashes and companion checksums.json. Extraction files remain
unchanged in their original unique run directories. No repeated extraction.

New verifier: scripts/verify_frozen_embedding_artifacts.py.
Tests: tests/test_frozen_embedding_artifacts.py;3/3 PASS in0.001s.
Tests reject modified array values, wrong dtype/shape/nonfinite, ID permutation
and mapping changes. Full legacy suites and old audits were not repeated.

Next proposed: fit equal-family normalization and bandwidth from TRAIN reference
only for each case/view, preserving G3-B contracts. Then transform A/B using
those frozen parameters. No such fitting or observed MMD was run in this batch.
