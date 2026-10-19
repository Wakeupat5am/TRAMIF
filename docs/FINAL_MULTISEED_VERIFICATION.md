# D054 — Final five-seed artifact verification

2026-10-17. **ARTIFACT_VERIFIED:15/15 epoch100 checkpoints;5/5 historical processing seeds.**
This is not RESEARCH_VALIDATED or full-V1/Future completion.

## Evidence and execution

- Verification run:`data/derived/final_multiseed_verification/20261017_164739_728045/`.
- PASS receipt SHA256:`fcd99219417e19f61520d85c868e7a3a074ce0b32ac1727bfd48d6681aad9f86`.
- Independent stdlib readback:`data/derived/final_multiseed_readback/20261017_164902_785835/receipt.json`.
- Readback SHA256:`84eb649811e3d3162fe676dba8819dd5a9e7aa2da222fcf13ae7765ca0941f7e`.
- Machine outputs:summary.json,checkpoints.csv,seed_view_results.csv,
  descriptive_statistics.csv,checkpoint_phase.json. Receipt hashes all5 outputs.
- Targeted tests:9 verifier +3 independent readback tests PASS. Not a whole-suite count.
- Source:`scripts/verify_final_multiseed.py`,`scripts/readback_final_multiseed.py`;
  corresponding test modules. No training/model forward,T fitting,MMD kernel
  evaluation,learned-static fit,real bootstrap,Future I/O,commit or push.

## Contracts verified

All seeds42–46 and views SBSMI/raw-byte/entropy are retained,with K51 and fixed
selected epoch100. Frozen TRAIN has18,061 unique samples from2019-08 through
2020-01. Three legacy seed42 configs/checkpoint payloads/registered hashes serve
as equivalent evidence; no new COMPLETE schema is imposed on legacy runs.

For seeds43–46:unique COMPLETE candidate per seed/view,100 ordered epoch metadata
receipts,18,061 samples and565 steps each,COMPLETE/selected/config/embedded-config
hash chain,seed/view/cutoff/mapping/preprocessing/Adam defaults and archived source
hashes match. Epoch100 state is loaded on CPU for config,classifier dimensions,
finite tensors and state digest only. No model is instantiated or executed.
All15 selected checkpoint payload hashes are distinct.

Each processing freeze has exactly three pinned view receipts,matching the
corresponding checkpoint and unchanged encoder-state hash. Source snapshots and
recorded metadata dependencies match; no source changes required adjudication.
Current source differences would require contract review,not automatic corruption
claims. No latest/performance-based candidate selection was necessary.

All three views and five seeds use canonical class/ID alignment. Legacy saved
prediction order differs from canonical order:verification explicitly joins by
unique sample ID and checks labels/classes,never fuses unchecked positions.
February A3799 fit IDs,March B4332 retained IDs,S25 and4064 B/S samples match.
Q keeps all51 outputs,uses float64 stable log-softmax(logits/T),and averages family
means equally. Per-sample NLL,mask,per-family means and standard NLL read back.
No cap128 is applied to Q.

TRAIN reference5354 and capped A/B1768 each use the frozen manifests. Verification
reads saved raw embeddings,checks payload digests and recomputes TRAIN-only
weighted moments and saved TRAIN pair distances/weighted median for consistency;
it does not fit new parameters or evaluate an observed MMD kernel. Coordinate mask
std<=1e-8 and float64 arrays match. Each view has five distinct reference-embedding
and normalizer payloads,each bound to its own seed checkpoint. A/B content IDs match
across views/seeds,including the legacy capped-vs-new full-retained correspondence.

Stored RBF kernel-mean terms,roundoff clipping,equal-family aggregation and
MMD2/2 arithmetic agree with reports. This is provenance/algebra verification,
not an independent rerun of MMD. Q=softmax(-L),QD=softmax(-L-d),beta1,gamma0 and
no U penalty all agree at full precision. Float64 checks use1e-12 absolute/relative
numerical tolerance; this is not tuned to classification performance.

Original image/binary corpus was not re-audited. Recorded image hashes/content
identities are linked through receipts and saved arrays; raw images were not
redecoded. Source/input metadata and output files were verified across1802 paths.

## Checkpoints

| Seed | View | Selected checkpoint | SHA256 | Status |
|---|---|---|---|---|
|42|sbsmi|`data/derived/sbsmi_training/20261005_105325_352508/epoch_100.pt`|`0c7d76ec3e077e5f88dd6d455b14a041ee2d701a2dc170d5718928695bcf2c4d`|VERIFIED|
|42|raw-byte|`data/derived/float_view_training/raw-byte/20261009_145102_408184/epoch_100.pt`|`893eb01789858abab523a3909d40a403328f27d17901dc76a9c94bfbd551341f`|VERIFIED|
|42|entropy|`data/derived/float_view_training/entropy/20261011_001259_203865/epoch_100.pt`|`1fc07d49f8dbc0f83cb8e88e1e987cbb070de707713f1f8d941f9e1626771cfa`|VERIFIED|
|43|sbsmi|`data/derived/final_seed_training/seed_43/sbsmi/20261017_041625_702882/epoch_100.pt`|`0c85747e0fdbfd153bb507b59c1aa83fb085bb2d76bcbd1481c38180266780f2`|VERIFIED|
|43|raw-byte|`data/derived/final_seed_training/seed_43/raw-byte/20261017_050605_805807/epoch_100.pt`|`9b587bb81f13d08187f6ca35e92e744f26d783435c2c4a1841f3629bae45d075`|VERIFIED|
|43|entropy|`data/derived/final_seed_training/seed_43/entropy/20261017_054905_114285/epoch_100.pt`|`33097bf2ab5dd71580d9266e0775bd6e52d0411b191ab7c8fedcefe84391bc12`|VERIFIED|
|44|sbsmi|`data/derived/final_seed_training/seed_44/sbsmi/20261017_070815_282298/epoch_100.pt`|`a1cd807a48426171f536535ab0a0e1f1a653e54577befcd94c7f60f45ce1c39c`|VERIFIED|
|44|raw-byte|`data/derived/final_seed_training/seed_44/raw-byte/20261017_082515_558157/epoch_100.pt`|`a90c5d49d90b6a3cb385ca4faacbaa9695f6d1714dd37b6a380650b82ea5c8d2`|VERIFIED|
|44|entropy|`data/derived/final_seed_training/seed_44/entropy/20261017_092716_551623/epoch_100.pt`|`7cbaf6663dfa43fed15c1e4e41d1611fa2f1846e40754f84e89cb5ff6512559c`|VERIFIED|
|45|sbsmi|`data/derived/final_seed_training/seed_45/sbsmi/20261017_102500_528100/epoch_100.pt`|`b01ea289bbfa622578e040c6963fe6851fb4cb4bfffa71f81ec2b356426fd253`|VERIFIED|
|45|raw-byte|`data/derived/final_seed_training/seed_45/raw-byte/20261017_113207_205893/epoch_100.pt`|`548542533585befecef498bec2e454d8e17d915a95a6cf136e755f48babd224c`|VERIFIED|
|45|entropy|`data/derived/final_seed_training/seed_45/entropy/20261017_122658_433208/epoch_100.pt`|`0b08b97998149d35f8a25c17ac9457ea667fd1adf8f219adcc65c517e297f1ee`|VERIFIED|
|46|sbsmi|`data/derived/final_seed_training/seed_46/sbsmi/20261017_132701_940777/epoch_100.pt`|`a1ef7fef790cd917b830ea1bbbb167633d5418952fc394a792f96e0fcd9c9490`|VERIFIED|
|46|raw-byte|`data/derived/final_seed_training/seed_46/raw-byte/20261017_142309_156298/epoch_100.pt`|`42be85f76cee87f39c7d3a588d04a1edb1c3e92780c25cc3c013bbb0486279aa`|VERIFIED|
|46|entropy|`data/derived/final_seed_training/seed_46/entropy/20261017_151737_144509/epoch_100.pt`|`c9f43cb296a34228642e434e6e7684a248e0863d322e0b738f5c7918ff9016dc`|VERIFIED|

## Verifier development and limitations

Three earlier STOP receipts are preserved under final_multiseed_verification:
164551_047539,164625_802991 and164704_837258. These were verifier implementation
assumptions,not artifact failures:legacy technical-status enum,report-vs-checksum
pin target,and explicit legacy prediction ID joining. Each was corrected using
existing schema/source evidence with regression tests. No frozen hash/data/source
was repaired to obtain PASS,and no training/postprocessing was retried.

All five registered seeds are included without choosing best performance. This
verifies recorded artifacts,not every possible decision outside provenance.
Five seeds share the same temporal history and are not five independent temporal
experiments. Descriptive MMD is not null-standardized or proof of future reliability.
E2's slightly worse QD fusion F1/NLL/Brier remains unchanged; beta1 is not retuned.

## Remaining pre-Future blockers

1. Approve the proposed learned-static numerical protocol,then perform its CPU
   empirical fits on each seed's March B/S; no fit performed in D054.
2. Resolve final statistical estimands/contrast family,multiplicity and coverage
   validation before any real interval procedure; U arms remain unavailable.
3. Resolve scope/claims for file-size,capacity/cost and broader framework controls.
4. Freeze a combined five-seed experimental package and verify backup/restore of
   the newly generated checkpoints/processing artifacts; off-device backup UNKNOWN.
5. Obtain separate approval for Future input completeness/retention/inference and
   six-month evaluation. No Future input was opened or prepared here.

Next optimal action:owner review/approve the already-proposed learned-static
solver protocol,then fit that missing baseline on the five historical B/S sets.
