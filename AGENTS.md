# TRAMIF development rules

## Research contract
- Study known-family malware classification under temporal shift using raw-byte,
  local-entropy and SBSMI views. Historical reliability and fusion remain research
  work to validate, not established improvements.
- Train: 2019-08 through 2020-01; validation: 2020-02 and 2020-03;
  future test: 2020-04 through 2020-09. Select families from train only:
  at least 50 unique samples across at least three train months (current K=51).
- Preserve original metadata, disarmed binaries, class mapping and chronology.
  Sample identifiers and disarmed-content hashes have different meanings.
- Current historical near-duplicate protocol is docs/near_duplicate_protocol.md:
  primary TLSH distance 0, sensitivity 10/20; labels do not decide similarity.
  The policy was developed after initial SBSMI validation. Do not rewrite history.
- Current temperatures use filtered February; March is development validation.
  Calibration is not historical reliability. No future-driven model/weight tuning.

## Work batches and evidence
- Read Git status and PROJECT_PROGRESS.md first. Select 2-4 bounded tasks with
  dependencies and acceptance criteria. Preserve working modules; avoid unrelated
  refactors. Read relevant code before editing. Add meaningful tests and run them.
- Distinguish IMPLEMENTED, UNIT TESTED, INTEGRATION VERIFIED, EXPERIMENTALLY
  VERIFIED and RESEARCH VALIDATED. Never infer scientific validity from smoke tests.
- Record task results, evidence, blockers and next tasks in PROJECT_PROGRESS.md;
  decisions in DECISION_LOG.md; architecture changes in TECHNICAL_WALKTHROUGH.md.
- Keep artifacts immutable in unique run directories under ignored data/derived/.
  Record source/config/input hashes, sample/class order, parameters and failures.
  Git does not back up data/. Never fabricate metrics or discard negative results.
- Lightweight CPU unit/smoke tests are authorized. Do not execute malware samples.

## Approval boundaries
- Obtain user approval before methodology changes (question, hypothesis, cohort,
  split, evaluation protocol), opening future-test processing/evaluation, costly
  GPU training or resource-intensive experiments, deleting important data/artifacts,
  irreversible data changes, or destructive Git/history operations.
- No commit or push without explicit user instruction. No malware execution.
- Investigate and report suspected leakage before continuing affected work.
- Do not turn provisional framework ideas into confirmed rules. Resolve scientific
  choices explicitly before implementing dependent experiments.
