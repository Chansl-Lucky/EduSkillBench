# Calibration-001 candidate review

Date: 2026-09-14
Generator: `Atria-Dawn-Preview`
Source: `artifacts/calibration_hinge_candidates.jsonl`

This is an internal data-quality review, not the human pairwise Judge calibration required by
the GRPO Go gate.

| Candidate | Automatic checks | Internal review | Decision |
|---|---|---|---|
| `...__001` fractions | schema valid; max official similarity 0.373 | Useful, but must verify the item tests conceptual selection of the divisor rather than mere recall of “keep-change-flip” | conditional |
| `...__002` bonding | schema valid; max official similarity 0.207 | The premise risks treating electronegativity thresholds as an absolute definition and may create science-validity ambiguity | reject |
| `...__003` elapsed time | schema valid; max official similarity 0.275 | Clear single decision, plausible base-100 misconception, appropriate rapid response and actionability requirements | accept for pipeline smoke |

No candidate is admitted to train/dev by this review. Candidate 003 may be used only to verify
the independent-pool rollout and scoring pipeline. Promotion into a frozen calibration set needs
subject-matter review and provenance/version metadata.
