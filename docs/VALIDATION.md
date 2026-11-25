# Executed evidence

## First campaign: raw predictions

Eight synthetic seeds, 12 held-out states each, 96 outputs per arm. Equal maximum of eight target training queries, not equal actual queries or compute. Reuse sees additional source experience. The target is a composition of two source generator programs, and may be unrepresentable at cold search depth. This intentionally tests compositional reuse, not broad out-of-distribution generalization.

| Arm | Correct | Incorrect | Abstained | Perfect tasks |
|---|---:|---:|---:|---:|
| Reuse + active queries | 62 | 22 | 12 | 5/8 |
| Cold + active queries | 39 | 9 | 48 | 2/8 |
| Reuse + passive queries | 55 | 0 | 41 | 2/8 |

The higher error count in the first reuse-active arm is a failure of the raw agreement criterion. Do not report only correct predictions. Full per-input records are in `results/campaign.json`.

## Second campaign: calibration gate

16 different seeds, 12 held-out states each. Five maximum training queries and three separately allocated calibration inputs per target. Predictions on final held-out inputs are committed before obtaining their labels. Calibration failures block output rather than selecting programs. Cold and reuse arms have different search-space sizes and prior data. Raw and guarded columns below use the same learned model within each arm.

| Arm | Raw correct / wrong / abstain | Gated correct / wrong / abstain | Training labels | Calibration labels | Source labels available |
|---|---|---|---:|---:|---:|
| Reuse active | 160 / 8 / 24 | 156 / 0 / 36 | 44 | 41 | 282 |
| Cold active | 103 / 41 / 48 | 84 / 0 / 108 | 38 | 31 | 0 |
| Reuse passive | 136 / 0 / 56 | 117 / 0 / 75 | 73 | 35 | 282 |

The zero observed wrong gated predictions do not demonstrate a zero future error rate. The gate also loses coverage. These small synthetic results support further investigation, not AGI or statistically established general superiority. All rows and target/source data are in `results/validation.json`.

## Challenge campaign

12 arbitrary random bijections on 25 states, 8 unseen inputs each. The raw model returned 8 wrong answers and no correct answers; gated predictions abstained on all 96. This is evidence of a capability failure on unstructured tasks, not successful learning. In a separate controlled change of known rules, 12/12 old action models blocked after an observed contradiction; fresh models fit 12/12 and matched 300/300 enumerated target states. The system requires explicit creation of a fresh epoch. On 120 generated reachable planning cases using two learned action models, all 120 proposed plans reached the goal under the separately implemented synthetic teacher. This does not imply real-world planning generality. See `results/challenges.json`.

## Engineering verification

The test suite tests grammar/space budgets, teacher cross-checks, ambiguity, unseen audits, contradiction handling, memory promotion, persistence, tamper detection, planning and CLI behavior. These are historical tests of the original learner. The current release verification is documented in [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md). No donor suite was run. No Windows execution, remote GitHub CI, independent expert validation or general-intelligence benchmark was run. No claim of formal verification, AGI or production readiness.
