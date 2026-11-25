# Contextual hypothesis generation and counterexample-guided recovery

## Scope

ACCE stage 4 addresses bounded aspects of CF-06 (generative cognition), CF-08 (metacognitive recovery), and CF-09 (contextual adaptability). It does not implement open-ended conversation, arbitrary hypothesis invention, or unrestricted context inference. The language engine and all preceding modules remain unchanged.

`ix_zero.contextual.Domain` defines a finite space of three to eight integer labels and two to four named contexts. The context identifier is an observed input, not inferred from raw text. An `Episode` accepts one base and up to three alternative binary integer capabilities. These are validated upstream in the supplied experiments. The host must provide honest capability availability functions and observed outcomes. Candidate programs comprise an incumbent and one context-conditioned substitution using an acquired alternative or a constant. Conditions include ordering, equality, one coordinate equaling a domain value, and one exact pair. The learner enumerates this family, deduplicates identical complete-domain tables, and filters against observations. It cannot execute arbitrary source code or evaluate a teacher function.

An inquiry is constructed from an unlabeled candidate pool. The engine chooses the input that maximizes pairwise disagreement among distinct candidate behaviors, using integer score `N² - sum(count(y)²)`. Its question text is a fixed rendering of the computed context and inputs; the content, candidate outcomes and selection are determined by the version space. The IX interpreter executes `policies/contextual.ix` to select inquiry, audit, completion, exhaustion or revision. The Python host independently checks action prerequisites.

## Evidence lifecycle

- Training requires distinct inputs; at least six are required before prediction. Validation requires at least three fresh audits. An ambiguous input abstains even after audits.
- Audits never refine the current model, and freeze further training. A failed audit blocks the episode. A validated contradictory prediction can be reported with `contradict`, including on a previously seen input.
- A contradiction quarantines the entire previous contextual episode. `recover()` constructs a new expanded version space, replays historical training and successful audits as training, then incorporates the counterexample. Old audit status is not inherited. New observations and fresh audits must be supplied before use. When an external evaluator establishes a context drift, `recover(changed_context=True)` drops earlier observations in that context; the engine does not independently identify the time of a drift. Recovery after a failed audit before a validated contradictory prediction is not implemented.
- A capability exported from an episode remains tied to that episode and a particular explicit context. Source retraction or episode quarantine blocks downstream ACCE dispatch. All source capabilities must remain available, even when a surviving hypothesis does not use one; this is deliberately conservative.
- Contextual knowledge is session-local, with no checkpoint persistence in this stage. The finite tables are snapshots of upstream capabilities and assume those capabilities remain behaviorally stable while available.

## Reproduction

```sh
python -m unittest discover -s tests -v
python -m ix_zero.contextual_experiment --trials 24 --seed 196000 --check --output contextual.json
python -m ix_zero.contextual_demo --check
python -m ix_zero.contextual_demo
```

In the interactive demo, `questions`, `hypotheses`, and `diagnosis` expose the generated recovery trace. Supported recursive sentences include `rule green cube blue cone` and `compare inspect red sphere rule green cube blue cone`. The demo includes a synthetic teacher and displays the questions that the learner generated during acquisition. The Python `Episode.inquiry(pool)` API can generate live questions from a caller-supplied unlabeled pool; callers provide the observed answers using `observe`. No pretrained language model is used.

## Evaluation protocol

Each seed trains real three-bit XOR, AND and OR operators using the existing synthesis engine. For each of four exception families, the incumbent is calibrated on 12 ordinary-context training inputs and three distinct ordinary-context audits. An evaluator selects a previously unseen changed-context input on which the incumbent is wrong and provides a counterexample. The old episode is blocked; both recovery arms start from the same historical evidence and counterexample. The active arm requests informative examples from a reserved pool; the passive arm receives the same number of additional labels per episode, in shuffled pool order (changed-context entries first). Both arms use the same eight reserved audit inputs (four changed and four other contexts). Remaining held-out inputs exclude the entire inquiry pool, not only queried inputs. The held-out data do not affect learning. The source acquisition costs are shared, not included in the target label counts. The passive baseline is one specific sampling strategy, not an optimized active learner.

The four supported families are a context-wide operator substitution, a context-and-ordering condition, a context-and-coordinate condition, and an isolated pair exception. They are intentionally expressible by the supplied candidate family. The unsupported control requires simultaneous substitutions in two contexts and is outside that family. The evaluator supplies enough contradictory examples to exhaust the representation. It is an explicit failure control, not a prediction test with an equivalent training budget.

In the first 24-seed campaign (`results/contextual_campaign.json`), 96 incumbent errors were observed and quarantined. The active arm generated 243 informative questions. Active recovery: 11,512 correct, 0 incorrect, 8 abstentions. Passive recovery: 5,538 correct, 0 incorrect, 5,982 abstentions. Unsupported control: 2,712 abstentions and no correct or incorrect outputs. Raw predictions, questions, audit inputs and episode status are included. These are synthetic developer-designed cases, not independent AGI benchmarks. Zero observed errors is not a correctness guarantee; hidden conditions outside the hypothesis family can still invalidate predictions. No evidence establishes unrestricted creative cognition or autonomous open-world recovery.

The disjoint follow-up campaign (`--seed 197000`, `results/contextual_fresh_campaign.json`) produced 11,505 correct, 0 incorrect, 15 abstentions under active recovery and 4,723 correct, 0 incorrect, 6,797 abstentions under passive recovery. All 96 deliberately selected initial counterexamples were quarantined; 238 informative questions were generated. The unsupported control again produced 2,712 abstentions. The first campaign was rerun and the complete JSON output compared byte-for-byte. Across both campaigns, active recovery recorded 23,017/23,040 correct predictions, versus 10,261/23,040 for passive recovery.
