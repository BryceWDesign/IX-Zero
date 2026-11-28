# IX-Zero

Experimental continual neural-symbolic cognitive architecture integrating executable skill acquisition, adaptive inquiry, controlled-language reasoning, source-scoped evidence, exact mathematics and finite program induction.

**Version 5.0.0 grounded research candidate** | Python 3.11+ | Source-available, noncommercial evaluation | No mandatory third-party runtime dependencies (optional browser extra)

IX-Zero investigates how a neural-symbolic learner can acquire executable knowledge from observations, reuse that knowledge across tasks, recognize the limits of its current evidence, and request more informative experiments. The implementation operates over bounded, evaluator-defined domains. It is not demonstrated artificial general intelligence (AGI), a general-purpose language model, or a production decision system.

## Grounded 5.0.0 research: learn action semantics, inquire and transfer

New executable mechanisms learn role-relative action rules from changes in observed object states, generate diagnostic interventions, transfer acquired rules across renamed attributes and reordered entity roles, and plan with learned action models. A relational grid solver searches output geometries and generated coordinate/connected-component features. All core mechanisms run without pretrained models or external AI inference.

| Executed diagnostic | Result |
|---|---:|
| Active action-language induction, four supported families | 3,072/3,072 vs random inquiry 160/3,072 |
| Unseen triggering equality | 768/768 vs conditional-expression-disabled 0/768 (abstained) |
| Executed plans using learned action models | 24/24 vs empty memory 0/24 |
| Cross-vocabulary semantic transfer | 3,072/3,072 vs from-scratch 0/3,072 under the small target label cap |
| Unary and ternary action learning | 1,536/1,536 shifted held-out transitions |
| Actual trusted local Chromium form | 64/64 held-out transitions, 2/2 goals |
| Six public ARC-AGI-1 diagnostic tasks | Relational 3/6, prior objects 1/6, original 0/6 |

All action families and object schemas are authored; word order, identifiers, role-relative expressions and mappings are inferred. Transfer tasks are structurally analogous by design. Candidate/query caps and realized compute/labels are not equivalent across all controls. The unsupported multiplication action remains 0/768 (abstained). Two repaired ARC tasks were previously exposed training tasks; the only public evaluation task remains unsolved. The three newly selected training tasks yield the same single success for both the prior object and relational solvers. This small diagnostic is not an official or representative ARC score. Full native donor integration, general semantic representation invention, open-ended architecture improvement, successful live web research, general Windows desktop competence, DreamCoder/PushGP comparisons and independent third-party verification remain open.

See [grounded research methods, reproduction and complete gap ledger](docs/RESEARCH_V5_GROUNDED.md), [raw evidence](results/v5_grounded/) and [release verification](docs/RELEASE_VALIDATION.md). The original v5 continuation and all previous evidence remain below for historical context.

## Expanded 5.0.0 research: iterative programs, revision and tool learning

The continuation adds generated state-transition programs, callable acquired programs, fresh-evidence program revision, object-centric grid induction, explicit standard-library tool discovery, conservative evidence auditing and actual local Chromium control learning. All 318 previous tests are retained; 339 total unit tests and 33 workflow commands passed locally on Python 3.13.5/Linux. GitHub-hosted matrix checks are still pending.

| Executed diagnostic | Result |
|---|---:|
| Positive-domain GCD, iterative vs expression-only | 2,784/2,784 vs 0/2,784 (abstained) |
| Positive-domain LCM, reuse vs disabled | 2,784/2,784 vs 0/2,784 (abstained) |
| Larger positive inputs | 960/960 per acquired operation |
| Signed/zero fresh-evidence GCD revision | 2,520/2,520 vs unchanged 881/2,520 |
| Signed/zero fresh-evidence LCM revision | 2,520/2,520 vs unchanged 994/2,520 |
| Independently implemented numerical tools | 3,840/3,840 correct, explicitly tool-assisted |
| Local Chromium goal execution with learned graph | 8/8 vs 1/8 blind random, 0/8 no-memory lookup |
| Public ARC diagnostic (3 tasks) | 0/3 solved; all abstained |

The signed/zero repair uses experimenter-selected boundary strata and is not general architecture self-modification. Positive-domain baseline failures, public ARC failures, common-mode corruption, and browser training costs are preserved. The new object and pixel grammars have different actual search sizes. Full native donor interoperability, DreamCoder/PushGP execution, general language/representation discovery, successful live Internet research and Windows desktop operation remain unresolved. See [continuation methods and roadmap ledger](docs/RESEARCH_V5_CONTINUATION.md) and [raw evidence](results/v5_continuation/).

The initial 5.0 research follows for historical context; it describes the preceding candidate, not the limits of the expanded build.

## Experimental research workspace (v5)

A new additive workspace unifies the retained native reasoning engine with executable finite-program exploration, disagreement-based active inquiry across multiple representations, lossless symbolic observations, independently scoped goal search, multi-observer audits, and direct read-only document/web retrieval. It runs without pretrained models or remote AI inference. The internet adapter is an actual HTTPS client, not a simulated API; successful live web retrieval has **not** been verified in the build environment. External tools do not establish intelligence by their existence.

Two authored 12-seed campaigns (24 seeds total) produced:

| Measure | Result |
|---|---:|
| Disagreement inquiry vs random inquiry, same candidate and label budgets | 432/432 vs 222/432 correct; 210 random-arm abstentions |
| Disagreement inquiry vs fixed-order enumerative inquiry | 432/432 vs 216/432 correct; 216 fixed-order abstentions |
| Second executable program, acquired macro reuse vs disabled | 816/816 vs 0/816 correct; 816 disabled-arm abstentions |
| Combined scalar, NAND and order-predicate families | 2,016/2,016 correct; GCD 672/672 abstentions |
| Scalar-only control across the same four families | 672/2,688 correct; 2,016 abstentions |
| External finite transition-table interface (authored randomized grammars) | 384/384 held-out predictions |
| Public ARC-AGI diagnostic | 0/1 exact tasks, one abstention |

The combined cross-mechanism run is **not compute- or realized-label-matched** to its scalar-only control. The fixed-order and random controls in the first two rows are matched in candidate and query counts. The macro experiment matches candidate budgets, not identical elapsed time or audit counts. The grammar tasks are not independently authored. Seeds vary partitions/order and randomized tables; they do not create 24 unrelated domains.

**Failure evidence:** with a targeted corruption of just one of 100 audit labels, the single-observer learner promoted an incorrect program in all 24 hidden-regime trials, producing 2,400 wrong held-out predictions. A three-observer majority protocol blocked that particular failure with one corrupted observer, but not with two or three correlated corruptions. It uses three times as many audit observations. All failures remain in the evidence. Majority is not ground truth, and source independence must be established by the experimenter.

Start the research console:

```sh
python -m ix_zero.research_cli
```

Example JSON-lines commands (the CLI reads one JSON object per line):

```json
{"command":"task","text":"Calculate 3*(2+7)"}
{"command":"research","subject":"Python","property_name":"pathlib documentation"}
{"command":"self_model"}
```

See [research methods and limitations](docs/RESEARCH_V5.md), [release verification](docs/RELEASE_VALIDATION.md), and [v5 raw evidence](results/v5/). The external ARC-format evaluator accepts the official JSON task directory. DreamCoder and PushGP have **not** been run; full native donor integration, broad grammar discovery, autonomous desktop operation and general architectural self-modification remain unproven. The retained v4 workspace and historical evidence follow below.

## Integrated native reasoning

The local reasoning workspace connects the retained continual neural-symbolic engine to controlled-language questions, source-scoped relational inference, exact rational arithmetic and affine equations, finite constraint search, executable sequence induction, and scoped goal planning. Acquired sequence programs can be composed with validated outputs from the existing engine; a contradiction in either source blocks dependent answers. The native workspace uses no pretrained language model or external API.

Across two authored six-seed campaigns:

| Evaluation | Result |
|---|---:|
| Sequence program acquisition, cumulative vs. reuse-disabled | 36/36 vs. 24/36 |
| Held-out sequence predictions, cumulative vs. reuse-disabled | 1,296/1,296 vs. 864/1,296 (432 abstentions) |
| Core-to-sequence composed predictions | 768/768 correct |
| Evidence and exact equation questions | 312/312 correct |
| Finite constraint problems | 216/216 matched exhaustive reference |
| Scoped planning checks | 60/60 |
| Workspace checkpoint replays | 36/36 |
| Corrupted audits and out-of-grammar controls | 24/24 withheld promotion |

These results are from authored, finite task families. They do not establish general language competence, broad software engineering, independent benchmark performance, or AGI. See [workspace methods and host protocol](docs/GENERAL.md) and [raw evidence](results/README.md).

```sh
python -m ix_zero.general_cli
python -m ix_zero.general_experiment --trials 6 --seed 810000 --check --output general-primary.json
python -m ix_zero.general_experiment --trials 6 --seed 820000 --check --output general-confirmation.json
```

The workspace accepts JSON-lines commands, for example `{"command":"ask","text":"Solve 3*(2*x-1)=x+4"}`. The original console and continual interfaces remain available.

## Continual cognitive development

The continual coordinator maintains a flat, replay-verified learning history across successive inquiries. It preserves verified executable skills, composes them into new candidate programs, retains frozen neural experts, trains new recognition heads on prior hidden features, selects extra boundary measurements, and requests fresh evidence after contradictions. All acquired rules remain subject to independent validation and scoped prediction. The learner proposes experiments; the host supplies measurements and grants access.

In the retained v3.0.0 continual campaigns, the following bounded results were observed:

| Evaluation | Result |
|---|---:|
| Cumulative five-law sequences, full reuse | 30/30 acquisitions |
| Matched control with executable-skill reuse disabled | 12/30 acquisitions |
| Common held-out predictions, full reuse vs. skill-disabled | 24,775/24,775 vs. 11,088/24,775 (13,687 abstentions) |
| Successive inquiries | 200/200 across two 100-inquiry runs |
| Held-out predictions in successive inquiries | 101,600/101,600 correct |
| Retention after checkpoint replay | 129,600/129,600 finite-domain predictions |
| Changed environment and sensor recovery | 6/6 |
| Additional probes detecting hidden corner exceptions | 6/6 |
| Adaptive two-gate vs. fixed one-gate acquisition | 6/6 vs. 0/6 |
| Flat journal replays | 32/32 |

The lateral-feature-disabled arm also acquired 30/30 laws; no lateral neural performance advantage was established. Modular addition outside the tested grammar was not acquired in any of six trials. The 100-inquiry runs reuse authored task families and compatible representations, not independent open-world challenges. The evidence does not establish AGI. See [continual architecture, methods and controls](docs/CONTINUAL.md) and the [raw evidence index](results/README.md).

A separate six-seed composition experiment acquired a third-level program by combining two second-level learned skills under a two-gate budget. All six dependency chains passed; the skill-disabled control acquired none of the six final laws. The full arm was correct on 24,576 common held-out inputs and passed 6,144 finite planning checks. One earlier control law was incorrectly promoted, producing 382 wrong common-holdout predictions. The failure is included in the [stack evaluation](docs/CONTINUAL.md#three-level-skill-composition-and-goal-search) and raw evidence.

Run the complete local experiment or host interface:

```sh
python -m ix_zero.continual_experiment --trials 3 --episodes 100 --check --output continual.json
python -m ix_zero.continual_cli
python -m ix_zero.stack_experiment --trials 6 --check --output stack.json
```

## Integrated neural-symbolic inquiry

The retained single-inquiry runtime uses three small, locally trained neural networks that learn observation decoding, outcome prediction, and inquiry guidance. Their outputs assist a symbolic learner over finite word operations and spatial transition rules. The learner selects discriminating experiments, requests additional environments, verifies candidate programs with separate measurements, and abstains when verification fails. Neural predictions cannot authorize executable knowledge. No pretrained model, external API, GPU, or remote inference service is required.

In the retained v2.0.0 reference experiments, across two authored six-seed campaigns, the hybrid configuration produced the following results:

| Measure | Observed result |
|---|---:|
| Validated word and grid acquisitions | 24/24 |
| Held-out symbolic word predictions | 13,032/13,032 correct |
| Held-out symbolic spatial predictions | 1,056/1,056 correct |
| Spatial planning checks | 144/144 correct (133 paths, 11 unreachable goals) |
| Cross-family weight transfers | 24/24 validated, no new calibration labels |
| Held-out predictions after transfer | 14,088/14,088 correct |
| Final checkpoint replays across evaluated arms | 184/184 |
| Negative controls | 16/16 blocked or invalidated |

A calibrated non-neural centroid decoder and an unguided hybrid control also acquired all tested laws. Guided inquiry used 509 requests versus 507 for unguided inquiry; no sample-efficiency advantage was established. Raw neural outcome predictions were substantially less accurate than audited symbolic predictions (1,502/13,032 word and 56/1,056 spatial). These are finite, synthetic, developer-authored tasks, not external AGI benchmarks. See [hybrid methods and results](docs/HYBRID.md) for protocol, control arms, reproducibility and limitations.

## Research contributions

- **Program induction:** discover finite executable operations from examples rather than receiving target expressions.
- **Active inquiry:** select observations that discriminate among competing hypotheses; abstain when evidence is insufficient.
- **Adaptive Epistemic Inquiry (AEI):** preserve hypotheses that are indistinguishable in one environment and request another environment when it can resolve the ambiguity.
- **Cumulative skill acquisition:** reuse validated programs as synthesis macros under a bounded search budget, with fresh evaluation of each acquired capability.
- **Grounded composition:** connect learned language, finite relational representations, numeric operations, and simulated actions through audited bindings.
- **Evidence lifecycle:** validate within declared scope, quarantine contradicted knowledge, invalidate dependent capabilities, and replay learning journals.

The implementation retains these mechanisms in a developmental coordinator, with IX policies directing bounded learning decisions. See [research design](docs/RESEARCH.md), [developmental architecture](docs/DEVELOPMENTAL.md), and [AEI](docs/EPISTEMIC.md) for the algorithms and their limits.

## Retained epistemic research results

The AEI evaluation comprises two disjoint six-seed campaigns using the same developer-defined synthetic task families. The engine receives observations and unlabeled input pools, not the hidden target expressions.

| Challenge | Observed outcome across both campaigns |
|---|---:|
| Double-shift acquisition | 12,264/12,264 correct |
| Four-shift acquisition | 12,264/12,264 correct |
| Constant-zero discrimination | 12,264/12,264 correct |
| Learned macro reuse | 12,264/12,264 correct |
| Matched reuse-disabled control | 12,264 abstentions |
| Contradictory audit control | 12,264 abstentions |
| Later contradiction invalidation | 12/12 laws quarantined |
| Denied environment access | 12/12 withheld promotion |
| Environment trial journal replay | 72/72 |

In the double-shift trials, the learner requested a four-bit environment after determining that its two-bit environment could not distinguish the surviving hypotheses. In the four-shift trials it instead requested a five-bit environment. Environment selection is derived from hypothesis disagreement; the evaluator controls access and supplies measured outcomes. The paired reuse comparison uses a common synthesis budget and matched evidence partitions, with the reuse-enabled arm receiving an acquired macro.

These results measure performance on finite mathematical fixtures, not broad generalization or human-level intelligence. A hidden exception outside the observation pools remains undetectable from the supplied evidence. Earlier experiments include wrong predictions, failed gates, and abstentions; the complete reported research scope is described in the [evidence index](results/README.md).

## Quick start

From the repository root, with Python 3.11 or later:

```sh
python -m ix_zero
```

On Windows, `py -3 -m ix_zero` or `Start-IX-Zero.cmd` can be used. Type `help` in the console. No API key, network connection, GPU, or installed package is required for the source-tree runtime.

To run the hybrid evaluator, open its local JSON-lines interface, or inspect the retained developmental curriculum:

```sh
python -m ix_zero.hybrid_experiment --trials 2 --check
python -m ix_zero.hybrid_cli
python -m ix_zero.developmental_cli --demo
python -m ix_zero.epistemic_experiment --trials 2 --check
python -m unittest discover -s tests -v
```

### Teach a finite rule

In the console, enter the following observations of a hidden transformation, without giving the engine its expression:

```text
new puzzle 2 5
teach 0 0 -> 1 0
teach 1 2 -> 3 1
teach 3 4 -> 0 3
ask
predict 2 1
audit 0 1 -> 2 0
audit 1 0 -> 1 1
audit 4 2 -> 3 4
validated 2 1
promote
```

The console exposes bounded learning commands, not unrestricted natural-language conversation. This example uses the original finite-state learner; see the [AEI API](docs/EPISTEMIC.md) for cross-environment inquiry.

## Reproduce the hybrid experiments

```sh
python -m ix_zero.hybrid_experiment --trials 6 --seed 410000 --check --output hybrid-primary.json
python -m ix_zero.hybrid_experiment --trials 6 --seed 420000 --check --output hybrid-confirmation.json
```

Reference records: [primary](results/hybrid_primary.json), [confirmation](results/hybrid_confirmation.json). For the host protocol and raster input, see [hybrid API](docs/HYBRID.md).

## Reproduce the AEI experiments

Use a separate working copy for experiments that write into `results/`, so that reference evidence is not overwritten.

```sh
python -m ix_zero.epistemic_experiment --trials 6 --seed 310000 --check --output aei-primary.json
python -m ix_zero.epistemic_experiment --trials 6 --seed 320000 --check --output aei-confirmation.json
```

Reference records: [primary campaign](results/aei_epistemic_campaign.json) and [confirmation campaign](results/aei_epistemic_confirmation.json). Each contains input partitions, predictions, decisions, and replay outcomes. [Release verification](docs/RELEASE_VALIDATION.md) describes the local engineering checks and reproducibility status. The [GitHub Actions workflow](.github/workflows/ci.yml) defines Linux and Windows checks; hosted execution is not claimed until the workflow actually runs.

## Repository structure

| Path | Contents |
|---|---|
| `ix_zero/` | Learners, developmental coordinator, console, evaluators, IX policies, and incorporated components |
| `tests/` | Engineering regression and failure-mode tests |
| `tools/` | Historical evaluation and native comparison tools |
| `docs/` | Research methods, protocols, limitations, and provenance |
| `results/` | Raw and summarized scientific evidence, including historical negative results |

## Technical documentation

| Topic | Reference |
|---|---|
| Integrated native reasoning and program induction | [Workspace](docs/GENERAL.md) |
| Continual cognition and progressive learning | [Continual architecture](docs/CONTINUAL.md) |
| Integrated neural-symbolic inquiry | [Hybrid architecture](docs/HYBRID.md) |
| Epistemic inquiry and cross-environment learning | [AEI](docs/EPISTEMIC.md) |
| Developmental scheduling and learned prerequisites | [Developmental curriculum](docs/DEVELOPMENTAL.md) |
| Context and contradiction recovery | [Contextual cognition](docs/CONTEXTUAL.md) |
| Representation synthesis and abstraction | [Representation](docs/REPRESENTATION.md) |
| Recursive grounded objectives | [Composition](docs/COMPOSITIONAL.md) |
| Capability composition and binding | [ACCE](docs/ACCE.md) |
| Grounded language and relational transfer | [Language](docs/LANGUAGE.md), [Analogy](docs/ANALOGY.md) |
| Strategy selection and operator discovery | [Strategy](docs/STRATEGY.md), [Operators](docs/OPERATORS.md) |
| Comparative and unified learning | [Comparative development](docs/SUPERIORITY.md), [Unified workspace](docs/UNIFIED.md) |
| Earlier discovery and IX policy integration | [Discovery](docs/DISCOVERY.md), [IX instructions](docs/IX_INSTRUCTIONS.md) |
| Historical evaluation and external task compatibility | [Historical evaluation](docs/FINAL_EVALUATION.md), [Validation](docs/VALIDATION.md) |
| Scientific context and donor provenance | [Research](docs/RESEARCH.md), [Donor review](docs/DONORS.md), [Notice](NOTICE.md) |

## Limitations

The hypothesis grammar, computational budgets, available representations, and environment interfaces are supplied by the implementation. The hybrid runtime uses three small supervised networks and a finite symbolic grammar. The continual runtime extends this with bounded frozen neural experts, lateral feature connections, skill reuse, recovery and flat checkpoint replay; it does not establish unbounded continual learning or superior neural performance. The integrated workspace adds controlled language, structured evidence and finite reasoning, but it does not provide general-purpose perception, broad natural-language competence, a physical sensor driver, autonomous deployment, unrestricted continual learning, or a general neural world model. AEI does not invent unrestricted environments, acquire general natural language, initiate arbitrary open-world objectives, or rewrite its architecture. Historical results may require the corresponding earlier engine for exact reproduction. Full independent benchmark evaluation, native donor comparisons, and general intelligence remain unestablished. No claim of universal correctness, research priority, consciousness, security certification, or production readiness is made.

## License

IX-Zero is source-available for noncommercial research and technical evaluation, not open source. See [LICENSE](LICENSE), [component licensing](LICENSE-COMPONENTS.md), and [NOTICE](NOTICE.md). Incorporated components retain their notices and applicable terms. Commercial, production, government, and other restricted uses require the applicable separate permissions.
