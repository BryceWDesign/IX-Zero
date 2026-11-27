# Integrated native reasoning workspace

IX-Zero exposes a local, bounded reasoning workspace that combines the retained continual neural-symbolic engine with source-scoped relational inference, exact arithmetic, finite constraint solving, sequence program induction and scoped goal search. All mechanisms run on CPU using Python's standard library. This is an experimental integration, not a general-purpose language model or a demonstration of AGI.

## Architecture and authority

`general.General` routes controlled-language questions and structured host commands. The continual learner (`continual.Continual`) retains its existing acquisition, neural calibration, recovery, composition and evidence gates. The new sequence learner (`general.Programs`) has its own finite typed program grammar, scoped validation and dependency graph. A compound request may consume a validated prediction from the continual engine, then apply an independently acquired sequence program. Both skills must remain live; invalidation of either blocks the compound answer. The compound answer is a derived prediction, not an independently audited law. No arbitrary Python or generated code is executed, and neither workspace may operate external actuators or initiate network access.

The `General` journal records document ingestion and retraction, program measurements, challenges and core host events. Checkpoint loading reconstructs state by replay and compares a content fingerprint; this is an integrity and determinism mechanism, not an authentication scheme. The local interface assumes a trusted host. Host assertions and measured outcomes are not authenticated for factual truth. Checkpoints are limited to 60 MB and 4,096 workspace events; the retained core has its own additional limits. Checkpoints from the original continual interface remain separate from workspace checkpoints.

## Controlled language and evidence

A document is a sequence of explicit assertions, one per line. Identifiers are ASCII names beginning with a letter; supported predicates are `inside`, `before`, `kind_of`, `connected_to`. For this formal language only, the first three are transitive. `connected_to` is directed and nontransitive. Negated assertions use `is not`. Each document has a named source; answers return supporting source identifiers and paths. Direct conflicting positive and negative assertions are reported as conflicts. A directly disputed premise is excluded from derived paths; a negative assertion conflicting with a derived path is reported as a conflict. Retracting a document recomputes support on the next query. No assumption of closed-world truth is made; lack of proof is unknown, not false. The workspace does not infer the truth of a real-world document or automatically extract assertions from unrestricted prose.

Example, using the JSON-lines interface (`python -m ix_zero.general_cli`):

```jsonl
{"command":"ingest","source":"inventory","text":"sensor is inside case.\ncase is inside cabinet."}
{"command":"ask","text":"Is sensor inside cabinet?"}
{"command":"ask","text":"Which items are inside cabinet?"}
{"command":"ask","text":"If sensor is inside cabinet then calculate 4*(3+2)"}
{"command":"ingest","source":"inspection","text":"case is not inside cabinet."}
{"command":"ask","text":"Is sensor inside cabinet?"}
{"command":"retract","source":"inspection"}
{"command":"ask","text":"Is sensor inside cabinet?"}
```

The first question returns a two-edge proof; after the conflicting assertion the same question is unknown. The directly disputed assertion itself returns a conflict. The conditional computation runs only when its premise has uncontested support. Unsupported questions return `unsupported`; malformed inputs return an error through the CLI. Natural-language support is a controlled grammar, not fluent dialogue or open-domain comprehension. Source collection is local and host-directed.

## Mathematics and finite constraints

`Calculate` evaluates exact rational expressions using a restricted Python syntax parser and an explicitly implemented interpreter. `Solve` handles one affine equation in `x` (including fractional coefficients, no solution and all-rational solutions). No `eval`, arbitrary calls, attribute access or exponentiation is permitted. Nonlinear or undefined expressions are rejected. Expression length, AST nodes and rational bit sizes are limited. For a unique equation solution, both original sides are evaluated and returned.

```jsonl
{"command":"ask","text":"Calculate 2/3 + 7/9"}
{"command":"ask","text":"Solve 3*(2*x-1)=x+4"}
{"command":"solve_constraints","domains":{"a":[1,2,3],"b":[1,2,3],"c":[1,2,3]},"constraints":[{"op":"distinct","vars":["a","b","c"]},{"op":"lt","vars":["a","b"]},{"op":"lt","vars":["b","c"]}]}
```

Constraint search supports finite integer domains and `distinct`, `lt`, `eq`, `neq`, `sum_eq` and `diff_eq`. It forward-filters values, branches on the smallest viable domain, and checks full assignments. It returns unique, ambiguous (two witnesses), unsatisfiable (exhaustive search) or budget (incomplete search). Limits: ten variables, sixteen values per domain, thirty-two constraints, fifty thousand visited nodes. This is finite combinatorial search, not a learned general mathematician.

## Executable sequence skill acquisition

The sequence interpreter contains fifteen primitive transformations over bounded integer lists (reverse, sort, stable uniqueness, parity filters, sum, length, arithmetic maps and prefix sums). A sequence is at most sixteen integers with magnitude at most one million. A task supplies a finite unlabeled input scope, disjoint training and audit input pools and a search budget of one or two primitive/macro calls. Candidate programs are enumerated and deduplicated by their behavior across the declared scope. The learner selects training observations that maximize pairwise disagreement. It cannot use reserved audit labels to select among ambiguous hypotheses. If the training pool cannot distinguish surviving behaviors it abstains. Once a single behavior remains, every reserved audit is consumed and must agree before promotion. Held-out inputs are within declared scope but outside the training and audit pools; their labels are never given to the learner.

Validated programs are callable in subsequent synthesis at a one-call cost. A new program composed of acquired skills receives its own fresh audits; dependencies are recorded and invalidated transitively after contradiction. Programs remain restricted to the specified prediction scope. Reuse during synthesis may evaluate a prior expression on new inputs as a hypothesis, not as established truth. The expanded program size is capped at 32 primitive calls, and the skill library at 32 entries. A scope contains at most 256 distinct inputs. Finite audits cannot rule out hidden exceptions or out-of-grammar target functions that agree with all supplied evidence.

The `begin`, `next`, `answer`, `deny`, and `challenge` commands implement the host-driven acquisition protocol. `begin` fields are `key`, `scope`, `train`, `audit`, and optional `budget`. `next` requests either an `observe` or `audit` measurement; the host responds using `answer` with an integer list `y`. `next` eventually returns `validated` or `failed`. A host may respond with `deny` instead of a measurement; the inquiry fails without promotion, and a new inquiry may be started. `challenge` takes `key`, `x`, and `y` and invalidates the law and its dependents if the observation contradicts the prediction. The scope and partitions contain inputs only. The external evaluator demonstrates the complete protocol without supplying target expressions to the learner.

Language commands `Apply <skill> to [integers]`, `Apply <skill> to result of calculate <expression>`, and `Plan from [integers] to [integers]` invoke validated skills. Planning uses bounded breadth-first search with at most eight skill applications, checking the scope and liveness of each transition. Plans are proposals, not executed physical actions. An arithmetic bridge requires an integer result. The compound command `Apply <skill> to core <episode> at width <w> with <a> and <b>` uses a live word-domain prediction from the retained continual engine as the new skill's input. The `core` command accepts a nested `request` following the retained continual host protocol ([Continual](CONTINUAL.md)); the workspace disallows nested reset/load/save operations. Unsupported or invalidated links produce unknown answers.

Save using `{"command":"save","path":"ix-zero-general-state.json"}`; restore with `python -m ix_zero.general_cli --load ix-zero-general-state.json`. Use trusted local paths and keep private checkpoints outside source control.

## Evaluation

Two authored six-seed campaigns use disjoint seeds, generated input sequences and identifiers, held-out observations, and independently coded target functions. The evaluator compares matched two-call budgets with and without acquired-program reuse. In each trial, the first rule reverses a sequence and increments each element; the second retains even prefix sums. A third target composes these two rules. The learner receives only measurements. The reuse-disabled arm cannot represent the third target within the tested budget and abstains. The experimental protocol is not an independently blinded AGI benchmark, and it does not establish that this architecture can synthesize arbitrary programs.

A second integrated experiment per seed induces a NAND rule through the unchanged continual learner, separately induces an increment sequence program, and queries their composition using controlled language over all 64 three-bit operand pairs. The composed query is withheld after a contradiction to the core rule. Additional tasks test exact affine equation solving, document inference and retraction, finite constraint assignments against exhaustive enumeration, reachable and unreachable sequence plans, and arithmetic-to-program composition. Negative controls introduce corrupted audit measurements and a cubic target outside the tested grammar. No external knowledge corpus or pretrained language model is included.

| Measure | Both campaigns |
|---|---:|
| Acquisitions with skill reuse | 36/36 |
| Acquisitions without skill reuse | 24/36 |
| Third program acquisitions, full vs. control | 12/12 vs. 0/12 |
| Held-out sequence predictions, full | 1,296/1,296 correct |
| Held-out sequence predictions, control | 864 correct, 432 abstentions, 0 wrong |
| Core-to-sequence composed predictions | 768/768 correct |
| Evidence and affine questions | 312/312 correct |
| Finite constraint cases | 216/216 matched exhaustive reference |
| Scoped plan checks | 60/60 |
| Workspace checkpoint replays (all three arms) | 36/36 |
| Contradiction blocks compound answers | 12/12 |
| Sequence dependency invalidation | 12/12 |
| Corrupted audits and unsupported cubic targets | 24/24 withheld promotion |

Run:

```sh
python -m ix_zero.general_experiment --trials 6 --seed 810000 --check --output general-primary.json
python -m ix_zero.general_experiment --trials 6 --seed 820000 --check --output general-confirmation.json
```

Raw records: [primary](../results/general_primary.json), [confirmation](../results/general_confirmation.json). Each includes oracle outputs, learner traces, control outcomes and per-case results. Input scope is finite and supplied by the evaluator. The core-composition predictions are checked across the full three-bit domain, not described as held-out core observations. The evidence questions use generated identifiers rather than factual general knowledge. The planning checks use simple induced increment transitions. Neither the observed successes nor passing engineering checks establish general-purpose language understanding, broad software engineering, external benchmark performance, model-level recursive reasoning, physical perception, or open-world intelligence. The implementation does not include a pretrained model connector or a new language-model training system.
