# Adaptive Cognitive Composition Engine (ACCE)

Development stage 1, version 0.1.0. Scope: CF-02 (cognitive integration) and CF-04 (intent-to-action translation).

## Architecture and acquisition

`ix_zero/acce.py` provides a typed capability workspace, a bounded program composer, and an evidence-supervised intent binding. The catalog exposes pure host primitives, projections from grounded objects to values or relation labels, promoted word operators, validated signed scalar models, and validated relational transfer models. The host supplies signatures and capability adapters. The system enumerates compatible compositions (one projection per argument, then one root capability), filters them against teacher-supplied outcomes, and requires at least two training inputs and three fresh successful audits. A binding is not executable until these requirements are met. Surviving programs must agree at the actual query; no answer is returned if a surviving program is unavailable or the predicted outcomes disagree. Source invalidation is checked live. Relational dependencies are reconciled with the word-operator library. No external tools, arbitrary host code generation, or external actions are supported.

The IX interpreter executes `policies/acce.ix` to recommend inquiry, audit, completion, exhaustion, or revision. The Python host checks the recommendation; it cannot grant permission to bypass evidence requirements. The candidate catalog freezes on the first binding so that adding a capability cannot silently change the hypothesis space. The dispatch learner uses measured examples, not similarity between an intent string and a capability name. Host-provided ordering and arithmetic primitives are not discoveries of the dispatch learner. The trained word, scalar and relational sources are actual instances of the existing cognitive engines.

The `Object` record carries identity, kind, attribute, optional integer value and optional relation label. For the current projections, evidence identity is conservatively keyed by the projected value or relation label, not object identity or color. A relabeled scene cannot count as a new audit on the same inputs. Catalog construction and binding training are explicit operations. The new bindings are session-local; earlier engine checkpoints continue to work unchanged. Refreshing the catalog clears the current dispatch bindings. The supported typed signatures contain one or two inputs and an integer or relation output. The search limit is 256 compatible programs, the catalog limit is 32 capabilities, and the per-binding training and audit limits are 256 each.

## Interactive demonstration

From the repository root, execute:

```
python -m ix_zero.acce_demo
```

The demonstration learns from a fixed synthetic teacher corpus; it does not know arbitrary English. The scene contains A (red cube, value 0), B (green cube, value 1), C (red sphere, value 2), and D (blue cone, value 3). Try:

```
compare red sphere green cube
rule blue cone green cube
inspect red cube
inspect orange sphere
```

The first request returns an ordering value of 1 (first object greater), with `ordering(value,value)` as the selected program. The second returns 2 from a promoted word operator registered under the distinct name `word7`. The third returns the integer 0; the fourth abstains. The previous `language_demo` remains available as a historical baseline and does not use ACCE.

## Main console

Run `python -m ix_zero`. Train or load source engines before `cworkspace`. The following example uses host numeric capabilities; outcomes are supplied by the teacher. The objective name is arbitrary.

```
cworkspace
ccatalog
cnew ["rank",["integer","integer"],"integer"]
cobserve ["rank",["integer","integer"],[0,3],-1]
cobserve ["rank",["integer","integer"],[3,0],1]
cadvance ["rank",["integer","integer"],[[1,1],[1,3],[2,0]]]
caudit ["rank",["integer","integer"],[1,1],0]
caudit ["rank",["integer","integer"],[1,3],-1]
caudit ["rank",["integer","integer"],[2,0],1]
cstatus ["rank",["integer","integer"]]
cdispatch ["rank",[3,2]]
```

Use `csay [language_episode,text,scene,values,optional_relations]` to route a validated grounded utterance through a learned object-signature binding. Use `help` for all command signatures. Structured object arguments use JSON dictionaries with the fields `identity`, `kind`, `attribute`, and optional `value` and `relation`. Source learners are acquired with their existing commands. Only the new dispatch bindings must be retaught when a workspace is refreshed or a session restarts.

## Evaluation

```
python -m unittest discover -s tests -v
python -m ix_zero.acce_experiment --trials 24 --seed 140000 --check --output results/acce_campaign.json
python -m ix_zero.acce_demo --check
```

The 24-seed experiment fits separate language episodes on randomized vocabularies and shuffled role orders, and uses arbitrary semantic intent aliases. The existing computational source models are fixed across seeds. Each dispatch binding receives two training examples and three separate audit examples; the evaluation excludes those projected inputs from its held-out numeric queries. Supported language intentions are inspect (unary), comparison (binary), and learned word-rule application (binary). A composition-disabled ablation uses the same catalog but cannot compose object projections with binary capabilities. The exact-name baseline is the original numeric-language bridge, not a general-purpose language system. It cannot match the intentionally unrelated aliases. These baselines isolate integration mechanisms and do not establish superiority over other complete AI architectures.

| Evaluation | Correct | Incorrect | Abstained |
|---|---:|---:|---:|
| Supported language intents, composed | 3213 | 0 | 0 |
| Composition disabled | 99 | 0 | 3114 |
| Exact-name numeric bridge | 0 | 0 | 3213 |
| Structured scalar dispatch | 480 | 0 | 0 |
| Structured relational dispatch | 264 | 0 | 0 |
| Unsupported intent/arity combinations | 0 | 0 | 1755 |

There are 4968 held-out utterances in total across the supported and unsupported sets. Supported intent execution covers 3213 of them. The scalar and relational rows are separate structured API tests, not additional language tasks. Hidden-exception tests deliberately changed a comparison outcome after validation: 24/24 initial predictions were incorrect; 24/24 bindings blocked further outputs following corrective feedback. The engine did not learn the missing exception from that correction. Tests also check source invalidation, ambiguity, audit duplication, policy rejection, and CLI use. Raw results are in `results/acce_campaign.json` and the associated campaign summary.

The test suite contains 167 passing engineering tests; all 16 configured workflow commands were executed locally. Native Windows and remote GitHub CI are not established. The preceding historical experiments are preserved; they do not measure ACCE.

## Scope and remaining deficiencies

CF-02 and CF-04 have bounded implementations and a direct regression test for the partial comparison response. The remaining eight deficiency categories are not claimed resolved. The catalog, type system, root primitives, projections and composition depth are programmer-defined. General capability synthesis, recursive program composition, variable-length natural language, arbitrary representation discovery, persistent dispatch bindings, contextual relearning, and open-ended autonomous learning are outside this stage. The host-provided semantic labels and teacher outcomes constrain the acquisition problem. No general intelligence or human-level AGI claim follows from these experiments.
