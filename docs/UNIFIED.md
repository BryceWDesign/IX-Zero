> Historical milestone record. The subsequent [comparative development milestone](SUPERIORITY.md) addresses the specific cross-modal failure family described below; its independent evidence and remaining limitations are documented separately.

# Unified developmental workspace — v0.1.0

This milestone integrates selected mechanisms rather than concatenating repositories. It retains the original learner, expression-discovery learner, IX interpreter, and their tests. The Python unified workspace operates on finite, closed-world symbolic fact sets and optional bounded integer vectors. It is not general intelligence, a language model, a self-modifying agent, or an external actuator.

## Architecture and provenance

- **IX-Sally:** 11 actual Python modules (plus two package initializers) from the supplied repository are included under `_sally_vendor`, with import namespace adjustments. `OpenGoalGenesis`, `GoalArbiter`, `RelationalTransferEngine`, and `MetaLearningController` are called by the unified workspace, not mocked. The supporting classes and dependencies are retained. Goal generation searches a bounded graph of learned transitions. A donor heuristic operates on graph-local integer IDs and is not a physical risk assessment. Goal arbitration receives evaluator-provided priority and utility. Structural analogy proposes a target fact, not proof of a target effect. Meta-learning ranks curriculum options by historical accuracy/abstention utility; it never overrides validation.
- **IX-BlackFox-AGI:** `relational.jl`, `procedures.jl`, `inquiry.jl`, and `metacognition.jl` informed Python implementations of role-lifted add/remove effects, conjunctive prerequisite induction, action-sequence abstraction, and inquiry. This is a selective reimplementation, not a Julia module port with guaranteed equivalence.
- **IX-AGI:** `hemisphere.jl`, `callosum.jl`, and `organism.jl` informed the separate full-context evidence lane and disagreement checks. This is not a copy of the full organism, a neural hemisphere implementation, or a claim of independent observations.
- **IX-Zero / IX-Main:** the existing discovery engine provides learned vector transformations; the original IX interpreter executes `policies/unified.ix` for planning, inquiry, goal generation, recovery proposal, or exhaustion. The Python host checks the policy response. This policy cannot execute external actions, approve itself, or synthesize host code.

`unified_sally_provenance.json` records source and adapted file hashes; `unified_donor_provenance.json` records supplied archive and selected source hashes. The Sally license is preserved verbatim; see [component licensing](../LICENSE-COMPONENTS.md) for the release-specific grant. The previous donor reviews describe earlier milestones, not the current vendor inventory.

### Learning and evidence

1. A grounded symbolic observation is a full before snapshot, a named action and its argument objects, and a full observed after snapshot. The learner infers effects, not supplied rules. It canonicalizes argument roles, learns local preconditions, and maintains a separately fitted full-context evidence lane. When both lanes offer conflicting predictions, the combined learner abstains. Nonlocal effects disable the local-only representation; a supported full-context prediction may still be usable.
2. Symbolic planning requires three passing fresh symbolic audits per action schema. Training freezes at the first audit. A numeric action requires a bound discovery episode and passing numeric audits. Joint planning additionally requires three passing joint audits on new grounded symbolic inputs. These are evidence gates, not calibrated safety guarantees. Both lanes use the same supplied observation source. Different action schemas and grounded contexts can still exhibit unobserved interactions.
3. Measured contiguous traces support variable-bound procedure patterns. At least two distinct bindings are needed before reuse. Every constituent action is predicted again under the current gates; a macro does not override quarantine. `plan` returns proposed simulated paths, never dispatches them.
4. Falsifying feedback quarantines a symbolic schema or joint binding. An explicitly requested symbolic recovery creates a new epoch, preserving historical observations and making a bounded depth-2 decision-tree representation available. Earlier data may become training evidence, but new audits are mandatory and prior audit inputs cannot be reused to validate the new epoch. If contradictory outcomes occur for the same exact grounded input, the new epoch starts without reused data. A cross-modal contradiction remains blocked: symbolic recovery is not a solution to coupled numeric/symbolic dynamics.
5. The joint workspace shares the discovery journal and checks a digest of its historical prefix when replaying joint audits. Numeric retraction blocks dependent predictions. Local checkpoints are replay-validated and atomically replaced individually; two-file saves are not a single transaction and checksums are not signatures. Back up both files together. No real-world or autonomous execution is provided.

## Run

From the extracted root:

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m ix_zero.unified_experiment --trials 12
py -3 -m ix_zero --state unified-demo.json
```

The last command opens the interactive console. Example symbolic learning session, one command per line:

```text
uobserve [[["ready","a"]],["step","a"],[["done","a"]]]
uobserve [[["ready","b"]],["step","b"],[["done","b"]]]
uobserve [[],["step","a"],[]]
uobserve [[],["step","b"],[]]
uaudit [[["ready","c"]],["step","c"],[["done","c"]]]
uaudit [[["ready","d"]],["step","d"],[["done","d"]]]
uaudit [[],["step","e"],[]]
upredict [[["ready","f"]],null,["step","f"]]
ugoal {"facts":[["ready","f"]],"actions":[["step","f"]]}
uplan {"facts":[["ready","f"]],"actions":[["step","f"]],"goal":{"present":[["done","f"]]}}
ustatus
usave
exit
```

`null` means symbolic-only. To use vectors, teach and audit a discovery episode using `dnew`, `dteach`, `daudit`, bind its name with `ubind`, and provide independent joint evidence with `ujaudit`. The executable `build_joint` fixture in `ix_zero/unified_experiment.py` is a complete example of the numeric and symbolic workflow. `help` lists the commands. Unknown requests are not fabricated. A fresh workspace is needed to change a joint binding. Recovery of a symbolic model uses `urecover ["step","f"]` after a detected contradiction.

## Executed evaluation

Python 3.13 on Linux; 12 seeds (100–111). The simulator supplies labels for fitting and audits, and independently replays plans. Final held-out evaluations occur after fitting/audits. These are developer-designed synthetic task families, not an external blind benchmark. No assertion of general superiority over a complete donor system. Ablations use the same fixture data; model classes, computational effort, and available constraints differ. Numeric state values are sampled from the same finite domain; the held-out joint cases use new object identities and combinations, not a new numeric domain.

| Symbolic arm | Correct (local / nonlocal) | Wrong | Abstentions (local / nonlocal) |
|---|---:|---:|---:|
| Local | 288 / 0 | 0 | 0 / 288 |
| Counter | 144 / 288 | 0 | 144 / 0 |
| Combined | 288 / 288 | 0 | 0 / 0 |

| Planning arm | Successful replay | Incorrect replay | Abstained |
|---|---:|---:|---:|
| Unified | 96/96 | 0 | 0 |
| Unified, no macros | 96/96 | 0 | 0 |
| Relational only | 0/96 | 96 | 0 |
| Numeric only | 60/96 | 36 | 0 |

The latter two ablations deliberately plan toward only one of two required constraints. This tests the utility of integration, not their standalone competence. The macro arm expanded 2,520 states versus 2,712 without macros; expansion count is not elapsed runtime or a compute-matched efficiency measure. All 96 proposed self-generated goals were achieved on simulator replay. All 12 renamed structural analogies identified and reached the intended role, and all 12 mismatched structural controls were rejected. The goal tests verify reachable self-generated outcomes, not meaningful intrinsic motivation or self-awareness.

### Falsification and recovery

All 12 hidden symbolic-context cases and all 12 cross-modal-interaction cases were incorrect before their counterexamples were provided. All 24 were quarantined after feedback, then abstained. These failures are included in the results, not discarded from a combined success percentage. The subsequent symbolic recovery experiment used two new observations and four fresh audits in each of 12 epochs; all 48 new audits passed and all 96 held-out cases were correct. The cross-modal interaction case remains unresolved. Recovery uses a predefined contextual tree family, not arbitrary ontology invention or unconstrained learning.

`results/unified_campaign.json` contains actual aggregate counts. The historical build passed 79 tests and eight local workflow commands. The `results/regression/` directory retains selected historical smoke datasets separately from the full research campaigns. Original campaign records remain intact.

## Remaining gaps

No true AGI or human-level generalization demonstrated. The workspace does not learn arbitrary sensory representations, discover an unbounded vocabulary, ground free-form natural language, conduct open-world continual learning, autonomously improve its own implementation, or independently infer joint symbolic/numeric interactions. It does not yet implement all of Sally's developmental mechanisms or BlackFox's complete Julia system. Candidate actions and evaluator evidence are supplied by the caller; selection heuristics and search limits are hand-designed. No direct common-benchmark rerun of all donor systems, no native Windows execution, and no remote GitHub Actions result were obtained. This is a local, noncommercial research artifact, not a safety certification or deployment recommendation.
