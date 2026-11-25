# Integrated developmental curriculum (ACCE stage 5)

## Scope and execution model

This stage implements a bounded, event-driven developmental coordinator in `ix_zero/developmental.py`. It addresses CF-10 through successive acquisition and reuse of learned skills, prerequisite discovery for newly offered finite domains, contextual recovery, and replayable persistence. It is not an unrestricted autonomous agent, an architecture that rewrites its code, an open-world goal generator, or a general intelligence benchmark result.

`Development` maintains named learning objectives. A job records its kind, explicit or inferred dependency identities, disjoint unlabeled training/audit pools, state, evidence history and failure reason. A dependency must be live and validated before its dependent job starts; missing dependencies can remain unresolved. The original IX interpreter runs `policies/developmental.ix` to select learning, waiting or completion. The host chooses a ready objective deterministically, prioritizing an existing learning job, then jobs with more waiting dependents, then name. Each step issues at most one observation request. The evaluator supplies only the requested outcome through `answer`. There is no oracle or executable target code in the coordinator. Completion of scheduling does not imply that every objective succeeded; consult `status` and `live`.

The coordinator integrates these existing mechanisms:

- Word-operator synthesis and library reuse, using the exact packed inquiry kernel from the strategy milestone. It does not perform a new autonomous strategy optimization during this stage.
- Reversible representation search and parameterized abstraction induction over validated source episodes. A narrow template can guide training, but before any audit the coordinator expands to the full candidate family. This prevents the narrow template from being treated as sufficient evidence in isolation; it does not guarantee correctness outside the family.
- Structural graph transfer from a validated two-bit operator, with inferred correspondences and independent target evidence.
- Grounded lexical induction, recursive semantic grammar and learned typed dispatch.
- Contextual hypothesis generation and active recovery following an explicit contradictory observation. Replayed historical observations become training; the replacement epoch has its own reserved fresh audits.

`offer(name, labels, train, audit)` accepts a fresh finite representation problem without supplied dependency names. The coordinator selects live word operators of the compatible width and, if enabled, a compatible abstraction-parent pair from previous validated representation episodes. If no source is available, it reports a missing operator-width prerequisite. The host still needs to supply a bounded problem domain, candidate evidence inputs and measured labels; this is prerequisite inference and learning-goal selection within a known task schema, not arbitrary goal invention. Compatible parents are chosen without target labels. The coordinator conservatively invalidates dependent jobs when any declared prerequisite is no longer live. The legacy engines remain available separately. See the later [epistemic milestone](EPISTEMIC.md) for changes to reserved audit enforcement, operator width bounds and cross-environment learning.

## Evidence and persistence

Training and audit inputs must be distinct within each objective. Bindings deduplicate by the relevant object value/relation projection. The scheduler freezes training at the audit boundary and never uses audit outcomes to select or refit a representation. An audit failure leaves the affected objective unavailable. Context recovery requires a previously validated contradictory prediction and a new objective identity, fresh observation/audit pools, and live sources. A recovery task may not reuse the previous episode's observed inputs as new evidence. The optional context-drift flag is supplied by the evaluator; the coordinator does not infer the time of change. The `abandon` action terminates an unanswered request without fabricating a label.

The journal stores task registrations, inferred prerequisites, decisions, supplied observations, recovery requests, abandonment and explicit operator retractions. Checkpoint loading reconstructs the engines by replaying the journal and comparing each scheduling decision, then checks the replayed journal. Atomic replacement protects a local single-writer save; the SHA-256 digest is for corruption detection, not authentication of an untrusted author. The checkpoint is bounded to 4 MB and 4,096 events, and the coordinator admits at most 32 jobs. No networking, shell execution, external actions, or self-modification is exposed by the coordinator. The previously published component-specific checkpoint formats are separate from this journal.

## API and local console

```sh
python -m ix_zero.developmental_cli
python -m ix_zero.developmental_cli --demo
python -m ix_zero.developmental_cli --check
```

On Windows use `py -3 -m ix_zero.developmental_cli` instead of `python -m ...` if needed. The demo runs a synthetic teaching curriculum, then replays its checkpoint. It prints the supplied scene, values and relation states. Example sentences:

```
compare inspect red sphere combine red cube with green cube
join red cube with green cube
what would you ask me about the world
```

The last sentence remains unsupported. The demo is not a general chatbot. The default non-demo console starts with an empty curriculum. Use `help` to inspect the JSON commands, `next` for the next evidence request, `answer` for a measured outcome, `status`, `predict`, `execute`, `recover`, `retract`, `challenge`, `abandon`, `save` and `load`. A save path may be set with `--state`. The demo uses an external synthetic evaluator, not a built-in source of real-world facts.

Minimal Python example of an unlabeled word objective:

```python
from itertools import product
from ix_zero.developmental import Development

m = Development()
xs = list(product(range(8), repeat=2))
m.add('sample', {'kind': 'operator', 'deps': [], 'width': 3,
                 'budget': 5, 'train': xs[:24], 'audit': xs[24:27]})
r = m.next()
# r['input'] is a requested input; supply its actual observed outcome.
# m.answer(observed_outcome)
```

Task schemas consist of common keys `kind`, `deps`, `train`, `audit` and these specific keys:

| Kind | Additional keys | Evidence input |
|---|---|---|
| operator | width, budget | pair of bounded word integers |
| epistemic | width, widths, budget | initial pools use pairs; requests use [width,a,b] or a width for an environment grant |
| representation | labels | pair of integers in the finite domain |
| context | labels, contexts | context identifier and two labels; dependencies are base and alternative representation jobs |
| analogy | states, relations, source_states, source_relations | two target state labels; dependency is a promoted two-bit operator |
| lexicon | none | utterance and scene rows |
| grammar | none | utterance and scene rows; dependency is a validated lexicon |
| binding | intent, inputs, output, contexts | typed argument list; an object is a JSON object matching `acce.Object`; `contexts` maps contextual dependencies to explicit context identifiers |

The evaluator supplies a complete observed outcome or semantic tree for each requested input. A recovery job is created through `recover` rather than ordinary registration. `offer` currently supports finite numerical representation objectives only. See `developmental_experiment.py` for a complete multi-stage usage example, including nested language execution. That file is an evaluator, not part of the learner's hypothesis vocabulary.

## Controlled experiments

Two 12-seed final campaigns (211000–211011 and 212000–212011) compare reuse enabled against operator-macro and abstraction reuse disabled. Both arms retain ordinary source representation transfer, contextual recovery, and structural transfer. They receive identical target partitions, although actual numbers of requested labels and candidate libraries differ. The second-generation operator task uses a lower search-depth limit than its initial source task. Its dependent representation cannot start if that operator was not acquired; the resulting abstentions are a dependency test, not a comparison against an independently unconstrained numerical learner.

The evaluator registers the initial jobs in reverse dependency order, then offers a new domain without naming prerequisites. It supplies labels only on requested training/audit inputs. The external evaluator defines target transformations independently of the candidate encoding implementation. A changed-context counterexample is then observed; the recovery epoch receives new training and audit partitions. The final language tests compose separately learned lexical, dispatch, representation, contextual and graph capabilities. The nested language cases are withheld compositions, but constituent value pairs can overlap earlier evidence; they are not independent language-corpus samples. Some previously scored initial-model/graph held-out inputs become training examples of later objectives; held-out scores are per objective, not disjoint across the entire developmental history.

Across both campaigns the reuse arm validated every intended active objective in 24/24 curricula. Each curriculum has 25 jobs including the deliberately contradicted historical world model. Results (correct / total):

| Task | Reuse | Disabled macro/template reuse |
|---|---:|---:|
| Higher-generation word operator | 888/888 | 0/888 (abstained) |
| Dependent representation | 888/888 | 0/888 (unmet dependency) |
| Newly offered domain | 888/888 | 888/888 |
| Context recovery | 1,176/1,176 | 1,176/1,176 |
| Held-out lexical meaning | 4,968/4,968 | 4,968/4,968 |
| Nested language execution | 768/768 | 768/768 |
| Graph prediction | 192/192 | 192/192 |
| Language-driven graph execution | 72/72 | 72/72 |

There were zero incorrect held-out predictions in these listed final tasks. All 24 deliberately selected wrong incumbent predictions were quarantined in each arm, with corrected downstream execution. The reuse arm replayed its checkpoint successfully in 24/24 trials and blocked numeric execution after source retraction in 24/24 trials. A separate unit test verifies that an unrelated graph remains usable after numeric source retraction, including after checkpoint reload, and becomes unavailable after its own source is retracted. The first final campaign reproduced byte-for-byte.

The narrow-template pilot with the automatic offered-domain feature passed 12/12 initial curricula, but failed its strict gate on a fresh range (seed 209010): the inferred abstraction fit six training inputs, then failed its first audit. The capability was withheld, yielding 37 abstentions for that objective. Both pre-broad campaigns are retained. The final coordinator broadens the hypothesis space before the first audit; the regression fixture now passes, and the two disjoint final ranges above pass. The safety check costs more search: 6,528 representation candidates across the two scheduled target representation tasks in the reuse arm versus 6,144 in the disabled arm. The candidate libraries differ, and this stage does not demonstrate lower search cost or sample efficiency overall. Earlier pilot results with narrower template search are historical, not measurements of the final algorithm. Their exact former code is not included in this cumulative build.

Separate negative controls on seeds 213000–213011: 444/444 random-mapping inputs abstained; 1,368/1,368 unsupported simultaneous two-context changes abstained; all 12 initial contradicted models quarantined. Thirty-six unsupported language requests (three per trial, including the original open-ended questions) abstained. These are intentional synthetic controls, not claims that all novel inputs are correctly recognized as unsupported.

Reproduction:

```sh
python -m ix_zero.developmental_experiment --trials 12 --seed 211000 --check --output developmental.json
python -m ix_zero.developmental_experiment --trials 12 --seed 212000 --check --output developmental-fresh.json
python -m ix_zero.developmental_challenge --trials 12 --seed 213000 --check --output developmental-negative.json
python -m unittest discover -s tests -v
```

Raw predictions, job states, evaluator feedback, IX decisions and journal events appear in `results/developmental_campaign.json`, `results/developmental_fresh_campaign.json`, and `results/developmental_negative.json`. Pilot and pre-broad experimental datasets are retained alongside them. All campaigns are developer-designed; no valid independent ARC-AGI, SCAN, COGS or BabyAI benchmark score is established. Full comparison with donor runtimes remains outstanding. The finite task schemas, observation pools, predicate families, language constructions and computational budgets are still provided by a developer or evaluator. There is no unrestricted open-ended learning, arbitrary conversational understanding, autonomous architecture rewriting, or demonstrated human-level AGI. The scope of the ten deficiency categories is bounded and further failure modes are expected. Version is 0.1.0; native Windows and remote GitHub Actions execution remain unverified.
