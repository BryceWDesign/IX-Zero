# IX-Zero 5.0.0 — continuation research

This research build extends the initial 5.0.0 candidate without changing the release version or discarding any of its historical evidence. It does not certify AGI, consciousness, unrestricted computer competence or completion of the full research roadmap. All core mechanisms operate without an external AI model. External software-assisted results are identified separately.

## New executable mechanisms

| Mechanism | Implementation | Boundary |
|---|---|---|
| Iterative program induction | `program_induction.py` | Fixed-arity 1–4 integer expressions, two-register loops, finite guards, generated state updates, 65-step execution limit, 30,000 unique-candidate universe cap. Learner receives examples and measurement access, not a task name. |
| Acquired callable programs | `ProgramMemory` | Audited code becomes a callable search leaf; a contradicted source withdraws syntactically dependent programs. At most eight available prior macros per search and bounded nesting. |
| Fresh-evidence program revision | `revise`, `propose_revisions` | Generate 300 wrapper alternatives over a previous executable body: input transformations, output transformations and guards. Select with training/inquiry, audit separately. Revised bodies inline prior code and are independently revalidated rather than silently trusting revoked library keys. No Python source rewriting. |
| Object-centric grid induction | `object_induction.py` | Choose among pixel, connected-component, projection, scale and object-operation hypotheses; 4/8-connectivity and single/multicolor segmentation. Search through a finite authored grammar; not general object or representation invention. |
| Independent software tool discovery | `tool_induction.py` | Inspect actual runtime function signatures and evaluate numerical functions in explicitly authorized `math` and `operator` modules, retaining a verified tool identity. This is tool-assisted behavior, not native synthesis. |
| Evidence-driven escalation | `research_workspace.py` | When the retained pair mechanism has no supported candidate, try expression/register induction; if that fails without using audit evidence and authorized tools are available, try tool inquiry. If an audit was consumed, request fresh evidence rather than reuse it as independent validation. This is typed, engineered dispatch; fallback uses extra compute. |
| Conservative auditing | `audit.py` | Optional unanimity instead of majority over three to seven named observers; host establishes source independence. Disagreement blocks promotion and does not prove truth. |
| Browser control learning | `browser_learning.py` | Observe visible text and enabled button controls in an actual Chromium browser, discover transition graph by clicks, store/reload it, revalidate path on new objective. Trusted operator-supplied local HTML only, offline browser context, no arbitrary remote page or Windows automation claim. |
| Separate-process ARC evaluator | `external_eval.py` | Candidate subprocess receives training pairs and unlabeled test inputs only, not IDs or test outputs. Logical separation, not a hostile-code sandbox. Reports exact task/pair scores, abstentions and execution errors. |

The research console's checkpoint format is now v3. Loader accepts historical v1 and v2 formats and verifies replay fingerprints; tool-discovery replay depends on the installed Python implementation/signatures. `program_revise` can use previous program code but requires fresh examples and audit; the host is responsible for disjointness from prior episodes. Revisions do not guarantee universal correctness. Browser graph snapshots have integrity checks and visible-state replay, not authenticated provenance or general state equivalence.

## Research results

Four six-seed campaigns (271100–271111 and 281100–281111) are recorded as `results/v5_continuation/campaign-{a,b,c,d}.json`. The compact aggregate is `summary.json`. Python's standard-library GCD and LCM are independently implemented reference functions, but the sampling, other authored families and evaluator are not externally designed benchmarks. The original v5 data remains in `results/v5/`.

### Native algorithm induction and composition

| Arm / task | Correct / held-out | Shifted positive inputs | Promotions |
|---|---:|---:|---:|
| Native iterative GCD | 2,784/2,784 | 960/960 | 24/24 |
| Iteration disabled GCD | 0/2,784 (abstained) | 0/960 | 0/24 |
| Acquired GCD reused to synthesize LCM | 2,784/2,784 | 960/960 | 24/24 |
| Reuse disabled LCM | 0/2,784 (abstained) | 0/960 | 0/24 |

Each arm attempts 12,000 unique candidate programs per task, not identical execution time or audit measurements. Training samples use positive integer pairs in 1–48; shifted pairs use 100–256. No target expression is delivered to the solver. The learned Euclidean-type update is executable in its own DSL. The LCM program calls the acquired operation inside a generated arithmetic expression. The positive-domain programs are NOT universally valid: on the six signed/zero boundary probes per seed, GCD produced 48 wrong predictions out of 144; LCM produced 48 wrong and 24 abstentions out of 144. A deliberate contradictory label control withdraws both the source and dependent program; it is not a true mathematical observation.

### Fresh-evidence revision

Two 12-seed experiments (291100 and 291200) compare unchanged bodies with revisions on disjoint signed/zero and positive regression observations. The experimenter provides boundary strata (zero/equal/sign cases) for the augmented pool, after observing the earlier failure. The ordinary random pool is retained as a control. This is targeted bounded repair, not an independently invented diagnostic input generator. `(0,0)` is training evidence and not included again in the final holdout.

| Target | Unchanged | Revision, random pool | Revision, stratified pool |
|---|---:|---:|---:|
| GCD | 881/2,520 correct, 1,639 wrong | 1,680 correct, 840 abstentions | 2,520/2,520 correct |
| LCM | 994/2,520 correct, 1,526 wrong | 1,890 correct, 630 abstentions | 2,520/2,520 correct |

The stratified pool uses eight extra candidate measurement inputs, so it is not an equal-resource comparison with random pool. All 48 stratified revisions were promoted; no wrong final prediction occurred in these specific partitions. A revision constructs an alternative program, not a new host-level algorithm for arbitrary architectures. The fresh evidence is evaluator-provided, and independent third-party evaluation remains necessary.

### Search controls

The enumerative, random and single-field tournament evolutionary controls search the same generated finite grammar with 6,000 distinct candidate attempts per task. Evolutionary scoring performs additional fitness evaluations and is not runtime-matched. On positive-domain GCD: enumerative 2,784/2,784; random 2,204/2,784; evolutionary 1,392/2,784; remaining cases abstained. These are internal comparison controls, not actual DreamCoder or PushGP implementations, and do not establish general superiority over program synthesis systems.

### Tool-assisted discovery

With explicit `math` and `operator` scope, Zero selected the actual standard-library functions `math.comb`, `math.isqrt`, `operator.xor` and `math.gcd` from numerical observations, using introspected signatures. Each produced 960/960 held-out correct results across 24 seeds. These are external computations performed by the selected function; they must not be attributed to native intelligence. The approved numerical function catalog is engineered, not autonomous access to arbitrary software.

### Grids and external ARC tasks

The authored four-family diagnostic scored 244/384 correct and 140 abstentions using the richer grammar vs 96/384 correct and 288 abstentions using the pixel-only grammar. Largest crop: 76/96, smallest mask: 72/96, scale: 96/96, wrapped translation: 0/96. No incorrect final grid predictions in either arm. Both arms have the same maximum candidate cap but the larger grammar tries 4,128 candidates per task vs 192 for the pixel-only arm, so this is not compute-matched.

The external diagnostic uses three official ARC-format tasks, one previously exposed evaluation task `00576224` and two public training tasks `007bbfb7` and `08ed6ac7` inspected after freezing the object solver source. All three modes (original, objects, pixels) abstained on all three. The numeric input files were manually transcribed from raw public data after network failure, and local hashes are not upstream byte hashes. These are a small convenience sample, not an independent blind test suite or official ARC score. No solver was tuned to the new public tasks after seeing them. The evaluator, not Zero, retrieved the two task sources through a separate research interface. URLs and source hash are in `external-arc-diagnostic.json`. Full dataset evaluation remains unperformed.

### Audit reliability

With 24 seeds, 100 audit inputs, three observers and one targeted corrupt label per compromised observer, majority falsely promotes the hidden-regime rule in 24/24 trials when two or three observers are corrupted. Unanimity blocks those cases with one or two compromised observers, but falsely promotes in 24/24 trials when all three are corrupted. It also withholds all 24 otherwise correct clean-target trials when even one observer disagrees. Both consume 300 raw audit labels per episode. Common-mode corruption remains unresolved. Earlier random/targeted 0–20% corruption campaigns remain preserved.

### Actual browser learning

Four local authored dynamic applications (two two-seed campaigns) were executed in Chromium with randomized opaque panel/control labels. Zero received only the visible observation and control interface, not the transition table. Training consumed 76 actual clicks. Restored graph memory reached 8/8 requested goals in 16 clicks total; no-memory route lookup achieved 0/8 and blind random clicking achieved 1/8 under the same two-click execution limit. Training costs are additional and not matched to the controls; this is a memory-reuse diagnostic, not a claim of superiority at equal overall cost. Unavailable-goal controls returned unknown route. The browser environment is a trusted local HTML experiment, not a hardened sandbox, not live web, not Windows desktop operation. Do not execute untrusted HTML with this adapter. Local visible-state observations may alias hidden state.

## Reproduce

Run from a clean checkout with Python 3.11+; the actual local validation used Python 3.13.5 on Linux. All standard numerical and grid mechanisms use only the standard library.

```sh
python -m unittest discover -s tests -v
python -m ix_zero.completion_experiment --trials 6 --seed 271100 --check --output fresh-a.json
python -m ix_zero.completion_experiment --trials 6 --seed 271106 --check --output fresh-b.json
python -m ix_zero.completion_experiment --trials 6 --seed 281100 --check --output fresh-c.json
python -m ix_zero.completion_experiment --trials 6 --seed 281106 --check --output fresh-d.json
python -m ix_zero.revision_experiment --trials 12 --seed 291100 --check --output fresh-repair-a.json
python -m ix_zero.revision_experiment --trials 12 --seed 291200 --check --output fresh-repair-b.json
python -m ix_zero.external_eval /path/to/ARC-AGI/data/evaluation --limit 20 --seed 310001 --budget 5000 --output fresh-arc.json
```

ARC official data repository: https://github.com/fchollet/ARC-AGI . Select evaluation tasks without inspecting their labels before a future frozen evaluation. Full corpus is not distributed here. Optional browser extra (network/install permissions required for installation):

```sh
python -m pip install ".[browser]"
python -m playwright install chromium
python -m ix_zero.browser_experiment --trials 2 --seed 301100 --output fresh-browser.json
python -m ix_zero.browser_learning /path/to/trusted-app.html --goal "desired visible line" --save browser-memory.json
python -m ix_zero.browser_learning /path/to/trusted-app.html --goal "another visible line" --load browser-memory.json
```

The actual local run used a system Chromium executable supplied with `--browser-executable /usr/bin/chromium`. Browser checks are optional, not part of the dependency-free CI matrix. The standard-library tool and native learning commands are exposed through `python -m ix_zero.research_cli` with JSON-lines requests; `tools` lists supported input representations. For example `program_inquire` needs `key`, `train`, `pool`, `audit`, optional `budget`, `reuse`, `allow_loops`, `search`; `program_predict` needs `key`, `input`; `program_revise` additionally needs `base_key` and new `key`; `tool_inquire` needs explicit `modules`. `audit_sources` and `audit_policy` may replace `audit`. `grid_induce` requires `train` pairs and unlabeled `inputs`. No user-provided code strings are executed by these commands.

## Roadmap completion ledger

| Requirement | Actual current status |
|---|---|
| v4 preservation and regression | Executed; retained tests and results. |
| Sally integration | Retained selected actual mechanisms; not complete capability parity. |
| BlackFox-AGI and IX-AGI native integration | Unresolved: Julia and Ada build/runtime unavailable; no full native interoperability claim. |
| Unified cognitive representation and mechanism selection | Expanded typed evidence dispatch and executable representation alternatives; arbitrary semantic representation and general natural language remain unresolved. |
| Autonomous inquiry, creative reframing | Active disagreement selection, finite graph route reframing and browser control discovery demonstrated. Arbitrary assumption discovery remains open. |
| Autonomous cognitive improvement | Bounded learned-call composition and fresh-evidence code-data revision demonstrated; no open-ended source/architecture self-improvement. |
| Developmental memory | Audited program calls, dependent withdrawal, checkpoint replay, actual browser graph reuse demonstrated. |
| Independently authored unfamiliar grammar and representation invention | External table interface retained; runtime software signature discovery and object alternatives added. General independent grammar induction and representational invention unproven. |
| Internet and external software | Existing real HTTPS client still fails DNS locally; actual authorized numerical tools and local Chromium executed. Broad semantic web research, unfamiliar application comprehension and Windows desktop operation remain open. |
| External benchmarks | Three public ARC diagnostic tasks attempted, all abstentions; no independent broad benchmark passed. |
| Conventional baseline / DreamCoder / PushGP | Equal attempted-candidate internal enumeration/random/evolution controls executed; native established systems remain unrun. |
| Noise and calibration | New unanimity trade-off measured; correlated all-observer corruption and general calibration remain unresolved. |
| Independent scientific verification | Reproducible code, scores, subprocess interface, controls and retained failures; third-party evaluation not performed. |

The leading unresolved prerequisite is a general learnable semantic representation that can construct new task/action grammars and independently gather meaningful evidence in unfamiliar domains. More external access alone is insufficient. The complete roadmap is not declared solved. Local CI passing does not establish remote GitHub CI or AGI.
