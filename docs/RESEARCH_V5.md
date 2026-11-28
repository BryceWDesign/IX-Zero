> Historical initial 5.0.0 candidate record. See [expanded continuation](RESEARCH_V5_CONTINUATION.md) for later results and the current roadmap ledger.

# IX-Zero v5.0.0 research candidate

## Scope and architecture

This additive candidate preserves the v4.0.0 implementation, source-available research licensing and original interfaces. It does not certify AGI, consciousness, general software competence, or successful completion of all research objectives. It contains no external LLM calls. Use only in authorized environments. The new public HTTPS adapter reads public pages; it does not circumvent authentication, use accounts, send credentials, execute web content, or invoke an arbitrary shell. An optional documentation root is explicitly selected by the operator. No background autonomous process is started. Every session is user/host initiated; after an objective is supplied, the bounded runtime chooses its own candidate queries, candidate programs and available retrieval paths.

| Source module | Executed behavior and boundary |
|---|---|
| `exploration.py` | Finite program search, structural dispatch from integer/sequence/grid inputs, external named transition tables, disagreement sampling, disjoint audits, executable macro acquisition and contradiction-based dependency invalidation. No arbitrary code generation. |
| `pair_inquiry.py` | Runs the retained Sally-derived `scalar.vocabulary` and native `operators.synthesize`, alongside generated order predicates, on common lossless pair observations. Active query over surviving candidate predictions; no target expression supplied. Semantic duplicates on exposed inputs are collapsed to a representative; they may diverge outside that scope. |
| `audit.py` | Named observers, per-input votes, explicit no-majority failure, raw observation accounting. The host is responsible for actual independence. |
| `research_workspace.py` | Shared JSON-lines command protocol, typed dispatch, prior General workspace, native Sally `SelfModel` measures, candidate histories, explicit invalidation and replay. The self-model score is promotion fraction, not calibrated general intelligence. |
| `web_research.py` | Actual read-only public HTTPS retrieval using the NimbleInfo search concept; URL encoding, source following, timeouts, bounded pages, hashes, source excerpts, typed factual conflict detection and cached evidence. Local document search precedes internet retrieval if supplied. No general comprehension, source truth certification or authenticated browsing. |
| `arc_eval.py` | Bounded compositional grid transformation search on external ARC JSON files, test labels excluded from solver arguments, exact task/pair scorer and abstention tracking. No benchmark-wide competence demonstrated. |

### Research interface

```sh
python -m ix_zero.research_cli
python -m ix_zero.research_cli --workspace /path/to/authorized/docs
python -m ix_zero.research_cli --load /path/to/state.json
```

The typed `task` command routes controlled text, finite observations, graph objectives, relational questions and explicit research needs. Its dispatch rules are engineered, not learned general natural-language comprehension. `explore` accepts `train`, `pool`, `audit` as distinct input/output rows; `pool` and `audit` are exposed to the learner through measurement interfaces. `audit_sources` can replace `audit` with 3–7 named sources over identical audit inputs. `pair_inquire` accepts signed pairs, a unique `key`, optional `gate_budget` (0–3), and `query_budget`. `task` chooses pair inquiry automatically for compatible signed-pair/integer observations; otherwise it selects finite exploration. For example:

```json
{"command":"explore","train":[[0,0],[1,1]],"pool":[[-1,1],[2,2]],"audit":[[-4,4],[5,5]],"depth":1,"budget":8,"query_budget":1}
{"command":"route","start":"start","goal":"goal","edges":[["start","walk","side"],["side","walk","goal"]],"assumed":["walk"]}
{"command":"resolve","subject":"alpha","relation":"inside","target":"beta"}
{"command":"save","path":"ix-zero-research-state.json"}
```

`route` distinguishes optional assumptions from actual `forbidden` transitions. It does not override real constraints. The experimental graph example can find an alternate route and correctly declares the inaccessible control unreachable. It does not autonomously infer physical affordances from vision or natural language. `predict` and `pair_predict` identify extrapolations as experimental, and `challenge`/`pair_challenge` withdraw contradicted predictions. Invalidation in the macro library conservatively checks contiguous expanded program dependencies, not every semantic relationship. No host source code is rewritten.

The deterministic workspace checkpoint replays recorded native events; its digest checks accidental alteration, not adversarial authenticity. Web records are retained snapshots of retrieved excerpts and hashes, not replay-verified measurements or truth. A failed explicit refresh invalidates cache reuse. Text excerpts are lexical retrieval, not a general semantic reasoner. `resolve` first checks existing scoped local evidence; if unknown, it formulates a research query and attempts to parse exact supported source assertions. Disagreeing assertions are reported as conflict, not resolved by popularity. No arbitrary source text becomes executable instructions. Only the goal terms are sent to the search provider; local document contents are not uploaded by this implementation. Do not use confidential goals in public search.

Network access depends on host DNS, internet access, remote service availability and applicable permissions. HTML search can change or be blocked. The actual live attempt in this build returned `unavailable` due to DNS; deterministic transport-fixture tests are explicitly not live Internet. The optional local-document probe actually searched the recovered IX-AGI `docs` corpus and returned an excerpt from `NEUROSCIENCE.md`. The source claims were not independently verified. Local indexing is bounded to 128 eligible documents and 1 MB per file. Public retrieval is HTTPS-only, bounded, credential-free and read-only; globally routable resolved addresses are pinned and checked for redirects. This is not a production security certification.

### Reproduction

```sh
python -m unittest discover -s tests -v
python -m ix_zero.research_experiment --trials 12 --seed 950000 --check --output primary-rerun.json
python -m ix_zero.research_experiment --trials 12 --seed 960000 --check --output confirmation-rerun.json
python -m ix_zero.arc_eval /path/to/ARC-AGI/data/evaluation --budget 1000 --depth 2 --output arc-rerun.json
```

Official ARC task format and public data: https://github.com/fchollet/ARC-AGI and https://arcprize.org/guide/1. Full ARC data download failed in this environment. One public task, `00576224`, was inspected from its raw source and numeric grids transcribed for the spot check after the grid solver had been written. The solver was not tuned to the task after exposure. It abstained. Its local transcription hash is not an upstream file hash. The result is one diagnostic, not an independent blinded benchmark score or statistically valid sample. For a stronger evaluation, freeze candidate source before retrieving the independently specified tasks and evaluate them in a separate process, with an evaluator that alone holds test outputs. The built-in host protocol separates inputs and outputs logically, not by a hardened process boundary.

### Evidence and interpretation

The primary and confirmation campaigns use authored finite task families. Partitions are disjoint within each episode; seeds can reuse finite inputs across episodes. The candidate never receives the final labels through its search API. In the active query experiment the arms each attempted eight unary primitives and used one extra label and three audit measurements per episode. Across both campaigns the active arm achieved 432/432, random 222/432, fixed-order enumerative 216/432; other predictions were abstentions. These are simple correlated task families, not independent general tasks.

The macro control uses the same 72 attempted program limit and two-call depth. It learns `2x+1`, then searches for a new program from observations whose target is `(2x+1)^2`. Reuse makes the earlier program available as an executable macro; the disabled control cannot express the target at the tested depth. The reuse arm scored 816/816 final predictions, the disabled arm abstained on all 816. This is a causal intervention on the search library and a form of bounded developmental learning, not a claim of arbitrary self-rewriting or advantage over optimized external synthesis software. It is matched in candidate budget, not in elapsed time or audit count.

The cross-mechanism experiment automatically searches multiple retained mechanisms without a supplied family name. Across 24 seeds: product-plus-two, NAND and nonstrict inequality each score 672/672; GCD scores 0/672 with 672 abstentions. Scalar-only scores 672/672 on product-plus-two and abstains on the other 2,016 predictions. The control terminates earlier on unsolved tasks; candidate and realized label costs differ. The prior multi-architecture research package identified these types of failures. Therefore this is a targeted diagnostic, not a blind novelty test or compute-matched superiority claim. The finite table grammar experiment uses randomized opaque symbolic alphabets and supplied primitives; it shows ability to use an external declarative interface, not discovery of an independently authored language or representation.

### Reliability: retained negative results

At each 0%, 1%, 5%, 10%, 20% audit corruption rate, the evaluation runs clean and hidden-regime targets with random and targeted perturbation. For the targeted 1% case, the only audit point exposing the hidden regime is falsified. Single-observer promotion then passes incorrectly in 24/24 hidden-regime trials across the two campaigns (24 false promotions out of 48 total clean/hidden trials, 2,400 wrong final predictions); clean targets with corrupted labels are also withheld. The 3-observer experiment repeats the scenario with 1, 2 or 3 corrupted named simulated sources. With 1 corrupted observer the hidden-regime promotion is blocked in all 24 trials and the clean target remains correct. With 2 or 3 correlated corrupted observers false promotions recur in all 24 hidden-regime trials. Each 3-observer audit costs 300 raw labels rather than 100. This is an explicit majority assumption and a limited recovery mechanism, not a proof of calibrated confidence or a guarantee against deception. All votes, audit failures, abstentions and wrong held-out predictions are in the raw JSON. Wilson lower bounds describe audit outcomes, not certainty under distribution shift.

### Roadmap evidence ledger

| Objective | Status / remaining prerequisite |
|---|---|
| Preserve v4 regression suite and historic outputs | Executed; local verification in release record. |
| Donor integration | Selected retained Sally Python mechanisms executed; BlackFox-AGI diagnostic ideas and IX-AGI role/lossless representation concerns informed implementation. Exact donor sources recovered and inspected. Full native Julia/Ada interoperability and comprehensive capability parity not established. |
| Generalized active inquiry and unified representation | Executed finite unary and cross-mechanism pair inquiry; GCD fails; arbitrary objects, language and coupled-tree inquiry remain open. |
| Developmental learning, mechanism construction and improvement | Executable macro creation/reuse changes later search and is ablated. General new mechanisms, autonomous source-level architecture changes, and independent improvement evaluation remain open. |
| Creative problem reframing | Optional constraint comparison in a finite graph executed. Independently discovering real-world assumptions, representations and goals remains open. |
| Autonomous research and tool selection | Typed task can initiate retrieval and choose local documents before web, with source provenance and cached evidence. Real local retrieval executed; live internet unavailable here. Broad comprehension, autonomous open-ended research and externally verified learning from web sources remain open. |
| Open Windows/desktop and tool construction | Not implemented. Requires perception/action representation, legitimate host interfaces, operating environment and independent testing. No placeholder tool or claim of capability. |
| Independent grammars and representations | Finite declarative interface and authored random tables tested; independently supplied language task evaluation and autonomous representational invention not established. |
| Independent external benchmark | One public ARC task attempted, not solved; full independently frozen ARC run missing. |
| Conventional, DreamCoder and PushGP comparisons | Fixed-order enumerative and random-query controls executed; no DreamCoder/PushGP native run or matched external comparison. |
| Evidence noise, abstention, false promotion and containment | Actual corruption/consensus campaign and dependency invalidation tests executed; correlated corruption remains a failure, uncertainty calibration and broader distribution shift remain open. |
| Independent scientific evaluation | Reproducible scorer, disjoint data partitions and raw outputs provided. No third-party verification or independent evaluator certification. |

The highest-priority missing prerequisite is a more expressive, learnable representation and task/action model: adding internet or desktop tools will not by itself give the current finite learner the semantics needed to use them. These are research gaps, not accomplishments implied by a module name or by a passing software test suite.
