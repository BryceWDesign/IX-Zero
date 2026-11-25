# Cross-environment epistemic inquiry (AEI)

Version: **0.1.0**. This is an executable extension of the ACCE developmental coordinator, not a demonstration of AGI. An adversarial evaluation of the prior Stage 5 implementation motivated falsification-oriented validation and experimental environment selection.

## Mechanism

`ix_zero.epistemic` enumerates NAND/shift programs and optionally previously audited program macros. It retains a representative for each joint semantic signature over the caller-declared widths (2 through 5 bits). Unlike single-width pruning, constant zero and a double left shift can remain distinct even when every two-bit input gives the same result. The candidate vocabulary, allowed widths and gate-cost limits are supplied by the design; this is not invention of unrestricted new representations.

The `Development` coordinator accepts an `epistemic` job. Training is selected by partitions of the surviving hypothesis set. If available training inputs cannot distinguish the survivors, the learner selects an ungranted width maximizing distinct surviving complete semantics, with a deterministic smaller-width tie-break. It requests that environment from the evaluator; it cannot grant itself access or invent evidence. After one joint semantic class survives, it requests every remaining declared width for independent scope verification, widest first. This distinction is recorded in the request's `inquiry` field. The evaluator supplies disjoint **unlabeled** training and audit pools for each grant, then answers individual queries with observed outcomes.

Promotion requires a nontruncated joint search, one surviving joint semantic class, six or more training observations, and every reserved audit in **every declared width** passing. There is no training after audits begin. A denied grant, unavailable informative evidence, exhausted hypothesis set, or inconsistent audit leaves the job unavailable. Predictions outside declared and audited scope are refused. This gate deliberately pays extra observation cost rather than extrapolating without validation. One semantic class within a finite grammar is not proof of the true law. Finite audits can miss unseen exceptions.

A live epistemic law can be reused as a bounded synthesis macro in a new epistemic job, or a live legacy operator can seed its search. All target evidence is acquired afresh. No direct bridge into legacy representation/grammar dispatch is implemented. A fresh post-promotion `challenge` can invalidate a law; declared dependent jobs become unavailable automatically. Existing contextual recovery remains unchanged. No automatically repaired model is promoted after a contradiction; a fresh learning objective is required. The scheduler's IX policy still controls which job may advance. The epistemic query-ranking and scope-selection algorithms are deterministic Python, not new IX language policies.

### Reserved audit correction

The legacy developmental operator scheduler now passes its full audit count to the IX operator policy. Representation, analogy, lexical, binding, grammar and contextual jobs also cannot become live with unused reserved audits. The standalone strategy evaluator passes the full count as well. Interactive low-level operator calls without a declared audit pool retain the old minimum of three; they cannot know which additional evidence exists. The old single-width learner continues to use local semantic pruning. Use the new epistemic job for cross-environment discrimination. The old domain bound is extended to 5 bits. This changes the former explicit rejection of `Domain(5)`; width 6 is still rejected.

Checkpoints remain replay-validated; old journals whose decisions differ under the strengthened gate are rejected, not silently migrated. Exact replay of an older journal may require the corresponding historical engine, which is not bundled in this release. The SHA-256 digest is corruption detection only, not authentication. Existing bounds: 32 jobs, 4096 journal events, 4 MB checkpoint; joint search 250,000 attempted candidates, 35,000 models, 127 expanded nodes, 8 cached joint searches, budget 1–5 and at most 6 macros. Widths 2–5 are finite, not open-world environments. No external action, arbitrary source execution, network access or automatic code modification is added.

## API

```python
from ix_zero.developmental import Development
from ix_zero.operators import Domain

m = Development()
xs = Domain(2).inputs
m.add('law', dict(kind='epistemic', deps=[], width=2, widths=[2,3,4,5],
                  budget=2, train=xs[:9], audit=xs[9:13]))
r = m.next()
# r['action'] == 'request'; r['phase'] == 'train', 'audit', or 'environment'.
# For an environment request, r['input'] is a width; answer with a dictionary
# containing disjoint unlabeled pairs: {'train': [...], 'audit': [...]}.
# For train/audit, r['input'] is [width, a, b]; answer with a measured word.
# m.abandon() declines a pending request; m.status() displays the actual state.
# m.predict('law', [width,a,b]) returns a word only for a live, audited law.
# m.challenge('law', [width,a,b], observed_word) uses a fresh outside-pool input.
```

Console: `python -m ix_zero.developmental_cli`; use the `add`, `next`, `answer`, `challenge`, `status`, `predict`, `save`, `load` commands. The evaluator's truth functions are in `ix_zero.epistemic_experiment`, not the learner.

## Reproduction and results

```sh
python -m unittest discover -s tests -v
python -m ix_zero.epistemic_experiment --trials 6 --seed 310000 --check --output aei-results.json
```

The six-seed campaign contains 18 original audit-trap trials, 36 environment trials and 12 paired transfer curricula (each with a source and a target). `results/aei_epistemic_campaign.json` contains request traces, partitions represented by supplied inputs and held-out predictions, status and replay verdicts. These are synthetic, author-designed experiments; they are not external independent AGI benchmarks. Held-out totals are per task; input pairs are not globally independent across seeds or tasks.

| Trial | Live / trials | Correct held-out | Wrong | Abstained |
|---|---:|---:|---:|---:|
| Original clean operator | 6/6 | 918 | 0 | 0 |
| Original late exception | 0/6 | 0 | 0 | 912 |
| Original early exception | 0/6 | 0 | 0 | 912 |
| Double shift | 6/6 | 6,132 | 0 | 0 |
| Four shifts | 6/6 | 6,132 | 0 | 0 |
| Constant zero | 6/6 | 6,132 | 0 | 0 |
| Environment denied | 0/6 | 0 | 0 | 18 |
| Contradictory audit | 0/6 | 0 | 0 | 6,132 |
| Later contradiction (before challenge) | 6/6 | 6,132 | 0 | 0 |
| Transfer with learned macro | 6/6 | 6,132 | 0 | 0 |
| Transfer without macro | 0/6 | 0 | 0 | 6,132 |

In the old late-exception probe the exception itself is an audit, not held-out. All six were requested and all affected laws were rejected; the 912 held-out ordinary inputs abstained. In the prior Stage 5 evaluation all six were incorrectly retained and the exception prediction was wrong. The original evaluator's comparison is retained in `results/aei_stage5_historical.json`. All 36 new environment trials replayed their journals. In the six later-contradiction trials, the six live laws became unavailable after a fresh inconsistent label. A unit test verifies dependent macro invalidation and replay. In the paired transfer, a two-gate budget sufficed to acquire four shifts using the learned double-shift macro; with macro reuse disabled, the same budget and evidence partitions failed. This is a bounded search-vocabulary improvement, not a claim about global efficiency.

The counterfactual negative control changes one untouched five-bit held-out truth in each double-shift trial. Because the changed input never entered a supplied pool, all six existing predictions would be wrong under that alternative truth. This is a counterfactual comparison against recorded predictions, not six additional learner executions. It explicitly demonstrates the residual indistinguishability problem: more scopes and full audit consumption do not guarantee universal truth.

The environment selection trace for seed 311000 double shift begins with width 2 where all seven survivors have identical local semantics, then requests width 4 where seven different semantics remain. It subsequently requests widths 5 and 3 for verification, not for discrimination. Four-shift acquisition instead first requests width 5. Candidate generation is not told the target expressions.

## Unresolved research

The next expansion would need an extensible representation grammar over sequences, graphs and structured state; genuinely new operation families; measured active selection between model families; recovery that proposes a fresh learning objective; larger independent challenge corpora; and evidence beyond synthetic deterministic environments. These are not implemented or claimed by this milestone. Empty curricula still return scheduler completion rather than spontaneous open-world objectives. No consciousness, autonomous scientific discovery, human-level generality or AGI is demonstrated.

### Fresh-seed confirmation

`results/aei_epistemic_confirmation.json` repeats the protocol at seeds 320000–320005 (and corresponding environment/transfer offsets). All first-campaign aggregate counts above were reproduced, including the reuse-disabled abstentions; all 36 additional environment journals replayed. This is new randomized partition evidence within the same authored task families, not a blinded independent benchmark. Across both runs, first requested widths were 4 in all 12 double-shift cases and 5 in all 12 four-shift cases. The historical Stage 5 evaluator (`tools/aei_stage5_historical.py`) is retained as protocol documentation; exact reproduction of its historical output requires the earlier Stage 5 engine, which is not bundled.
