# Bounded learning strategy improvement

IX-Zero can generate inquiry configurations, evaluate them on a supplied labeled task corpus, and retain a configuration only if it satisfies selection, confirmation and challenge gates. This is configuration-level optimization of the finite-word operator learner. It does not generate arbitrary Python, edit its implementation, change its evaluator, or relax the operator learner's minimum training and audit requirements.

## Mechanism

The configurable inquiry dimensions are output diversity, minimax or collision scoring; equal, cost-weighted or inexpensive-frontier hypothesis scoring; high or low coordinate tie breaking; and checked or packed table extraction. The heuristic mode enumerates 18 packed configurations. The exact mode compares the existing checked configuration with a packed equivalent. In both modes the original IX `Operators` policy authorizes inquiries and the host rejects out-of-pool or already observed inputs. Only the inquiry ordering or its computation changes. Candidate models, evidence gates, training budgets and promotion rules are unchanged.

The packed kernel validates each candidate input once when constructing the eligible set, then extracts its value using the same shift and mask as `Domain.value`. This removes repeated validation of an already validated pair. It does not remove any evidence validation. The two kernels select identical questions for identical candidate populations and eligible input sets.

The evaluator expects disjoint training, audit and held-out inputs within each task, and disjoint named families across three partitions. Duplicate task identities and identical complete evidence across partitions are refused. These checks do not establish semantic independence of arbitrary user-supplied task labels. Selection ranks eligible proposals; only the selected challenger proceeds to confirmation and, if confirmed, challenge. The host and `policies/strategy.ix` must both approve adoption. A failed gate retains the incumbent; no second challenger is tried on those outcomes.

The gate requires no increase in incorrect predictions and no reduction in correct predictions on any task. In addition, either all aggregate work and label measures are nonincreasing with at least one strict improvement, or candidate/input scoring work falls by at least 50% with at most 5% additional aggregate labels and at most two additional labels per task. `query_work` counts scored candidate/input pairs, including fallback scoring; `redundant_validation_work` counts repeated `Domain.value` input validations within scoring. These counters exclude synthesis, host policy execution, observation filtering and audits. They are not total runtime or compute complexity estimates. The tradeoff is an explicit experimental policy choice.

## Experiments

Initial 12-configuration inquiry experiment (seed 17000): no candidate qualified. A later strict comparison (seed 37000) selected a challenger but rejected it on confirmation because its scoring work increased. An exploratory tradeoff campaign (seed 47000) provisionally accepted a frontier strategy after two partitions. A separate 56-task probe (seed 57000) invalidated that decision: the frontier strategy produced 3,133 correct, 57 incorrect and 4,930 abstentions; the incumbent produced 3,001 correct, 42 incorrect and 5,077 abstentions. The frontier configuration is not installed as an approved default. The final three-partition gate incorporates a challenge requirement. Historical outputs reflect their respective earlier evaluator revisions, not the final three-partition evaluator.

The subsequent exact-kernel experiment (seed 67000) used 90 task episodes across 15 function families and three seeds, at widths 3 and 4. The final packed configuration passed all three partitions. Every requested training row, audit row, held-out prediction and promotion decision matched the checked configuration exactly.

| Partition | Correct (each) | Incorrect (each) | Abstained (each) | Labels (each) | Scored pairs (each) |
|---|---:|---:|---:|---:|---:|
| Selection | 4,109 | 0 | 241 | 277 | 1,702,005 |
| Confirmation | 3,480 | 0 | 870 | 242 | 1,764,424 |
| Challenge | 956 | 12 | 3,382 | 192 | 1,734,815 |
| Total | 8,545 | 12 | 4,493 | 711 | 5,201,244 |

The checked kernel performed 5,201,244 redundant validations; the packed kernel performed none inside its scoring loop. Both still validate input eligibility and all observations and audits. The 12 hidden-condition errors remain unresolved. This is an efficiency result, not higher intelligence, accuracy, coverage or label efficiency. A separate paired process-CPU timing experiment over 24 inquiry fixtures, seven alternating-order repetitions, measured the median scoring-only CPU times in `results/strategy_timing.json`. Timing is host-specific and is not an adoption criterion or an end-to-end speedup claim.

## Reproduce and use

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m ix_zero.strategy_experiment --trials 3 --seed 67000 --mode exact --check --output strategy.json
py -3 -m ix_zero.strategy_timing --output timing.json
py -3 -m ix_zero
```

Console:

```text
mfit "results/strategy_corpus.json"
mstatus
onew ["logic",3,5,false]
madvance ["logic",[[0,0],[1,2],[3,4],[2,5],[6,1],[7,7]]]
msave
```

Supply the requested observed output using `oobserve`, and repeat `madvance` with the eligible training pool. Supply independent audits using `oaudit` when requested. The existing `oadvance` command always uses its original inquiry strategy. The selected configuration is explicitly applied through `madvance`. `mrollback` restores the checked strategy without deleting experimental records; `msave` persists the rollback. A single strategy epoch may be evaluated per state file. A new independent corpus requires a new state file. Checkpoints reconstruct the experiment, compare the report and verify a digest; they are not authenticated against malicious local modification. Corpus and checkpoint byte limits apply. No network access or external model is required.

Raw final evidence: `results/strategy_final.json`, `results/strategy_corpus.json`, `results/strategy_final_summary.txt`, `results/strategy_timing.json`. Exploratory and failed experiments are retained in `strategy_initial*`, `strategy_strict*`, `strategy_heuristic_provisional*` and `strategy_probe*`. The synthetic evaluator and strategy configurations were designed within this project; this is not an independent blinded benchmark. The approach selects among programmer-defined strategies, not arbitrary learned algorithms or autonomous recursive architectural modification.
