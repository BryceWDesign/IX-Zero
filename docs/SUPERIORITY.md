# Comparative development evaluation — v0.1.0

**Objective: measurable advantages over the original donor projects. Status: partially achieved; universal superiority is not established.** This is a development evaluation within v0.1.0, not an AGI claim.

## New mechanisms

- Signed two-channel integer prediction. The actual Sally expression generator provides its 779 original candidate entries; the new scalar learner extends them with constant translations, scaling, floor division, remainder and literal constants (8,580 total syntactic entries, not necessarily unique semantic functions). It maintains consistent hypotheses, uses minimum-complexity agreement, freezes training at audit, and requires three fresh passing audit observations. Inputs are bounded to signed integers in [-32,32]; training targets are bounded. The original Sally learner is not replaced by a mock. Original donor source hashes and adapted-file hashes are in `scalar_donor_provenance.json`. The three additional adapted Sally files retain their original license.
- Coupled induction. A bounded depth-two decision tree jointly conditions role-lifted fact changes and numeric updates on facts, coordinate equality, order, constants and coordinate comparisons. Leaves use either modular deltas or absolute values. All retained minimum-cost tree predictions must agree across viable families, and truncated search abstains. This is a specified finite grammar, not arbitrary ontology invention. Coupled models have separate evidence epochs; contradictory feedback quarantines the model. A fresh coupled epoch can supersede a failed separable model, but cannot inherit its validation. This addresses a previously observed failure for the tested families. Coupled predictions participate in the existing planner, goal machinery and IX policy; scalar tasks share the persistent workspace journal, but are not automatically used as joint planning primitives.
- Original IX policy, previous learners and all 28 earlier evidence files remain. A new check in CI runs the component regression campaign and fails on known-fixture regressions. Passing it is not proof of donor superiority.

## Direct Sally comparison

Run `python -m ix_zero.superiority_experiment --trials 12 --with-sally --check` with the original Sally checkout's `src` directory on PYTHONPATH. The untouched original Sally `InductiveLearner` is imported directly, not from the adapted vendor. The original Sally repository was extracted from the supplied archive; 182 Python source files were verified byte-identical to that archive. Its original pytest run exited 0, and its original three-seed research gate passed with 100% held-out accuracy in its own task suite. Those gate results are separate from the common benchmark.

Common test: 12 seeds (900–911), nine developer-selected scalar families, 32 held-out signed pairs per seed/family/arm. Both arms get the same candidate input pool, the same hidden function and maximum ten observed labels; Zero reserves three labels for frozen audits, while Sally may choose fewer probes and does not use Zero's additional audit gate. Queries and realized label counts differ. Final inputs lie outside the training numeric range. This is budget-capped, not compute-matched or exact-evidence-matched. Several extension families were selected to probe the added grammar, so the result is not independent blind validation. Neither learner sees held-out labels while fitting.

| Family | Zero correct / wrong / abstain | Sally correct / wrong / abstain |
|---|---:|---:|
| Product | 384 / 0 / 0 | 384 / 0 / 0 |
| Difference | 384 / 0 / 0 | 384 / 0 / 0 |
| Composition | 384 / 0 / 0 | 384 / 0 / 0 |
| Absolute difference | 384 / 0 / 0 | 384 / 0 / 0 |
| Offset | 384 / 0 / 0 | 0 / 192 / 192 |
| Scale | 384 / 0 / 0 | 4 / 28 / 352 |
| Remainder | 384 / 0 / 0 | 57 / 135 / 192 |
| Floor division | 384 / 0 / 0 | 0 / 128 / 256 |
| Random mapping (negative control) | 0 / 0 / 384 | 0 / 0 / 384 |
| **Total** | **3072 / 0 / 384** | **1597 / 483 / 1376** |

Zero used 986 observed labels (including audits) versus Sally's 291. Both achieved 1,536/1,536 in the four original-style arithmetic families, and Zero has not demonstrated better sample efficiency. Zero wins measured coverage and error rate on this particular expanded suite, not universal cognitive superiority over Sally. All 384 unstructured predictions were abstentions in each arm. No natural-language or open-world intelligence demonstrated.

## Coupled results (12 seeds)

| Experiment | Result | Important boundary |
|---|---|---|
| Held-out object identities | 1,224 correct / 0 wrong / 0 abstain | Full fit crossing of contexts and vector values; object identities withheld. |
| Sparse combinations | 432 correct / 0 wrong / 0 abstain | Per task 32 train combinations, six audit combinations, 12 disjoint final combinations; three families. |
| Planning | 612/612 replayed goals reached | Single-action outcome-defined goals in the bounded simulator, not open-world planning. |
| Previous separable model recovery | 12/12 incorrect before feedback; 12/12 blocked after feedback; 600/600 correct after fresh coupled fitting and auditing | 50 new training cases and six new audit cases per seed. Recovered the fixture, not all cross-modal failures. |

`results/superiority_campaign.json` contains row-level counts. `--check` checks these known synthetic fixtures, not AGI. Some negative tests deliberately falsify a model; failure is retained as evidence. A new valid epoch requires fresh validation and cannot recycle prior audit inputs. The earlier unified evaluation documents its historical limitations and results as they existed then.

## Native Julia comparison: still open

The supplied IX-BlackFox-AGI and IX-AGI donor implementations use Julia and their complete live interaction paths also require the native Ada guard. Neither Julia nor the required Ada toolchain was available in this build environment; a network attempt to acquire Julia failed. No substitute Python port is reported as a donor execution. `tools/export_native_comparison.py` exports the exact same symbolic fixtures and runs Zero against them. `tools/native_relational.jl` is an **unexecuted** proposed adapter for original donor predictive components (not their complete live runtimes), receiving the same symbolic evidence; donor components may train on the audit labels while Zero does not. It has no external actuator. The exported fixture contains 12 seeds and 576 final cases. Native donor result fields remain unavailable. The Julia source adapter needs native verification before its outputs can be used.

Example from repository root, with original donors extracted separately and Julia installed:

```powershell
$env:PYTHONPATH = "."
py -3 tools/export_native_comparison.py --out native_fixture.tsv
julia tools/native_relational.jl C:\path\IX-BlackFox-AGI-main C:\path\IX-AGI-main native_fixture.tsv
$env:PYTHONPATH = "C:\path\IX-Sally-main\src"
py -3 -m ix_zero.superiority_experiment --trials 12 --with-sally --check
```

The unexecuted adapter is not a substitute for the complete donor systems' original test campaigns. Full comparison requires native tests and common tasks that genuinely exercise all their capabilities, including their own strongest domains, matched actual evidence and compute, retention and continual learning, and independent evaluator-designed tasks. General superiority is not demonstrated by combining score totals across unrelated task suites.

## Local use

`usstart ["calc"]` creates a signed scalar task. `usobserve ["calc",[1,2],4]` supplies an observation; `usask ["calc",[[0,1],[2,3]]]` requests an informative input. `usaudit` supplies independent evidence, `uspredict` returns a prediction or null, and `usave` persists the shared workspace journal. For cross-modal induction use `ucstart [["prep","item"],2,5]`, `ucobserve [facts,vector,action,outcome_facts,outcome_vector]` and `ucaudit` with the same shape. See `help`, `tests/test_superiority.py` and `superiority_experiment.py` for fully specified executable examples. No API access, network actions, generated host code or automatic dispatch is introduced.

## Remaining limits

This comparative evaluation is a bounded research implementation. It does not implement every advanced donor mechanism, does not have unrestricted natural-language understanding, does not discover an unbounded computational grammar, and does not establish AGI. No full native Julia comparison or universal superiority claim. No remote CI, native Windows or formal proof claim. The new code remains v0.1.0, evaluation-only; see [component licensing](../LICENSE-COMPONENTS.md) for the distribution terms.

## Local build verification

94 unit tests passed on Linux/Python 3.13.5; all nine local workflow commands exited zero. The wheel built and installed without network or runtime dependencies and the installed one-seed comparative regression check passed. Historical comparative and installed-run data remain in `results/superiority_campaign.json` and `results/regression/comparative_installed.json`. These historical verification counts are not the current release test counts.
