# Representation discovery development milestone (v0.1.0)

## Mechanism

The first learner composes ten predefined whole-state transformations. The new learner is separate and preserves that implementation and its experiments. It instead enumerates scalar expression trees over coordinates, constants, modular addition/subtraction/multiplication and comparisons. Independent output heads can use different expressions. Programs are pure and finite, never generated Python, shell commands or external effects. Full finite-domain truth tables eliminate equivalent expressions, without collapsing merely training-equivalent expressions. Minimum-cost fitting (adaptive mode) or all compatible enumerated expressions (strict mode) produces competing hypotheses. Maximum expanded expression size is 31, library size 8, width 2–3, modulus 2–7, search 6,000 unique tables or 120,000 binary attempts, default cost 5 in strict mode. Search truncation is reported. This is an expanded hypothesis class, not unrestricted discovery of new operators or new modalities.

A structurally different projection learner searches coordinate subsets for consistent observed output mappings. It requires repeated observations per key, selects smallest supported subsets, and does not extrapolate to unseen keys. Adaptive mode uses either family and abstains on explicit disagreement. Strict mode additionally abstains if any compatible enumerated expression disagrees at the requested input, even when a projection is available. If there is no compatible symbolic expression, projection predictions may still be made. Neither family constitutes independent external evidence; the evaluator supplies that. A three-passing-audit threshold with no wrong audit enables predictions; abstentions do not count as passes. A wrong non-abstaining audit blocks the episode. Training and search are frozen at the first audit, and audit inputs cannot be recycled. No statistical guarantee follows from this gate.

An output expression can be promoted only when unique under its selected symbolic search mode, of nontrivial bounded size, and consistent with every recorded audit on that output, after at least three passing whole-vector audits. An acquired expression acts as one search token in a later episode. Transfer to a new modulus requires new target examples and audits; semantic equivalence in one modulus does not imply equivalence in another. Dependencies are tracked conservatively: quarantine of a source disables all descendant episodes that received its library, even if they did not use it. A local replay journal reconstructs learning, audits and invalidation; its digest detects corruption, not deliberate forgery. The direct Python API and IX configuration are trusted local interfaces, not hostile-code sandboxes.

`discovery.ix` runs in the original IX interpreter and selects `discover`, `inquire`, `audit`, `abstract`, `complete`, `revise`, or `exhausted`. Python independently checks the request and does not fabricate outcomes. `discover` expands cost 3 to 5 in adaptive mode; strict mode starts at cost 5. `abstract` retains an eligible expression. `dplan` proposes shortest paths over audit-gated same-space action models; a proposal is not permission to execute and may fail when models are wrong. The new episode store is independent from the legacy store. No natural-language cognition, sensors, unrestricted self-modification or external execution is implemented.

## Research basis

- Ellis et al., DreamCoder (2021), https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf : executable library learning and reusable program components. IX-Zero does not implement DreamCoder's neural recognition or wake-sleep algorithm.
- Solar-Lezama, Program Synthesis course, Lecture 3, https://people.csail.mit.edu/asolar/SynthesisCourse2020/Lecture3.htm : bottom-up synthesis and observational equivalence. Here deduplication uses all bounded inputs to preserve rival generalizations.
- Chollet, On the Measure of Intelligence (2019), https://arxiv.org/abs/1911.01547 : skill acquisition, priors and generalization. The synthetic campaigns below do not meet an AGI criterion.

## Executed measurements

Seed ranges 100–111 (development comparison) and 200–211 (follow-up), same task families, not independently authored or blinded. Development probes used seeds 0–2. The target families were chosen to challenge and illustrate the new representational biases; the legacy baseline intentionally has a smaller, different grammar and may be unable to express them. Arms have equal target training and allocated audit inputs within a trial, but different compute and sometimes different numbers of actually consumed audits because failures stop acquisition. Test labels never enter model fitting or audit selection. Full per-case predictions, truths, training and consumed audits are in the JSON reports.

| Development arm | Correct | Incorrect | Abstained |
|---|---:|---:|---:|
| Original legacy cold | 0 | 0 | 2196 |
| Symbolic only | 1502 | 24 | 670 |
| Projection only | 132 | 0 | 2064 |
| Adaptive combined | 1634 | 24 | 538 |

24 wrong predictions remained after the audit gate, all conditional cases. Arbitrary full random mappings remain unsolved (264 abstentions in the adaptive arm). Balanced coordinate coverage allowed 132 correct projected permutation predictions, but random training coverage yielded 132 abstentions. No novel key values were inferred by projection. The three-coordinate family contributes 1140 of 2196 cases, and repeats one structural rule across seeds; the 2196 cases are not 2196 independent tasks.

| Follow-up, both cost 5 | Correct | Incorrect | Abstained |
|---|---:|---:|---:|
| Adaptive | 1732 | 8 | 456 |
| Strict | 1736 | 0 | 460 |

The strict mode is the default for new interactive discovery episodes because it reduced observed errors in this follow-up. Zero observed errors is not a future error guarantee. No external benchmark was executed. The earlier adaptive failures remain in the package.

Separate source-transfer test: 12 seeds, 372 held-out target examples per arm. A source expression learned with ten source examples and four allocated audits at modulus 5 is reused to fit a more complex target at modulus 7 (14 target examples, four allocated audits). Without source reuse, 0 correct/0 wrong/372 abstained; with reuse, 372 correct/0 wrong/0 abstained. This is near transfer within a single expression family, not independent domains, with extra source experience and effective hypothesis capacity in the reuse arm. Active query comparison (same pool, 10 training observations each): passive 24/0/48, active 48/0/24, conditional family only. Separate planning simulation (seeds 300–311): 144/144 goal cases correct, zero wrong/abstained, after learning two fixed simulated actions. Start=goal cases are included; case-level paths are retained. These tests do not establish human-level reasoning, open-world exploration, or AGI.

## Reproduce

```
python -m unittest discover -s tests -v
python -m ix_zero.discovery_experiment --out results/discovery_campaign.json
python -m ix_zero.discovery_experiment --strict --seed 200 --out results/discovery_strict.json
python -m ix_zero.discovery_planning --out results/discovery_planning.json
```

Run locally without third-party runtime dependencies. The package and runtime version is v0.1.0.
