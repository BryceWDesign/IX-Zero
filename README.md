# IX-Zero

Experimental cognitive architecture for evidence-gated skill acquisition, compositional reasoning, and adaptive epistemic inquiry.

**Version 0.1.0** | Python 3.11+ | Source-available, noncommercial evaluation | No third-party runtime dependencies

IX-Zero investigates how a symbolic learner can acquire executable knowledge from observations, reuse that knowledge across tasks, recognize the limits of its current evidence, and request more informative experiments. The implementation operates over bounded, evaluator-defined domains. It is not demonstrated artificial general intelligence (AGI), a general-purpose language model, or a production decision system.

## Research contributions

- **Program induction:** discover finite executable operations from examples rather than receiving target expressions.
- **Active inquiry:** select observations that discriminate among competing hypotheses; abstain when evidence is insufficient.
- **Adaptive Epistemic Inquiry (AEI):** preserve hypotheses that are indistinguishable in one environment and request another environment when it can resolve the ambiguity.
- **Cumulative skill acquisition:** reuse validated programs as synthesis macros under a bounded search budget, with fresh evaluation of each acquired capability.
- **Grounded composition:** connect learned language, finite relational representations, numeric operations, and simulated actions through audited bindings.
- **Evidence lifecycle:** validate within declared scope, quarantine contradicted knowledge, invalidate dependent capabilities, and replay learning journals.

The current implementation integrates these mechanisms into a developmental coordinator, with IX policies directing bounded learning decisions. See [research design](docs/RESEARCH.md), [developmental architecture](docs/DEVELOPMENTAL.md), and [AEI](docs/EPISTEMIC.md) for the algorithms and their limits.

## Selected experimental results

The AEI evaluation comprises two disjoint six-seed campaigns using the same developer-defined synthetic task families. The engine receives observations and unlabeled input pools, not the hidden target expressions.

| Challenge | Observed outcome across both campaigns |
|---|---:|
| Double-shift acquisition | 12,264/12,264 correct |
| Four-shift acquisition | 12,264/12,264 correct |
| Constant-zero discrimination | 12,264/12,264 correct |
| Learned macro reuse | 12,264/12,264 correct |
| Matched reuse-disabled control | 12,264 abstentions |
| Contradictory audit control | 12,264 abstentions |
| Later contradiction invalidation | 12/12 laws quarantined |
| Denied environment access | 12/12 withheld promotion |
| Environment trial journal replay | 72/72 |

In the double-shift trials, the learner requested a four-bit environment after determining that its two-bit environment could not distinguish the surviving hypotheses. In the four-shift trials it instead requested a five-bit environment. Environment selection is derived from hypothesis disagreement; the evaluator controls access and supplies measured outcomes. The paired reuse comparison uses a common synthesis budget and matched evidence partitions, with the reuse-enabled arm receiving an acquired macro.

These results measure performance on finite mathematical fixtures, not broad generalization or human-level intelligence. A hidden exception outside the observation pools remains undetectable from the supplied evidence. Earlier experiments include wrong predictions, failed gates, and abstentions; the complete reported research scope is described in the [evidence index](results/README.md).

## Quick start

From the repository root, with Python 3.11 or later:

```sh
python -m ix_zero
```

On Windows, `py -3 -m ix_zero` or `Start-IX-Zero.cmd` can be used. Type `help` in the console. No API key, network connection, GPU, or installed package is required for the source-tree runtime.

To inspect the integrated developmental curriculum and AEI evaluation:

```sh
python -m ix_zero.developmental_cli --demo
python -m ix_zero.epistemic_experiment --trials 2 --check
python -m unittest discover -s tests -v
```

### Teach a finite rule

In the console, enter the following observations of a hidden transformation, without giving the engine its expression:

```text
new puzzle 2 5
teach 0 0 -> 1 0
teach 1 2 -> 3 1
teach 3 4 -> 0 3
ask
predict 2 1
audit 0 1 -> 2 0
audit 1 0 -> 1 1
audit 4 2 -> 3 4
validated 2 1
promote
```

The console exposes bounded learning commands, not unrestricted natural-language conversation. This example uses the original finite-state learner; see the [AEI API](docs/EPISTEMIC.md) for cross-environment inquiry.

## Reproduce the AEI experiments

Use a separate working copy for experiments that write into `results/`, so that reference evidence is not overwritten.

```sh
python -m ix_zero.epistemic_experiment --trials 6 --seed 310000 --check --output aei-primary.json
python -m ix_zero.epistemic_experiment --trials 6 --seed 320000 --check --output aei-confirmation.json
```

Reference records: [primary campaign](results/aei_epistemic_campaign.json) and [confirmation campaign](results/aei_epistemic_confirmation.json). Each contains input partitions, predictions, decisions, and replay outcomes. [Release verification](docs/RELEASE_VALIDATION.md) describes the local engineering checks and reproducibility status. The [GitHub Actions workflow](.github/workflows/ci.yml) defines Linux and Windows checks; hosted execution is not claimed until the workflow actually runs.

## Repository structure

| Path | Contents |
|---|---|
| `ix_zero/` | Learners, developmental coordinator, console, evaluators, IX policies, and incorporated components |
| `tests/` | Engineering regression and failure-mode tests |
| `tools/` | Historical evaluation and native comparison tools |
| `docs/` | Research methods, protocols, limitations, and provenance |
| `results/` | Raw and summarized scientific evidence, including historical negative results |

## Technical documentation

| Topic | Reference |
|---|---|
| Epistemic inquiry and cross-environment learning | [AEI](docs/EPISTEMIC.md) |
| Developmental scheduling and learned prerequisites | [Developmental curriculum](docs/DEVELOPMENTAL.md) |
| Context and contradiction recovery | [Contextual cognition](docs/CONTEXTUAL.md) |
| Representation synthesis and abstraction | [Representation](docs/REPRESENTATION.md) |
| Recursive grounded objectives | [Composition](docs/COMPOSITIONAL.md) |
| Capability composition and binding | [ACCE](docs/ACCE.md) |
| Grounded language and relational transfer | [Language](docs/LANGUAGE.md), [Analogy](docs/ANALOGY.md) |
| Strategy selection and operator discovery | [Strategy](docs/STRATEGY.md), [Operators](docs/OPERATORS.md) |
| Comparative and unified learning | [Comparative development](docs/SUPERIORITY.md), [Unified workspace](docs/UNIFIED.md) |
| Earlier discovery and IX policy integration | [Discovery](docs/DISCOVERY.md), [IX instructions](docs/IX_INSTRUCTIONS.md) |
| Historical evaluation and external task compatibility | [Historical evaluation](docs/FINAL_EVALUATION.md), [Validation](docs/VALIDATION.md) |
| Scientific context and donor provenance | [Research](docs/RESEARCH.md), [Donor review](docs/DONORS.md), [Notice](NOTICE.md) |

## Limitations

The hypothesis grammar, computational budgets, available representations, and environment interfaces are supplied by the implementation. AEI does not invent unrestricted environments, acquire general natural language, initiate arbitrary open-world objectives, or rewrite its architecture. Historical results may require the corresponding earlier engine for exact reproduction. Full independent benchmark evaluation, native donor comparisons, and general intelligence remain unestablished. No claim of universal correctness, research priority, consciousness, security certification, or production readiness is made.

## License

IX-Zero is source-available for noncommercial research and technical evaluation, not open source. See [LICENSE](LICENSE), [component licensing](LICENSE-COMPONENTS.md), and [NOTICE](NOTICE.md). Incorporated components retain their notices and applicable terms. Commercial, production, government, and other restricted uses require the applicable separate permissions.
