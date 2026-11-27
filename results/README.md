# Experimental evidence

The files in this directory are retained research records, including unsuccessful trials, negative controls, abstentions and earlier developmental evaluations. They are not a collection of independent AGI benchmark scores. The protocol, scope, and relevant limitations are documented in the corresponding technical notes. Historical results describe their original engine and evaluator, not necessarily the current implementation.

| Research area | Records | Protocol |
|---|---|---|
| Integrated native reasoning, program induction and cross-engine composition | `general_primary.json`, `general_confirmation.json` | [Workspace](../docs/GENERAL.md) |
| Three-level composition and bounded goal search | `stack_campaign.json` | [Continual](../docs/CONTINUAL.md#three-level-skill-composition-and-goal-search) |
| Continual learning and cumulative skill acquisition | `continual_primary.json`, `continual_confirmation.json` | [Continual](../docs/CONTINUAL.md) |
| Historical hybrid stress evaluation | `historical/hybrid_stress.json` | [Evaluator](../tools/hybrid_stress.py), [Continual](../docs/CONTINUAL.md) |
| Integrated neural-symbolic inquiry | `hybrid_primary.json`, `hybrid_confirmation.json` | [Hybrid](../docs/HYBRID.md) |
| Adaptive Epistemic Inquiry | `aei_epistemic_campaign.json`, `aei_epistemic_confirmation.json`, `aei_stage5_historical.json` | [AEI](../docs/EPISTEMIC.md) |
| Developmental curriculum | `developmental_*.json` (including pilot, pre-broad and negative controls) | [Developmental](../docs/DEVELOPMENTAL.md) |
| Contextual recovery | `contextual_*.json` | [Contextual](../docs/CONTEXTUAL.md) |
| Representational synthesis | `representation_*.json` | [Representation](../docs/REPRESENTATION.md) |
| Recursive and grounded composition | `compositional_*.json`, `acce_campaign.json` | [Composition](../docs/COMPOSITIONAL.md), [ACCE](../docs/ACCE.md) |
| Language and relational transfer | `language_campaign.json`, `analogy_campaign.json` | [Language](../docs/LANGUAGE.md), [Analogy](../docs/ANALOGY.md) |
| Strategy and operator learning | `strategy_*.json`, `operator_*.json` | [Strategy](../docs/STRATEGY.md), [Operators](../docs/OPERATORS.md) |
| Discovery and comparative learning | `discovery_*.json`, `unified_campaign.json`, `superiority_campaign.json`, native comparison records | [Discovery](../docs/DISCOVERY.md), [Unified](../docs/UNIFIED.md), [Comparative](../docs/SUPERIORITY.md) |
| Earlier finite-state learning | `campaign.json`, `validation.json`, `challenges.json` | [Validation](../docs/VALIDATION.md) |
| Frozen evaluation | `historical/frozen_evaluation.json` | [Historical evaluation](../docs/FINAL_EVALUATION.md) |

Selected historical smoke-run and installed-package datasets are grouped in `regression/`. They should not be counted as independent confirmations. Summary text files accompany several datasets for readability. The release's current engineering checks are documented separately in [release verification](../docs/RELEASE_VALIDATION.md).

To preserve the distributed reference evidence, run experiments that generate output in a separate checkout or direct their output to new filenames. `MANIFEST.sha256` at the repository root records the distributed file hashes.
