# Initial donor review and design decisions

This document describes the initial design review; the current incorporated components are identified in [NOTICE.md](../NOTICE.md).

Seven supplied archives were opened, their README and license read, and selected implementation files inspected. `donor_inventory.json` records archive hashes, member counts, exact inspected paths and file hashes. This is a targeted source review, not a full independent code audit or execution of all donor suites. No donor runtime was linked or vendored.

| Donor | Source-level observation | IX-Zero decision |
|---|---|---|
| IX-Sally | `developmental_learning.py` describes bounded numeric program induction; `lifetime_learning.py` tracks episode and strategy history. | Retain example-driven hypothesis learning and memory as ideas; implement finite-state transformation grammar and composition of learned programs. The overlap is substantive, not an assertion of a wholly unprecedented approach. |
| IX-BlackFox-AGI | `programs.jl` defines typed expression nodes; `inquiry.jl` derives objectives from coverage and failure; `metacognition.jl` distinguishes empirical coverage and accuracy. | Use a finite instruction vocabulary and uncertainty-driven input selection; do not treat disagreement as a calibrated probability. |
| IX-AGI | `callosum.jl` defines delayed numerical packets and matrices; `hemisphere.jl` learns nonlocal relational effects; `organism.jl` binds decisions to the donor environment and guard. | Do not simulate hemispheres; maintain competing programs with explicit consensus instead. No biological claim. |
| IX-Function | `source_learning.py` tracks source-domain evidence; `falsification.py` implements a falsification ledger. | Separate source support from target calibration and retain failed predictions in raw results. |
| IX-IntentRealityLoop | `delta.py` tracks prediction/outcome discrepancy; `permission.py` separates intent from authority. | Separate predictions from supplied observations and block models on counterexamples; no physical or remote actions. |
| IX-BlackFox | Wave 19 documentation binds plans, approvals and external verification; agent authorization code represents scoped grants. | Keep plan generation purely simulated and require audited action models. Do not claim cryptographic authorization or reuse of the BlackFox gateway. |
| IX | `ix/cognition.py` supplies canonical cognition-contract taxonomy; `ix/runtime.py` describes deterministic traceable execution. | Keep the grammar small and reviewable; checkpoint stores examples, audit outputs, contradictions and promotions for replay. No IX parser integration. |

The donor documents explicitly constrain their AGI claims. IX-Zero respects those boundaries rather than relabeling prior prototypes as achieved general intelligence.
