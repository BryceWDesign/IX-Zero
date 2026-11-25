# Final development evaluation

Version: 0.1.0. Evaluation subject: unmodified milestone 4 cognitive engine and IX policies. This is a frozen-build engineering and capability evaluation, not an independent certification of AGI. The evaluation harness was authored after the build was frozen; its synthetic families are not externally blinded.

## Historical evaluation protocol

The following commands describe the earlier frozen milestone. Running them against the current AEI implementation is a new evaluation, not a replay of the earlier executable. Use a disposable checkout to avoid overwriting archived results. The exact earlier engine is not bundled in this release.

```text
python -m unittest discover -s tests -v
python -m tools.final_evaluation
```

The evaluator writes `results/milestone5/frozen_evaluation.json` and prints the family counts; it can overwrite the historical record. A second local execution produced a byte-identical JSON result. The 14 workflow commands for that historical build were run locally without modification. The original experimental output is in `results/milestone5/frozen_evaluation.json`. All 152 tests passed. An offline wheel build and installation into a clean virtual environment succeeded; the installed language demonstration and two-seed language experiment passed. These results apply to the local Linux/Python environment, not the remote GitHub matrix or native Windows.

## External task compatibility

External benchmark descriptions and public examples were obtained from public sources; the complete benchmark datasets were not obtained or evaluated. Numeric examples and selected task information were transcribed into the evaluator, not the learner. No official benchmark score is reported.

| Source | Executed diagnostic | Outcome |
|---|---|---|
| BIG-bench `simple_arithmetic_json` | All 30 numeric example pairs, using an adapter that supplies two integers rather than text; six separate training labels and three separate audits | 10 correct, 20 rejected as outside the signed input range |
| ARC-AGI-1 evaluation task `00576224` | First actual grid demonstration submitted to the native finite-state API | Rejected: native vector representation cannot accept the grid or variable output shape. No prediction or official score |
| SCAN | Four public README command examples submitted to the trained grounded-language demo | Two unsupported constructions, two abstentions; no action-sequence prediction. No SCAN training or official split |

The BIG-bench diagnostic is supervised, unlike the source's zero-shot text protocol. The engine learns the addition relation from separate examples, and does not receive the target answers. The 10/30 figure measures these adapted examples, not an official BIG-bench result. All 20 larger pairs exceed the existing input range of [-32,32].

Sources:

- BIG-bench, public example task: https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/simple_arithmetic_json/task.json
- ARC-AGI-1 dataset and task protocol: https://github.com/fchollet/ARC-AGI ; exact task: https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation/00576224.json
- Lake & Baroni (2018), SCAN: https://proceedings.mlr.press/v80/lake18a.html ; public examples: https://github.com/brendenlake/SCAN

## Fresh synthetic tests

The evaluation defines 12 seeds (202611240–202611251), 12 signed-scalar families and 8 finite-word operator families. All partitions are fixed by the harness. No cognitive source, policy, hypothesis vocabulary or adoption threshold was changed to fit the results.

Scalar episodes receive up to 12 training pairs and three distinct audits sampled from [-4,4]^2. Each episode then predicts 24 unused in-range pairs and 24 out-of-range pairs from the signed ranges [-20,-12] and [12,20]. In the random mapping control labels are independently generated for each input. The hidden-threshold control adds an offset only when the first input is at least 10, deliberately outside the fitting range. Operator episodes use three-bit words, search budget five, 12 training pairs, three audit pairs and all 49 remaining pairs; the single hidden exception is deliberately excluded from training and audit. Counts include all episodes, including blocked or unresolved learners.

| Suite | Correct | Incorrect | Abstained | Total |
|---|---:|---:|---:|---:|
| Scalar, in-range | 1770 | 6 | 1680 | 3456 |
| Scalar, shifted range | 1628 | 148 | 1680 | 3456 |
| Bit-vector operator | 2879 | 12 | 1813 | 4704 |

Scalar addition, product, sum of squares, cube and remainder were correct on all 576 cases per family. Minimum, GCD, XOR, offset seven and unstructured mapping were entirely abstained on by the scalar learner. The hidden threshold was correct on 288/288 in-range but wrong on 146/288 shifted inputs. Equality was only partially learned (88 correct, 8 incorrect, 480 abstained). The separate bit-vector engine did learn XOR, which highlights that task routing between representations is not unrestricted. AND yielded 539 correct and 49 abstentions; OR, XOR and NOR yielded 588 correct each. Three-bit modular addition, minimum and rotate-left were entirely abstained on. The hidden exception produced 576 correct and 12 incorrect predictions.

Corrective feedback on the first wrong held-out input was supplied only after all initial predictions for that episode were recorded. All 14 scalar and 12 operator episodes with incorrect predictions blocked subsequent prediction on the contradicted input. This is quarantine, not automatic successful relearning. All raw inputs, expected outcomes, predictions and feedback records are preserved in the JSON file.

Fourteen language probes against the existing demo gave six correct supported interpretations, two appropriate abstentions for unknown/ambiguous references, three abstentions on unsupported meanings or word orders, and three input-format rejections. Articles, politeness, negation, reordered language and sequential clauses were not generally understood. This is not open-world language acquisition.

## Milestone disposition

1. Operator discovery: implemented in a bounded finite-word grammar; no unrestricted discovery of arbitrary operators.
2. Strategy improvement: bounded strategy selection and verified exact scoring optimization; no autonomous arbitrary source rewriting.
3. Cross-domain representation: finite structural transfer; no unrestricted learning across unrelated raw domains.
4. Grounded language: supervised finite lexicon/role induction; no general conversation or open-world semantics.
5. Evaluation: frozen-build local testing and external example compatibility diagnostics completed. Full external benchmark execution and independent blinded evaluation remain outstanding.

The results do not establish human-level general intelligence, a standardized AGI score, or superiority over the complete donor systems. Native Julia and Ada/SPARK donor execution remains unavailable in this environment. The previous benchmark comparisons, licenses and evidence are retained. An independent reviewer would need to run a broader pre-registered protocol against frozen artifacts and native donor systems, using equal resource budgets and external held-out tasks.
