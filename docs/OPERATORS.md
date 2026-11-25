# Operator synthesis and acquisition

IX-Zero v0.1.0, development milestone 1.

## Scope

The operator learner synthesizes finite-word functions of two inputs. Its base grammar contains input terminals, constants zero and one, bitwise NAND, logical left shift by one and logical right shift by one. No XOR, AND, OR, XNOR, NOR, implication, minimum, GCD or arithmetic addition target is installed as a named operation. Programs are immutable expression trees interpreted by a bounded evaluator, not generated Python source or arbitrary host code.

The learner enumerates increasing operator cost (0–6), using packed semantics over every input pair at the selected word width (2–5 bits) to prune equivalent expressions. It receives training outcomes only from the caller. It retains one representative per semantic equivalence class at a given width; this pruning does not establish equivalence at other widths. It retains an expanded expression only after at least six distinct training observations, a unique cheapest surviving semantic class, and at least three fresh passing audits (the developmental scheduler and strategy evaluator now require all their reserved audit inputs). This is a finite statistical/inductive evidence gate, not a formal proof of the hidden task. A later contradiction blocks the source and conservatively blocks every episode that used its library. Incomplete search layers are discarded; budgets are 250,000 candidate attempts, 35,000 semantic models, 127 expanded expression nodes, 32 episodes and six acquired operators per library. No external model or paid dependency is required.

Learned operators can be invoked on arbitrary subexpressions during subsequent synthesis. Their cost is one library call plus the costs of their arguments; fully expanded size remains bounded. Library reuse changes the hypothesis space and consumes previous source experience. A lower synthesis cost is not evidence of lower execution time or lower total compute. Fresh target training and audits remain mandatory, including across word widths. An operator is new to the search vocabulary, not a fundamentally new mathematical primitive or discovery unknown to science.

The original IX interpreter executes `policies/operators.ix` to select inquiries, audits, abstraction, completion, abstention or revision. The Python host enforces the evidence gate. The revised inquiry policy considers disagreement among all surviving hypotheses on available training inputs, rather than prematurely auditing a unique minimum-cost candidate when an available input could distinguish a competing explanation. No free-form language learning or automatic modification of the host implementation is introduced.

## Console

```text
onew ["logic",3,5,false]
oadvance ["logic"]
ostatus ["logic"]
```

`oadvance` requests an input pair until ready for audit; provide its measured outcome with `oobserve ["logic",[a,b],outcome]`, replacing the placeholders with integers. When the policy requests an audit, choose a fresh pair not supplied during training and use `oaudit ["logic",[a,b],outcome]`. Repeat until `oadvance` reports abstraction or completion. `opredict ["logic",[a,b]]` returns an audit-gated output or JSON null. `onew ["next",4,3,true]` opens a new episode with acquired operators. `osave` writes a separate journal, replayed and checksum-checked at startup. `oretract ["logic"]` invalidates the source and dependent episodes. The checkpoint checksum detects accidental damage, not malicious tampering. The console accepts an optional training pool via `oadvance ["logic",[[a,b],...]]` to support evaluator-controlled acquisition.

## Evaluation

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m ix_zero.operator_experiment --trials 12 --seed 9400 --check --output operator-results.json
```

The synthetic campaign partitions each width's input domain into 14 training, six audit and remaining held-out pairs. Each discovery task has 44 held-out pairs at width 3 and 236 at width 4. There are twelve seeds. Fixed-budget reuse and cold arms receive identical target training inputs, the same allocated audit set and maximum search cost of three; unresolved episodes cannot enter audit. Source acquisition uses a separate episode and cost budget five. The evaluation oracle is separate from the learner but was authored alongside the research implementation. These are not blinded external tasks.

Final campaign (seed 9400):

| Evaluation | Correct | Incorrect | Abstained |
|---|---:|---:|---:|
| Seven representable discovery families | 23,004 | 0 | 516 |
| Addition, minimum and random mapping controls | 0 | 0 | 10,080 |
| Target tasks with learned XOR | 6,720 | 0 | 0 |
| Same targets, cold learner | 0 | 0 | 6,720 |
| IX-controlled active XOR acquisition | 528 | 0 | 0 |

The seven families are AND, OR, XOR, XNOR, NOR, implication and shifted intersection. Three episodes in these families abstained in the final campaign. All twelve active episodes promoted an acquired operator with six training labels and three audit labels per episode. The initial campaign at seed 8400 had one failed active audit; its result and trace are preserved separately. The revision was evaluated on new seeds. The three-generation composition regression checks a source XOR, shifted XOR and a third function composed from both acquired skills. Transfer evidence is bounded to related bit-vector tasks. The expanded AST can be inspected and evaluated independently of the packed search table.

Raw records, training labels, audit outcomes, held-out predictions and policy traces are in `results/operator_campaign.json` and `results/operator_campaign_initial.json`. Historical campaigns are retained as evidence of the earlier policy; running the current code with seed 8400 is not a replay of that earlier inquiry policy. All numbers describe this finite benchmark only.

Related research: Ellis et al., *DreamCoder: growing generalizable, interpretable knowledge with wake–sleep Bayesian program learning*, 2023, https://doi.org/10.1098/rsta.2022.0050. This implementation uses symbolic enumeration and library retention, not DreamCoder's neural or Bayesian algorithm.
