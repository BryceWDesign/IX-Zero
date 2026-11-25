# Relational representation transfer

## Scope

The learner transfers binary operations across finite relational representations. A representation contains three to seven opaque state labels and one to four observed total unary relations. The source contains either a fully observed binary operation table, an earlier uniquely resolved and audited transferred table, or a replay-validated promoted two-bit operator. The target receives independently supplied graph observations and a small number of operation labels. The learner is not given the correspondence between source and target states or relation names.

This is finite structural analogy, not unrestricted cross-domain learning. The graphs must have the same cardinality and the same multiset of unary relations up to a bijection. Graph extraction from raw text, imagery, physics or arbitrary observations is not implemented. The domains in the experiments are synthetic finite representations, and their shared algebraic structure is chosen by the evaluator.

## Algorithm

For each available source with compatible cardinality, enumerate every bijection between its state indices and the target state indices. Conjugate each source relation by that bijection and compare the multiset of resulting relations with the observed target relations. Relation labels and state ordering are not used as semantic matching signals. Retain only structurally compatible bijections, and transport the source binary operation under each retained bijection. Deduplicate identical operation tables while retaining their source provenance. The search is exhaustive within the bounded state space; it does not assume that structural similarity guarantees the same binary operation.

Target training labels eliminate incompatible tables. A prediction requires unanimous agreement among remaining tables. Audits use fresh target inputs, do not filter hypotheses, freeze new training, and quarantine an episode on contradiction. At least three passing audits and one training observation are required for prediction. Promotion into the reusable source library additionally requires a unique surviving transferred table. Retraction propagates conservatively to all episodes and promoted tables whose candidate generation used the retracted source. A new epoch is required after quarantine. Audits are not proof of general correctness.

The IX policy in `ix_zero/policies/analogy.ix` chooses inquiry, audit, promotion, completion or abstention. The Python host enforces the evidence gates. The optional source list on `anew` restricts the source library without supplying a state mapping. This is useful for isolating transfer through previously acquired representations.

## Operator bridge

`aimport` replays the original operator acquisition journal, requires a live promoted two-bit operator, and obtains its complete predicted table. Four source labels are supplied in the order of word values 0, 1, 2, 3. The operator table is inferred from the original validated model, not claimed to have been exhaustively measured. Target relation observations are separate and every target still requires its own validation. The source snapshot and provenance are recorded in the relational journal. The CLI checks the originating operator journal before each analogy operation and at startup, quarantining imported sources and descendants if their origin is missing, retracted or replaced. When using the Python API directly, call `reconcile_operators` with the current originating `OperatorMind` before using an imported source. Save both journals when persisting imported knowledge (`osave`, `asave`). Journal digests detect corruption, not malicious replacement or forged observations.

## Console example

Start the existing console and enter:

```text
asource ["cycle",["a","b","c"],{"step":["b","c","a"]},[[["a","a"],"a"],[["a","b"],"b"],[["a","c"],"c"],[["b","a"],"b"],[["b","b"],"c"],[["b","c"],"a"],[["c","a"],"c"],[["c","b"],"a"],[["c","c"],"b"]]]
anew ["target",["blue","red","green"],{"advance":["green","blue","red"]},"structural"]
aadvance ["target"]
aobserve ["target",["red","red"],"red"]
aaudit ["target",["blue","blue"],"green"]
aaudit ["target",["green","blue"],"red"]
aaudit ["target",["green","green"],"blue"]
apredict ["target",["red","green"]]
aadvance ["target"]
asave
```

The expected held-out prediction is `green`. The final `aadvance` promotes the transferred table. The labels and source graph are intentionally different; the target relation is encoded in the target list order. The example is illustrative synthetic evidence. Use `astatus` to inspect mapping counts, training and audit counts, and source candidates. `aretract ["cycle"]` quarantines the target and downstream dependents. `aimport [name,states,relations,operator_episode]` imports a previously acquired operator. `anew` accepts modes `structural`, `unstructured` (all bijections, ignoring relation structure), and `local` (projections, constants, and observed unary relations applied to either input). The latter two are comparative controls, not general-purpose learners.

## Reproduction

```text
python -m unittest discover -s tests -v
python -m ix_zero.analogy_experiment --trials 24 --seed 88000 --output results/analogy_campaign.json --check
```

The campaign consists of 24 seeds for each of eight task families and three experimental arms, plus 96 policy acquisition episodes and 24 operator-chain episodes. The fixed-partition comparison supplies identical two target training labels, three allocated audit labels, and all remaining held-out target labels per arm. Source observations and complete target unary relation observations are additional prior information. Training, allocated audit, and test input sets are disjoint. An allocated audit may not be attempted if the learner has no agreed prediction or is quarantined. The local arm does not receive an inferred source table. The unstructured arm has the same source operation and enumerates all state bijections, but does not use structural evidence. The local arm is a deliberately limited ablation, not a comparison against a strong general-purpose AI. Full graphs encode substantial information, and all representations are developer-designed.

| Matched task family | Structural correct | Structural incorrect | Structural abstained |
|---|---:|---:|---:|
| Rotation / phase | 1056 | 0 | 0 |
| Resource order / signal intensity | 713 | 0 | 31 |
| Switches / set membership | 264 | 0 | 0 |
| Permutation / tile layout | 744 | 0 | 0 |
| Total | 2777 | 0 | 31 |

The unstructured arm scored 264 correct, zero incorrect and 2544 abstentions; the local arm scored 32 correct, 30 incorrect and 2746 abstentions. The structural arm retained 456 distinct initial table candidates across the matched episodes versus 40416 without the structural filter. It still enumerates permutations, so this is not a measured runtime speedup.

Negative controls: wrong operation, nonisomorphic graph, and random target mappings each resulted in 1056 abstentions, zero correct and zero incorrect predictions in the structural arm. A near-match with one hidden exception produced 817 correct, 19 incorrect and 220 abstentions; the three-audit gate did not detect those 19 errors beforehand. A subsequent contradictory audit quarantined all 19 affected episodes and prevented further predictions. This is retrospective correction, not anticipation or independent discovery of the hidden exceptions.

The separate IX-driven policy experiment promoted 96 of 96 target models and predicted all 2904 remaining inputs correctly, after 96 selected training labels and 288 passing audits. These remaining inputs were not preselected before inquiry, so the result is not directly comparable with the fixed-partition campaign. In the operator bridge experiment, 24 independently synthesized XOR operators transferred to a set-membership representation and then into another relabelled representation using only the first transferred table. Across the two target stages, 528/528 held-out predictions were correct; all 24 chains were quarantined when their original operator was retracted. These are related finite algebras, not evidence of transfer between arbitrary real-world disciplines.

`results/analogy_campaign.json` contains individual source observations, graph structures, target labels, predictions, attempted audits, feedback, traces and replayable chain events. The experiment intentionally preserves errors and abstentions. Earlier evidence files remain unchanged. No external model, network connection or paid dependency is required. No claim of AGI or independently blinded testing is made.
