# Recursive grounded language and objectives

Development stage 2, version 0.1.0. Scope: CF-01 (semantic generalization) and CF-07 (recursive compositionality).

## Mechanism

`ix_zero/compositional.py` induces surface constructions from full utterances, scenes and teacher-provided semantic trees. The reference vocabulary and object descriptor order come from an existing audited `LanguageEpisode`. A semantic tree contains an object identifier or an intent with one or two child trees. The learner searches for grounded child spans, substitutes child slots, and retains the unique remaining pattern of lexical anchors and ordered slots. A new construction must not introduce ambiguity into previously observed training examples. Two distinct grounded training meanings and three distinct passing audit meanings are required per rule; training freezes after the first audit. Scene identity renaming alone does not produce fresh evidence. This is tree-supervised induction, not learning arbitrary semantics from raw text or from scalar rewards alone.

A memoized span parser recursively applies the induced constructions to unseen expressions. The grammar may contain prefix, infix, postfix and multiword anchors and reversed argument order. Ambiguous interpretations abstain. Each rule consumes literal tokens, and the parser bounds utterances to 64 tokens, objective depth to 8, grammar size to 32 rules, training and audit records to 256 each, and parse work to 100,000 steps. Ambiguity is checked against the full grammar, including unvalidated rules. All rules implicated in a successful parse must satisfy the evidence gates. A contradictory fresh audit quarantines the grammar. Invalidation of the source reference learner blocks use of the derived grammar.

The evaluator recursively resolves referents and executes each objective through the existing live, audit-gated ACCE bindings. No arbitrary host code, external actions, or pretrained model are involved. The demonstration additionally learns dispatch bindings for integer/integer and mixed object/integer arguments, from observed outcomes. It can therefore feed intermediate outputs to other validated capabilities. Unavailable bindings or invalidated cognitive sources cause abstention. The existing IX interpreter continues to govern the prior dispatch acquisition policy; the new recursive parser is implemented in Python, not as a modification to the IX interpreter. Grammar and dispatch sessions are not persisted. The original three- and five-token language implementation remains available as a baseline.

## Interactive execution

```
python -m ix_zero.compositional_demo
```

The demonstration learns seven phrase constructions on a synthetic scene before accepting input. Try:

```
please compare the red sphere with the green cube
red sphere versus green cube
compare inspect red sphere rule green cube blue cone
combine inspect red sphere with rule green cube blue cone
```

The first two expressions return 1 (first value greater). The third returns 0 (equal), because the red sphere has value 2 and the learned word operator computes 1 XOR 3 = 2. The fourth returns 0 by applying that operator again to two equal results. Actual execution traces include the selected bindings and their evidence. `inspect orange sphere` and arbitrary open-ended questions abstain. This is a restricted learned language; a phrase must be supported by the induced grammar and audited capabilities.

The main console exposes `rnew`, `robserve`, `raudit`, `rinterpret`, `rexecute`, and `rstatus`. Initialize a grammar using a previously trained and audited reference episode; use the existing `cworkspace`, `cnew`, `cobserve`, `caudit` commands to teach the corresponding cognitive bindings. Use JSON arguments as shown by `help`.

A complete semantic label for a comparison is `["compare","C","B"]`. A nested label is `["compare",["inspect","C"],["rule","B","D"]]`. The teacher supplies these complete trees but does not identify lexical anchors, token spans or child ordering. The mechanism infers those from the aligned utterance and scene.

## Reproduction

```
python -m unittest discover -s tests -v
python -m ix_zero.compositional_experiment --trials 24 --seed 180000 --check --output results/compositional_campaign.json
python -m ix_zero.compositional_demo --check
```

The campaign uses 24 seeds, independently sampled construction aliases, and independently shuffled argument order. The audited object descriptor vocabulary and computational bindings remain fixed. Each grammar receives ten shallow teaching sentences and fifteen shallow audit sentences, across five constructions. Held-out shallow semantic pairs exclude training and audit pairs, including those with relabeled object identities. The nested expressions have semantic depths absent from grammar training and audit. Each seed receives 48 shallow and 96 nested execution queries. Results require both the correct semantic tree and correct evaluated value. The flat-grammar ablation is an explicit depth-one restriction on the same learned grammar's output; it is not a separately trained competing model. The original fixed-vocabulary parser is evaluated without retraining on the randomized aliases, so its zero coverage is a compatibility result, not a matched-training generalization comparison.

| Test | Correct | Incorrect | Abstained |
|---|---:|---:|---:|
| Shallow held-out objectives | 1152 | 0 | 0 |
| Nested depth 2 | 576 | 0 | 0 |
| Nested depth 3 | 576 | 0 | 0 |
| Nested depth 4 | 576 | 0 | 0 |
| Nested depth 5 | 576 | 0 | 0 |
| Flat-grammar ablation, same 3456 cases | 1152 | 0 | 2304 |
| Unknown word, ambiguous reference, unsupported typed objective | 0 | 0 | 72 |

All 24 deliberate semantic shifts produced an incorrect interpretation before feedback; each grammar then quarantined itself after a contradictory fresh audit. It did not automatically learn a replacement grammar. Raw held-out predictions are retained in `results/compositional_campaign.json`. The campaign was executed twice with byte-identical results. All earlier experimental result files are retained unchanged.

## Limitations

CF-01 and CF-07 have bounded implementations. The construction family, reference roles, typed capability signatures and recursion limits remain developer-defined. The learner requires structured semantic supervision, cannot infer arbitrary new semantic categories, cannot handle unseen idioms or arbitrary paraphrases, and does not autonomously invent new recursive operators from language. The evaluation uses synthetic teacher data, known reference vocabularies and fixed underlying computational models; it is not an official SCAN, COGS or AGI benchmark score. Session persistence, context-dependent meaning recovery, general-purpose conversational reasoning and open-ended learning remain unresolved. No human-level AGI claim follows from these results.
