# Grounded language acquisition

## Model

`ix_zero/language.py` implements a finite lexicon and grammar induction engine. It begins with an empty lexicon and 720 possible role-order pairs: 3! orders for one-referent sentences and 5! for two-referent sentences. A scene contains up to 16 objects, represented as `[id, kind, attribute]`. The measured interpretation is `[action, object_id]` or `[action, first_id, second_id]`.

Each training example supplies a complete utterance, scene and semantic outcome. For each role-order hypothesis, the outcome and scene constrain word-to-typed-symbol associations. Repeated words must retain consistent associations. Incompatible hypotheses are eliminated. The learner does not receive the evaluator's word dictionary, role-order permutation or per-token alignment. The allowable roles, sentence lengths, tokenization and scene schema are supplied by the developer; it does not discover arbitrary syntax or semantic categories.

A prediction requires agreement across all surviving hypotheses, known words and unique referents in the current scene. Three passing fresh audits are required per sentence arity. The first audit freezes all training for that episode; audit labels do not fit the lexicon or select a grammar. A contradiction blocks the episode. A new episode is required for relearning after a shift. Audits are separate evaluator-supplied outcomes, not guarantees of semantic correctness. Reordered scene rows normalize to the same evidence input. Journal replay reconstructs hypotheses and validation state, with a SHA-256 corruption check (not authentication).

`express` enumerates bounded candidate realizations from learned words and tests whether they resolve to the supplied intent across all surviving models. This verifies internal round-trip consistency, not fluency or human preference. Synonyms can be learned from additional training demonstrations, but a word with multiple context-dependent meanings is outside the supported representation.

The IX policy `policies/language.ix` selects inquiry, audit, completion, revision or abstention using learner state. Inquiry ranks unlabeled candidate utterance/scene pairs by disagreement and lexical coverage. The host separately rejects unsupported actions. The external teacher supplies scenes, candidate sentences and outcomes; the engine does not observe the real world directly.

## Runtime integration

- `compute` interprets a binary sentence into an existing promoted, unblocked operator episode. Object IDs are mapped to evaluator-supplied word values. The operator's own audit gate remains mandatory.
- `transfer` interprets a binary sentence into an existing relational episode and queries its audited prediction. Original operator dependencies are reconciled first; retraction prevents downstream predictions.
- `simulate` interprets an utterance into an existing symbolic/coupled action model in the unified workspace and obtains an audit-gated simulated transition. It does not execute that transition on the host or in the physical world.

These are semantic bridges, not newly acquired unrestricted reasoning. The teacher's semantic action symbol must match the corresponding model name or action predicate. No arbitrary command execution or external tool access is provided.

## Interactive example

From the repository root:

```text
python -m ix_zero.language_demo
```

The example fits the language engine using an explicit synthetic teaching corpus, separately audits it, learns and promotes a two-bit XOR operator from its observations, and presents a four-object scene. Try:

```text
inspect red cube
compare green cube blue cone
rule green cube red sphere
inspect orange cube
exit
```

The third request yields a numeric result from the learned operator. The fourth abstains because the word is unsupported. `inspect` and `compare` produce structured interpretations, not invented explanations. The demo does not persist its training journal; the main CLI supports persistent custom episodes.

For manual use, launch `python -m ix_zero`, then use `lnew ["lang"]`, `lobserve ["lang", "inspect red cube", [["a","cube","red"]], ["inspect","a"]]`, and more contrasting complete demonstrations. The first example alone does not resolve all grammar hypotheses. After sufficient examples and three fresh passing audits for the desired arity (`laudit` has the same arguments as `lobserve`), `lcontext ["lang", [["new","cube","red"]]]` selects a scene; `say inspect red cube` requests an audit-gated interpretation. `lstatus ["lang"]` exposes the learned lexicon and possible role orders. `lsave` persists a replay journal. `help` lists the remaining commands.

## Evaluation protocol

Run:

```text
python -m ix_zero.language_experiment --trials 24 --seed 99000 --check --output results/language_campaign.json
python -m ix_zero.language_demo --check
python -m unittest discover -s tests -p test_language.py -v
```

The evaluator generates a nine-word vocabulary covering three kinds, three attributes and three actions. Two conditions use conventional English-like content words and randomized artificial words with randomized role orders. Both use 24 seeds (48 episodes). In each episode, 30 complete utterances are training examples, six disjoint utterances are reserved for validation, and 207 distinct unseen utterances are reserved for testing. Object IDs are distinct across the partitions. The finite scene includes every kind/attribute combination in randomized order. Exact-utterance memorization has no matching test strings. The corpus is synthetic and developer-designed; it does not establish general linguistic competence.

The fixed-partition learner produced 9,936 correct interpretations and zero observed incorrect interpretations or abstentions in the 48 matched-grammar episodes. Generated sentences round-tripped to 9,936 correct intents; this is consistency within the induced model. The unknown-word and duplicated-referent checks (1,152 total) all abstained. In 48 additional deliberately shifted semantic cases, the original interpretation was incorrect by construction; a fresh contradictory audit quarantined every affected episode. The prior wrong predictions are recorded in the campaign's feedback fields.

For 24 additional seeded episodes, the IX policy selects examples from an unlabeled training pool, then receives the separately reserved audit partition. It used 132 training labels in total and answered 4,968/4,968 held-out utterances correctly. A passive learner received the same number of training labels per episode, selected in the fixture's random order, plus the same allocated audit partition; it answered 3,220 correctly with zero observed errors and 1,748 abstentions. This comparison is specific to the supplied grammar, candidate pool and inquiry scoring; active inquiry sees unlabeled candidates that the passive learner does not use.

For 24 integration trials, English-like utterances independently rendered by the evaluator invoked the already learned two-bit XOR operator and its relational transfer. Each bridge was checked on all 16 argument pairs: 384/384 correct numeric results and 384/384 correct relational results. Of these, 72 numeric pairs and 264 relational pairs were held out from the respective target training/audit sets; all were correct. Retraction blocked both numeric and downstream relational evaluation in every trial. The bridge test includes previously seen pairs and is not a general intelligence benchmark.

Raw evaluation input partitions, predictions, negative controls, operator/relational event journals, strategy traces and shifted outcomes are in `results/language_campaign.json`. The symbolic simulator bridge, synonyms, ambiguity, ordering, freeze, checkpoint replay and corruption are also tested in `tests/test_language.py`.

## Limits

The engine is supervised by grounded demonstrations, not unsupervised natural-language learning. It supports only three- and five-token constructions, one attribute and kind per object, one meaning per token, and supplied finite scenes. Articles, general syntax, negation, recursive clauses, polysemy, conversational pragmatics, open-world reference, perceptual grounding and arbitrary language are not supported. It does not independently infer the meanings of arbitrary English text. Hidden meaning changes may evade validation until contradictory feedback arrives. No human-level intelligence, consciousness, or general-purpose conversation is demonstrated.
