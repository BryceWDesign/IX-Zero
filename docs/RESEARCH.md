# Research design

## Hypothesis

A finite learner may acquire skills more efficiently if verified source programs are made available as compressed grammar tokens, while novel tasks still require fresh evidence. The specific risk is semantic aliasing: two programs that are indistinguishable in one finite space may diverge in another (for example, rotate and reverse are identical on length-two vectors but not length-three vectors). Active selection can also stop on an incorrect unanimous answer when the true program is absent from the grammar. The experiments deliberately retain such errors.

No single benchmark proves AGI; the present experiments are generated from a small family of modular-vector operations, so they only probe restricted compositional learning, not broadly general cognition. The proposed next research gate would require independently specified, genuinely distinct tasks, hidden test data, compute-matched comparisons, scaling studies, multimodal acquisition and a credible generality criterion. None is claimed as achieved.

## Algorithm

State space S = {0,...,m-1}^w, w in {2,3}, m in [2,7]. Programs are compositions of at most three tokens from the fixed primitive grammar plus up to eight learned composite tokens. Expanded length is at most 12. There are at most 32 retained task records. All functions are total on valid states. `inc`, `dec`, `mix`, `unmix`, `mul`, `zero`, `flip`, and `gate` update the first element using modular arithmetic; `rot` and `rev` rearrange positions. Code does not evaluate generated Python or execute external tools.

For observations D, the version space contains all enumerated programs p such that p(x)=y for every (x,y) in D. If D is nonempty and all survivors agree at x, return their prediction; otherwise abstain. For an unseen input x, let n_y count candidates predicting y. The active query maximizes N² - sum(n_y²), equivalent to maximizing expected candidate elimination under the uniform syntactic prior. This prior is not calibrated and duplicate semantics can bias selection.

An audit freezes training. A failed or ambiguous audit blocks the task. Audit-gated prediction requires three separate passing audits, while concept promotion additionally requires unanimity for every input in the source state space. Promotion chooses the shortest lexicographic representative, which need not be semantically equivalent in another space. Future tasks snapshot available concepts and refit from fresh observations. Invalidation excludes a concept from future library snapshots; existing snapshots are retained with their own evidence, not transitively revoked. Drift is detected only when a counterexample is actually supplied; re-learning requires a new task. No unrestricted recursive self-modification is present.

Shortest-path search uses only action models with full source consensus and at least three passing audits in the planning space. Planning outputs simulated states; independent teacher execution checks the plans in `challenges.py`.

## Language choice

Python 3.11+ was selected for a zero-dependency experimental runtime, readable search algorithms, standard-library persistence and testing, and accessibility on Windows. It is not presumed to create AGI by virtue of language choice. Julia is an attractive numeric research language and is already used by two donors; Rust's ownership model and Ada/SPARK's deductive-verification facilities would be useful for a later hardened implementation. Neither would address the current theoretical generality limits. C/C++, Java/C#, JavaScript/TypeScript, Go, Lisp, Haskell and Prolog are reasonable alternatives for different performance, type-system, runtime and symbolic-programming tradeoffs; they were not exhaustively benchmarked or individually researched for this release. An exhaustive study of all programming languages and patents was not performed. No novel patentability claim.

## Research sources (public research, not copied code)

- François Chollet, *On the Measure of Intelligence* (2019), https://arxiv.org/abs/1911.01547 — skill-acquisition efficiency, explicit priors and generalization.
- Ellis et al., *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning*, https://arxiv.org/abs/2006.08381 — program induction and reusable library learning. IX-Zero does not implement DreamCoder's Bayesian or neural training machinery.
- Schölkopf et al., *Towards Causal Representation Learning*, https://arxiv.org/abs/2102.11107 — motivation for testing transfer rather than assuming invariance. IX-Zero does not establish real-world causal variables.
- Rolnick et al., *Experience Replay for Continual Learning*, https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html — background on retaining experience; IX-Zero retains explicit symbolic examples, not neural replay.
- Python standard library, https://docs.python.org/3.12/library/ ; Julia documentation, https://julialang.org/manual/ ; Rust ownership, https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html ; AdaCore SPARK overview, https://www.adacore.com/languages/spark .

## Limits and threat model

No open-world perception, learned natural language, general causal discovery, self-generated open-ended goals, real-world control, unsupervised code execution, or human-level reasoning. All learning is limited by the supplied grammar, finite search, explicit teacher and small memory. Audit pass is evidence about sampled inputs, not a correctness proof. The checkpoint hash detects accidental changes but is not authentication; a hostile file with a recomputed digest is not proven trustworthy. No sandbox security certification. Single-user, single-writer local evaluation only. The teacher is separate code in the same repository, not independent institutional verification. Synthetic seeds are public and the second protocol was designed after seeing the first experiment; treat all results as exploratory, not preregistered or independently blinded.
