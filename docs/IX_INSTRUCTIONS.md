# IX-Zero v0.1.0: executable cognitive instructions (historical integration evaluation)

## Source and design

The supplied IX-main archive's canonical `ix/` Python package is copied without source modifications to `ix_zero/_ix_vendor/`. Original MIT license retained. Archive commit: `3a629832f6b3057aa4656be4314080dd20bac491`. The donor `docs/LANGUAGE_REFERENCE.md` defines agent events, branches, replies, memory, assertions, traces and governed tools. Its `examples/cognitionkernel_wave6_contract.ix` records obligations, including prediction before trial, measured outcomes, contradiction handling and transfer. The donor runtime expressly describes contract metadata extraction as not executing cognition. This release therefore uses an actual agent event, not an attempt block presented as an intelligence mechanism.

## Runtime

1. Python derives facts directly from the selected Task: contradiction state, consensus over finite states, actual audit count, observation count, unused inputs and promotion eligibility.
2. The original IX parser and runtime execute `agent Zero` / `on advance` in `ix_zero/policies/cognition.ix`.
3. Exactly one IX reply is interpreted as a named native cognitive instruction. Missing/multiple replies, unknown instructions, required approvals and inconsistent actions are rejected. These new instruction names are an application protocol, not additions to the IX grammar.
4. Python performs the bounded operation or requests actual evidence. Inquire chooses an input using existing active-learning search and leaves at least three unused states for validation. Audit requests a user-selected unused input with an independently measured outcome. Promote invokes the existing core evidence gate. Complete ends the bounded task. Revise retains the blocked task and requests a new epoch. Exhausted explicitly abstains.
5. `policytrace` exposes the latest successful decision, source digest, facts and IX interpreter trace; it is an in-memory record, not an independently authenticated audit log.

## Reproduce

`python -m unittest discover -s tests -v` runs existing and added tests. `python -m ix_zero` starts the console. `new puzzle 2 5`, then `advance`; supply observed results via `teach`, and call `advance` again. When the engine asks for audits, use fresh inputs and obtain correct outputs independently rather than using the learner's own predictions. Three successful audits plus full source-space agreement permit promotion of a multi-step program. The `advance` action can perform promotion; it never fabricates an observation or audit. `policytrace` displays its actual decision evidence. The learner does not verify whether a human's supplied data really came from an independent oracle.

## Limitations

This is an instruction/control integration, not increased general intelligence. The finite-state hypothesis grammar and its computational limitations are unchanged. The policy cannot discover a representation beyond that grammar, infer unrestricted natural language, or autonomously revise its own implementation. The `advance` workflow is optional; the original manual commands and their original evidence checks remain. The policy is trusted local configuration, not a security sandbox for hostile code or untrusted IX programs. An edited policy can alter workflow choices within the host's checks, including choosing completion rather than promoting a valid skill. Python and the original IX runtime execute locally. No autonomous network, shell or external tool actions are connected by the integration. The original IX package is MIT, while the new integration and policy are evaluation-only; previously granted MIT permissions cannot be withdrawn by this release.

The new tests establish workflow execution, policy-dependent behavior, traces, evidence enforcement, rejection of malformed/multiple/unknown actions and pending approvals, finite-space exhaustion and revision requests. They do not establish an improvement in transfer accuracy or AGI. The prior raw experiment files are unchanged historical results. Windows and remote GitHub CI have not been executed for this release.

The subsequent representation discovery evaluation adds a separate IX policy and Python runtime. See `DISCOVERY.md`; this document describes the earlier integration only.
