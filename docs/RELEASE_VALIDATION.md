# Release verification — grounded v5.0.0 research candidate

Actual execution: Linux, Python 3.13.5. No GitHub push or hosted Windows/Ubuntu × Python 3.11/3.13 matrix execution. Local engineering success does not establish AGI, or completion of the research roadmap. The current claim boundary is in `RESEARCH_V5_GROUNDED.md`.

| Verification | Observed result |
|---|---|
| Parent expanded ZIP integrity | Passed |
| Parent SHA-256 manifest | 303/303 verified against archive bytes |
| Parent unit suite | 339 passed before changes |
| New unit suite | 361/361 passed (339 retained, 22 added) |
| Exact dependency-free workflow command lines | 35/35 passed in a separate checkout; all executable CI sources match final source |
| Historical result files | 116/116 byte-identical to preceding build |
| Grounded learning campaigns | Two 12-seed campaigns, failures and controls retained |
| Semantic transfer/arity campaigns | Two 12-seed campaigns, failures and controls retained |
| Deterministic reproduction | Primary action-learning and primary transfer campaign files reproduced byte-for-byte on final source |
| Public ARC-AGI-1 diagnostic | 3/6 relational, 1/6 prior objects, 0/6 original; convenience sample and two repaired tasks previously exposed |
| Actual Chromium semantic form | 64/64 held-out predictions, 2/2 goals, 260 actual button clicks; optional dependency, executed directly from packaging tree |
| Offline wheel build and installed-package smoke test | Passed |
| Live HTTPS | Attempted by Zero; DNS failed, no successful live retrieval |
| Native Julia/Ada and external named synthesis comparisons | Not executed; toolchain/access prerequisites unavailable; no DreamCoder/PushGP result |

Full command output, parent check, source hashes, campaign data, and reproduction hashes are preserved in `results/v5_grounded/`. Prior release validation is preserved as `RELEASE_VALIDATION_V5_EXPANDED.md`, `RELEASE_VALIDATION_V5_INITIAL.md`, and `RELEASE_VALIDATION_V4.md`. `MANIFEST.sha256` covers all distributed files except itself and proves byte integrity, not scientific validity. No user credentials or paid services are required for the standard-library CI. The browser extra is optional and the local app is trusted. The release uses version 5.0.0, not a new published version.
