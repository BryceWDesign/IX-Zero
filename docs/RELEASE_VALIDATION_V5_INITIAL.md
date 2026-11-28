# Release verification — v5.0.0 research candidate

Local verification on Linux, Python 3.13.5. The GitHub-hosted Windows/Ubuntu × Python 3.11/3.13 matrix has not been executed. No GitHub push was made. Green engineering checks do not indicate achieved AGI or completion of the entire research roadmap.

| Check | Actual result |
|---|---|
| Frozen v4 archive manifest | 248/248 verified |
| Frozen v4 unit tests | 286 passed |
| Final v5 unit tests | 318 passed (286 retained + 32 new) |
| All exact workflow command lines on final source | 31/31 exited 0 locally |
| Primary and confirmation research campaigns | Both executed with 12 seeds; negative outcomes retained |
| Primary deterministic same-seed rerun | Byte-identical JSON |
| Offline wheel build and installation | Passed on local Python 3.13.5 |
| Installed-package typed arithmetic task | Passed, result 27 |
| Original v4 result data files | 82/82 byte-identical |
| Original Sally native selected tests | 8 passed; not a full donor suite |
| Native Julia/Ada donor execution | Not performed; runtimes unavailable |
| Actual live HTTPS attempt | DNS failure; no successful remote retrieval |
| Public ARC-AGI task diagnostic | 0/1 exact; abstention |
| DreamCoder and PushGP comparisons | Not performed |

Final local test and CI transcripts, provenance, raw observations, and failure cases are in `results/v5/`. The historic v4 verification document is preserved in `RELEASE_VALIDATION_V4.md`. The full external benchmark, independent evaluation, general architectural self-modification, general language learning, unrestricted software operation and native Windows validation remain unresolved. See `RESEARCH_V5.md` for precise experimental methods and limitations.

`MANIFEST.sha256` records distributed files excluding itself. It checks integrity, not authorship or scientific validity. The historical evidence files were restored from the original ZIP after the CI smoke commands that write their default output paths. Reruns should write explicit new paths to avoid overwriting historical results.
