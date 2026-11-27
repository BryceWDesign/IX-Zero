# Release verification

IX-Zero v4.0.0. Local verification: Linux, Python 3.13.5. The workflow commands were executed locally; the GitHub-hosted matrix is not represented as executed.

| Check | Result |
|---|---|
| Unit tests (`python -m unittest discover -s tests -v`) | 286 passed, including 268 retained tests |
| Commands in `.github/workflows/ci.yml` | 30 passed locally |
| Native reasoning evaluation, seeds 810000 and 820000 | Both six-seed campaigns passed executable checks |
| Same-seed reproductions | Both campaign JSON files reproduced byte for byte |
| Offline wheel build and isolated installation | Passed |
| Installed-package native reasoning evaluation and CLI | Passed |
| Historical evidence from v3.0.0 | 80/80 JSON, TXT and TSV records byte-identical |

The retained engine and its tests are unchanged except for version display metadata in the original package and console. The new workspace is additive; the original interfaces remain available. Historical results describe their original engines and protocols. See [workspace methods](GENERAL.md), [continual methods](CONTINUAL.md), [hybrid methods](HYBRID.md) and [evidence index](../results/README.md). The historical control's incorrect predictions are preserved. The new campaigns include abstentions in the reuse-disabled control, corrupted audits and unsupported tasks. No external AGI benchmark score is reported.

The root manifest records SHA-256 hashes of distributed files, excluding the manifest itself. It is an integrity check, not a signature or proof of scientific correctness. From a pristine checkout on Linux, run `sha256sum -c MANIFEST.sha256` before experiments that write into the results directory. Experiments that generate result files should use a separate checkout or new output path.

GitHub-hosted CI for this release, native Windows execution, Python 3.11 execution, independent expert evaluation, and broader benchmark competence remain unverified. No universal correctness, general-purpose language competence, execution-speed gain, security certification, or general intelligence is claimed.
