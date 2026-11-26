# Release verification

IX-Zero v3.0.0. Local verification: Linux, Python 3.13.5. All checks listed below were run on the local source distribution; the GitHub-hosted matrix is not represented as executed.

| Check | Result |
|---|---|
| Unit tests (`python -m unittest discover -s tests -v`) | 268 passed |
| Commands in `.github/workflows/ci.yml` | 29 passed locally |
| Continual evaluation, seed 510000 and seed 610000 | Both complete campaigns passed their executable checks |
| Same-seed primary continual campaign reproduction | Byte-identical JSON |
| Three-level skill campaign, seed 710000, six trials | Full-arm chain, prediction, planning and invalidation checks passed |
| Offline wheel build and isolated installation | Passed |
| Installed-package continual evaluator | Passed |
| Retained v2 reference result files | 76/76 byte-identical |

The third-level campaign's control arm produced a false promotion and incorrect held-out predictions. The error is documented and retained in the raw record, not represented as a passing scientific hypothesis. The full arm passed its specified checks; no external AGI benchmark score is reported. See [continual methods](CONTINUAL.md), [results](../results/README.md) and [hybrid methods](HYBRID.md). Historical results describe their original engines and protocols. The original single-inquiry hybrid interface remains available.

The root manifest records SHA-256 hashes of distributed files, excluding the manifest itself. It is an integrity check, not a signature or proof of scientific correctness. From a pristine checkout on Linux, run `sha256sum -c MANIFEST.sha256` before experiments that write into the results directory. Experiments that generate result files should use a separate checkout or new output path.

GitHub-hosted CI, native Windows execution, Python 3.11 execution, independent expert evaluation, and broader benchmark competence remain unverified. No universal correctness, execution-speed gain, or general intelligence is claimed.
