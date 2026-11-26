# Release verification

IX-Zero v2.0.0. Local verification: Linux, Python 3.13.5.

| Check | Result |
|---|---|
| Unit tests (`python -m unittest discover -s tests -v`) | 257 passed |
| Commands in `.github/workflows/ci.yml` | 27/27 passed locally, including the hybrid evaluator |
| Primary hybrid campaign (seed 410000, six trials) | Passed |
| Confirmation hybrid campaign (seed 420000, six trials) | Passed |
| Checkpoint replay across recorded arms | 184/184 |
| Negative hybrid controls | 16/16 blocked or invalidated |

The complete hybrid results and matched control arms are in [results](../results/README.md). See [hybrid methods](HYBRID.md) for reproduction and the limitations of the synthetic evaluation. The retained AEI research records refer to their original evaluations; the current build passed the two-trial AEI CI command. Historical numerical evidence is retained; the frozen evaluation file was relocated and its protocol label normalized for clarity.

The root manifest records SHA-256 hashes of distributed files, excluding the manifest itself. It is an integrity check, not a signature or proof of scientific correctness. From a pristine checkout on Linux, run `sha256sum -c MANIFEST.sha256` before experiments that write into the results directory.

GitHub-hosted matrix execution, native Windows verification, Python 3.11 execution, and independent expert or benchmark validation are not established by the local checks. No execution-speed comparison is claimed.
