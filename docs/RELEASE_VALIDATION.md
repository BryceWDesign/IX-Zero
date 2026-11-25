# Release verification

IX-Zero v0.1.0. Local verification: Linux, Python 3.13.5.

| Check | Result |
|---|---|
| Unit tests (`python -m unittest discover -s tests -v`) | 239 passed |
| Commands defined in `.github/workflows/ci.yml` | 26/26 exited successfully in a local run |
| Offline wheel build | Passed |
| Wheel installation and one-seed AEI evaluation outside the source tree | Passed |
| Full primary AEI campaign, seed 310000, six trials | Passed; JSON byte-identical to reference |
| Full confirmation AEI campaign, seed 320000, six trials | Passed; JSON byte-identical to reference |

These checks cover the distributed executable source and policies. Both reference campaigns are available in [results](../results/README.md). To reproduce them, follow [AEI](EPISTEMIC.md) or the commands in the root [README](../README.md). The archive also contains earlier negative results; historical engineering counts refer to the build evaluated at the time.

The root manifest records SHA-256 hashes of the distributed files, excluding the manifest itself. It is an integrity check, not a signature or proof of scientific correctness. From a pristine checkout on Linux, run `sha256sum -c MANIFEST.sha256`. Run this before experiments that write into the results directory.

GitHub-hosted matrix execution, native Windows verification, Python 3.11 execution, and independent expert or benchmark validation are not established by the local checks. No execution-speed comparison is claimed.
