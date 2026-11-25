# Notice

Copyright (c) 2026 Bryce Lovell. IX-Zero is source-available for noncommercial evaluation, not open source. See [LICENSE](LICENSE) and [LICENSE-COMPONENTS.md](LICENSE-COMPONENTS.md).

## Incorporated components

- **IX interpreter:** the canonical Python IX interpreter from the IX-main archive, commit `3a629832f6b3057aa4656be4314080dd20bac491`, is vendored without source modifications in `ix_zero/_ix_vendor/`. Its original MIT copyright and license are retained at `ix_zero/_ix_vendor/LICENSE`. The IX-Zero evaluation license does not restrict rights granted by that MIT license.
- **IX-Sally:** fourteen selected Python modules, with namespace adaptations where needed, and two package initializers are incorporated under `ix_zero/_sally_vendor/`. The original IX-Sally evaluation license and copyright are preserved at `ix_zero/_sally_vendor/LICENSE`. The release-specific component addendum defines the intended narrow permission for this integrated distribution, effective upon authorized publication by the rights holder. No general relicensing of IX-Sally is intended. Source and adapted hashes are recorded in `docs/unified_sally_provenance.json` and `docs/scalar_donor_provenance.json`.
- **Other donor projects:** selected design principles were reviewed and independently implemented. BlackFox-AGI and IX-AGI mechanisms in `relational.py` are selective Python reimplementations informed by the supplied Julia source, not unmodified native runtime modules. See `docs/DONORS.md`, `docs/DISCOVERY_DONORS.md`, `docs/UNIFIED.md` and the provenance inventories. The complete original Sally comparison archive and native Julia donor runtimes are not included; the native Julia comparison adapter has not been executed.

## Evidence and limitations

The experimental results are developer-designed and locally executed. The distributed research evidence includes historical experiments, failures, abstentions and negative controls. Routine engineering transcripts are not part of the research dataset. See [results](results/README.md) for evidence scope and [release verification](docs/RELEASE_VALIDATION.md) for the current build checks. No independent AGI certification or production readiness is claimed.
