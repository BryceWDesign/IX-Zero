# Integrated neural-symbolic inquiry

## Architecture

`ix_zero.hybrid.Hybrid` integrates three independently trained dense networks with executable symbolic inquiry. `ix_zero.neural.Dense` implements a hidden tanh layer, sigmoid outputs, seeded initialization and online binary-cross-entropy gradient updates in Python. No pretrained model or remote service is used.

| Network | Layer sizes | Input and training objective |
|---|---|---|
| Recognition | 3, 6, 1 | Normalized three-channel samples; supervised binary calibration |
| World prediction | 16, 16, 10 | Family, domain size, arguments and action; predict bits of observed output |
| Strategy | 18, 10, 1 | World features and symbolic partition statistics; estimate fraction of candidates eliminated by an observed experiment |

The perception decoder interprets a finite word as a row of bit markers and a spatial state as a single active marker in a square raster. The word learner reuses the existing cross-environment `Inquiry` grammar. The spatial learner enumerates 32 candidate programs (four rotations, two reflections, two strides, two boundary rules). The host supplies unlabeled training and audit pools, plus measurements only when requested. The learner chooses discriminating inputs or requests a different environment if necessary. Neural uncertainty and the strategy estimate break exact ties in symbolic query ranking; they do not discard candidate programs. Recognition is used on training observations, not on independent audit measurements. Once all required audits pass with a single surviving program across the declared scope, symbolic prediction is enabled. Contradictory audits and later challenges block prediction.

For spatial goals, breadth-first search traverses the verified transition graph; neural predictions only order alternative actions at equal search depth. Unreachable goals return no plan. There is no external actuator or authorization to execute plans.

## Memory and transfer

A checkpoint saves configuration, event history, pending request and network parameters. Restore reconstructs the model, replays training and audits, and requires canonical equality rather than trusting serialized parameters. The SHA-256 digest detects accidental corruption; it is not an authentication mechanism. Transfer copies recognition and strategy parameters from a verified source inquiry into a different family, resets the world network, and requires fresh target training and audits. The source journal is replayed on restore. Transfer is bounded to one source inquiry and assumes a compatible sensor encoding; it does not transfer the source symbolic law into a different grammar.

## Evaluation protocol

`ix_zero.hybrid_experiment` generates synthetic word and spatial transitions and bounded three-channel rasters with randomized sensor polarities and noise. Calibration samples and held-out calibration samples are separate. Training, audit, and final evaluation inputs are disjoint. Within each seed, all control arms receive matching partitions and the same underlying rules; per-input noise is deterministic. The evaluator is separate from the learner, but both are developer-authored, not independently blinded. Six seeds beginning at 410000 and six at 420000 form the two campaigns. All six control arms are recorded, along with transferred inquiries and adversarial cases.

| Configuration | Validated acquisitions | Correct held-out predictions | Abstentions | Requests |
|---|---:|---:|---:|---:|
| Hybrid | 24/24 | 14,088 | 0 | 509 |
| Unguided trained networks | 24/24 | 14,088 | 0 | 507 |
| Oracle observations (no raster decoding) | 24/24 | 14,088 | 0 | 507 |
| Calibrated non-neural centroid | 24/24 | 14,088 | 0 | 507 |
| Uncalibrated threshold | 8/24 | 4,696 | 9,392 | 250 |
| Frozen networks | 2/24 | 1,174 | 12,914 | 164 |

No incorrect validated predictions occurred in these arms. All three trainable networks changed weights in each hybrid and unguided run. Recognition in the hybrid arm passed 2,304/2,304 held-out calibration samples after training, compared with 1,294 before training. The centroid baseline also passed all 2,304 samples. The hybrid arm's raw neural world predictions were correct for 1,502/13,032 word inputs and 56/1,056 spatial inputs after training. Audited symbolic predictions were correct on all these inputs. The guided arm required two more requests than unguided; this protocol does not demonstrate neural superiority or improved query efficiency. Lower request counts in failed controls do not signify efficiency.

The hybrid arm passed 144 spatial planning checks, including 11 independently verified unreachable goals. The 24 cross-family transfers (12 in each direction, matching sensor conditions) acquired a target program with no new calibration labels and made 14,088 correct held-out predictions. These reuse the campaign seeds and are not independent additional test families. Across all recorded arms, 184 final checkpoint replays matched. Sixteen negative controls checked audit corruption, denied environment access, contradiction after promotion, and laws outside the supplied grammar: each blocked or invalidated promotion. For later contradiction cases, recorded held-out predictions precede invalidation; the separate `invalidated` field verifies prediction was blocked afterward.

## Reproduction

From the repository root:

```sh
python -m ix_zero.hybrid_experiment --trials 6 --seed 410000 --check --output hybrid-primary.json
python -m ix_zero.hybrid_experiment --trials 6 --seed 420000 --check --output hybrid-confirmation.json
python -m unittest discover -s tests -v
```

See [primary results](../results/hybrid_primary.json) and [confirmation results](../results/hybrid_confirmation.json) for every partition, request, measurement, prediction, plan, final network state, weight digest and control outcome. Floating-point serialization may vary across Python/platform builds; the assertions test behavior rather than requiring cross-platform byte equality.

## Host interface

Use `python -m ix_zero.hybrid_cli` for a local JSON-lines interface. Create an inquiry using a declared scope and disjoint input pools. A word example with a single width:

```json
{"command":"new","family":"word","widths":[2],"initial":2,"train":[[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1]],"audit":[[0,2],[1,2],[2,2],[3,2]],"seed":42,"budget":2,"mode":"oracle"}
{"command":"next"}
```

The host answers each returned request with `{"command":"answer","value":...}`. In oracle mode, training and audit values are the exact numerical result (or coordinate pair for grids). In hybrid mode, first submit labeled calibration rows with `{"command":"calibrate","rows":[[[0.1,0.1,0.1],0],[[0.9,0.9,0.9],1]]}`. This is illustrative syntax only; use sufficient calibration observations for the actual encoding. Training values are arrays of three-channel samples, ordered by bit or row-major cell. The `answer_ppm` command accepts a local ASCII P3 file with the matching dimensions, not a camera feed. Audits always receive exact independent measurements. Environment requests receive new `train` and `audit` pools, or the host can send `deny`. `next` must return `complete` before prediction can be enabled.

Other commands: `status`, `predict` (`input`), `neural_prediction` (`input`, explicitly unvalidated), `challenge` (`input`, `value`), `plan` (`start`, `goal`, `width`), `save` and `load` (`path`), and `transfer` (new family, widths, initial, training and audit pools, seed and budget). Python clients can call the corresponding `Hybrid` methods. The console terminates at end of input.

## Limitations

The observation representation, program families, training targets, resource limits and evaluator interface are designed by the developer. Synthetic raster classification is not general visual or multimodal understanding; transfer is narrow calibration/strategy reuse, not universal skill transfer. Raw neural forecasts can be wrong even after training. The experiments do not measure broad real-world agency, general natural-language competence, consciousness, or independent AGI benchmark performance. The candidate grammar cannot learn an arbitrary out-of-family law. The supplied reference evidence includes these limitations and negative outcomes.
