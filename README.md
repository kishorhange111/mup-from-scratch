# mup-from-scratch

[![CI](https://github.com/kishorhange111/mup-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/mup-from-scratch/actions/workflows/ci.yml)

Under standard parametrisation (SP), the best learning rate changes as a model gets wider,
so tuning a small proxy model doesn't tell you how to train the big one. **Maximal Update
Parametrisation** (μP; Yang et al. 2022, *Tensor Programs V*) rescales initialisation,
learning rates and the output so every layer's update stays Θ(1) as width grows. The best lr
for a small model then transfers. This repo implements μP by hand for a GPT trained with
Adam and checks both claims: the coordinate check and lr transfer.

## The rules ([`model.py`](src/mupx/model.py))

The width multiplier is `m = width / 128`. Head dim is fixed at 64, so heads scale with
width.

| tensor | SP (GPT-2) | μP (Adam, output-multiplier form) |
|---|---|---|
| embeddings | std 0.02, lr η | std 0.02, lr η |
| hidden matrices (qkv, proj, MLP) | std 0.02, lr η | std 0.02/√m, **lr η/m** |
| readout | std 0.02 | std 0.02, logits **× 1/m** |
| attention scores | `q·k/√d_head` | `q·k/√d_head` (head dim is fixed) |

At m = 1 every μP factor is 1, so **μP at the base width is exactly SP** (a test checks
this). Two common variants are kept as switches, because the first run used both and they
were what made it fail. See [Ablations](#ablations-what-went-wrong-in-the-first-run).

The reason: Adam's update to every weight is about lr in size regardless of width. A hidden
unit sums `fan_in` such updates, so under SP its change grows like width, and so does the
change in the logits. To stay stable, the lr has to shrink as the model grows. Dividing
the hidden lr by m cancels the fan-in sum. The 1/m logit multiplier does the same for the
readout, and it makes the initial logits shrink like 1/√m.

## Experiments ([`train.py`](src/mupx/train.py))

- **Coordinate check** (`--exp coord`): widths 128 to 2,048 train for five Adam steps. Each
  layer's output change since initialisation (RMS, on a fixed batch) is recorded per step.
  The μP curves should be flat in width; SP's grow.
- **lr transfer** (`--exp sweep`): widths 128, 256, 512 and 1,024, depth 4, with a grid of
  log₂ lr from −14 to −6. Each run trains for 600 steps on Python-stdlib bytes, and the
  script records final val loss. Under μP the arg-min lr should be the same at every width;
  under SP it should move left.

The tests check:
- every parameter lands in exactly one optimizer group with the right lr;
- the μP init scales and the zero readout;
- the logit multiplier;
- **a coordinate check on CPU** (16× width): SP's logit update grows by more than 3×, while
  μP's stays within 0.3–2×;
- μP equals SP at m = 1;
- initial logits shrink like 1/√m.

```
pytest -q                                                # 9 tests, CPU
python -m mupx.train --exp coord --out results --device cuda
python -m mupx.train --exp sweep --out results --device cuda   # resumable
```

## Results (Tesla T4)

Raw logs are in [`results/`](results).

### Coordinate check

This is the RMS change of each layer's output after 5 Adam steps at lr 0.01:

| width | SP block 4 | SP logits | μP block 4 | μP logits |
|---|---|---|---|---|
| 128 | 28.3 | 3.48 | 28.3 | 3.48 |
| 256 | 94.6 | 3.64 | 25.1 | 3.36 |
| 512 | 318 | 7.35 | 28.5 | 3.63 |
| 1024 | 350 | 10.6 | 24.3 | 3.54 |
| 2048 | 1,120 | 14.2 | 23.9 | 3.55 |

From width 128 to 2,048, SP's last hidden layer moves **40× more** and its logits 4.1× more.
Under μP both are flat: ×0.84 and ×1.0. The width-128 rows are identical because μP is SP
at m = 1.

### Learning-rate transfer

This is val loss in nats/byte after 600 steps. Bold marks the best lr at each width.

| width | SP best log₂ lr | SP best loss | SP at lr 2⁻⁸ | **μP best log₂ lr** | μP best loss |
|---|---|---|---|---|---|
| 128 | −8 | 1.760 | 1.760 | **−8** | 1.760 |
| 256 | −9 | 1.549 | 1.686 | **−8** | 1.704 |
| 512 | −10 | 1.482 | 2.261 | **−8** | 1.666 |
| 1024 | −11 | **1.459** | 2.422 | **−8** | 1.628 |

The full 9-point grids are in [`results/sweep.md`](results/sweep.md).

- **The optimum transfers under μP and drifts under SP.** μP's best lr is 2⁻⁸ at every width.
  The grid runs from 2⁻¹² to 2⁻⁴, so this optimum is bracketed on both sides. SP's best lr
  halves with every doubling of width, 8× smaller across the range, as the coordinate check
  predicts.
- **Transferring SP's small-model lr fails.** At lr 2⁻⁸, SP gets *worse* as it grows
  (1.760 → 2.422). μP at the same lr gets better with every doubling (1.760 → 1.628).
- **SP tuned separately at each width is still better here.** At width 1,024 it reaches
  1.459 against μP's 1.628, and it also wins at 256 and 512. μP's promise is that one sweep
  on the small model replaces a sweep at every size. It does not promise a better model, and
  at this scale (≤ 1,024 wide, 600 steps) it is not one. Per-layer multipliers (embedding,
  readout, attention), which the paper also tunes on the proxy, would be the next thing to
  tune. They are left at 1 here.

### Ablations: what went wrong in the first run

The first sweep used two common μP variants: a zero-initialised readout and `q·k/d_head`
attention. Its μP models trained far worse than SP, with a best of 2.132 against 1.459.
Since μP at m = 1 should *be* SP, that pointed to a bug rather than a finding.

| μP variant (width 128, lr 2⁻⁸, 600 steps, CPU check) | val loss |
|---|---|
| SP | 1.782 |
| μP, zero readout | 2.394 |
| μP, std-σ readout (now the default) | **1.782** |

- **Zero readout.** With zero logits, no gradient reaches the hidden layers until the
  readout has moved, and on this budget the model never catches up. Its sweep
  ([`ablations/sweep_zero_readout.jsonl`](results/ablations)) still transfers the lr (best
  2⁻⁹ or 2⁻⁸ at every width), but its best loss at each width is 0.5–0.6 higher.
- **`q·k/d_head`.** μP prescribes it for a head dim that grows with width. Here the head dim
  is fixed at 64, so it only made every attention logit 8× smaller. Both are now
  `mup_readout="zero"` and `mup_attn="d"`.

The lesson: an "equals SP at m = 1" test catches both, and it is now in the suite.
