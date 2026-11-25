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
| readout | std 0.02 | **zero init**, logits **× 1/m** |
| attention scores | `q·k/√d_head` | `q·k/d_head` |

The reason: Adam's update to every weight is about lr in size regardless of width. A hidden
unit sums `fan_in` such updates, so under SP its change grows like width, and so does the
change in the logits. To stay stable, the lr has to shrink as the model grows. Dividing
the hidden lr by m cancels the fan-in sum. The 1/m logit multiplier does the same for the
readout. Zero-initialising the readout removes the width-dependent random logits at step 0.

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
  μP's stays within 0.3–2×.

```
pytest -q                                                # 7 tests, CPU
python -m mupx.train --exp coord --out results --device cuda
python -m mupx.train --exp sweep --out results --device cuda   # resumable
```

**Results are pending** from the Colab run.
