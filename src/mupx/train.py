"""Coordinate checks and learning-rate transfer, SP against μP.

    python -m mupx.train --exp coord --out results --device cuda
    python -m mupx.train --exp sweep --out results --device cuda

- ``coord``: for each width, train a few Adam steps at a fixed lr. After every step, record
  the RMS of each layer's output *change* since initialisation (``‖x_t − x_0‖ / √n``,
  on a fixed batch). Under μP these curves should be flat in width. Under SP they grow,
  and that growth is what moves the optimal lr.
- ``sweep``: train every (parametrisation, width, lr) for a fixed number of steps on byte
  sequences from the Python stdlib and record final val loss. Under μP the arg-min lr should
  be the same at every width; under SP it should shift left as width grows.

Finished runs are kept in ``<out>/<exp>.jsonl`` and skipped on restart.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sysconfig
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT, Config, param_groups


def load_corpus():
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    if not files:
        raise FileNotFoundError(root)
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            (va if zlib.crc32(rel.encode()) % 1000 < 50 else tr).append(fh.read())
    return (
        np.frombuffer(b"\n".join(tr), np.uint8).copy(),
        np.frombuffer(b"\n".join(va), np.uint8).copy(),
    )


def batch(data, bs, ctx, rng, device):
    ix = rng.integers(0, len(data) - ctx - 1, bs)
    x = torch.from_numpy(np.stack([data[i : i + ctx + 1] for i in ix]).astype(np.int64))
    return x[:, :-1].to(device), x[:, 1:].to(device)


def make(width: int, mup: bool, device: str, seed: int = 0, **kw) -> GPT:
    torch.manual_seed(seed)
    return GPT(Config(width=width, mup=mup, **kw)).to(device)


def coord_check(widths, mup: bool, data, device, steps=5, lr=1e-2, bs=16, ctx=64, **kw):
    """RMS change of each layer's output (embedding, blocks..., logits) per step and width."""
    rows = []
    for w in widths:
        m = make(w, mup, device, ctx=ctx, **kw)
        opt = torch.optim.Adam(param_groups(m, lr))
        rng = np.random.default_rng(0)
        x_fixed, _ = batch(data, bs, ctx, np.random.default_rng(123), device)
        with torch.no_grad():
            _, acts0 = m(x_fixed, return_acts=True)
        for t in range(1, steps + 1):
            x, y = batch(data, bs, ctx, rng, device)
            loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                _, acts = m(x_fixed, return_acts=True)
            delta = [
                float((a - a0).pow(2).mean().sqrt()) for a, a0 in zip(acts, acts0, strict=True)
            ]
            rows.append({"mup": mup, "width": w, "step": t, "delta_rms": delta})
    return rows


def run(width, mup, lr, data, steps, device, bs=32, ctx=128, seed=0, **kw) -> dict:
    train, val = data
    m = make(width, mup, device, seed, ctx=ctx, **kw)
    opt = torch.optim.AdamW(param_groups(m, lr), betas=(0.9, 0.95))
    base = [g["lr"] for g in opt.param_groups]
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    diverged = False
    for step in range(1, steps + 1):
        f = min(1.0, step / (steps // 10)) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g, b in zip(opt.param_groups, base, strict=True):
            g["lr"] = b * max(f, 0.01)
        x, y = batch(train, bs, ctx, rng, device)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            loss = F.cross_entropy(m(x).float().flatten(0, 1), y.flatten())
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    m.eval()
    vr = np.random.default_rng(999)
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=amp):
        vals = [
            float(F.cross_entropy(m(x).float().flatten(0, 1), y.flatten()))
            for x, y in (batch(val, bs, ctx, vr, device) for _ in range(10))
        ]
    v = float(np.mean(vals))
    return {
        "mup": mup,
        "mup_attn": m.c.mup_attn if mup else None,
        "width": width,
        "lr": lr,
        "steps": steps,
        "val_loss": None if diverged or not math.isfinite(v) else v,
        "params": sum(p.numel() for p in m.parameters()),
        "seconds": time.time() - t0,
    }


def _variant(r) -> tuple:
    return (r["mup"], r.get("mup_attn", "sqrt") if r["mup"] else None)


LABELS = {(False, None): "SP", (True, "sqrt"): "μP", (True, "d"): "μP with q·k/d_head"}


def report_sweep(recs) -> str:
    lines = []
    for variant, label in LABELS.items():
        rs = [r for r in recs if _variant(r) == variant]
        if not rs:
            continue
        lrs = sorted({r["lr"] for r in rs})
        widths = sorted({r["width"] for r in rs})
        lines += [
            f"### {label}: val loss (nats/byte) by width × log2(lr)\n",
            "| width | " + " | ".join(f"{math.log2(lr):.0f}" for lr in lrs) + " | best log2(lr) |",
            "|---|" + "---|" * (len(lrs) + 1),
        ]
        for w in widths:
            by = {r["lr"]: r["val_loss"] for r in rs if r["width"] == w}
            ok = {lr: v for lr, v in by.items() if v is not None}
            best = min(ok, key=ok.get) if ok else None
            cells = []
            for lr in lrs:
                v = by.get(lr, "missing")
                cells.append(
                    "·"
                    if v == "missing"
                    else "✗"
                    if v is None
                    else (f"**{v:.3f}**" if lr == best else f"{v:.3f}")
                )
            lines.append(
                f"| {w} | "
                + " | ".join(cells)
                + " | "
                + (f"{math.log2(best):.0f}" if best else "–")
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def report_coord(rows) -> str:
    lines = []
    for mup in (False, True):
        rs = [r for r in rows if r["mup"] == mup and r["step"] == max(x["step"] for x in rows)]
        if not rs:
            continue
        n = len(rs[0]["delta_rms"])
        names = ["embed"] + [f"block {i}" for i in range(1, n - 1)] + ["logits"]
        lines += [
            f"### {'μP' if mup else 'SP'}: RMS change of each layer's output after "
            f"{rs[0]['step']} Adam steps\n",
            "| width | " + " | ".join(names) + " |",
            "|---|" + "---|" * n,
        ]
        for r in sorted(rs, key=lambda r: r["width"]):
            lines.append(
                f"| {r['width']} | " + " | ".join(f"{v:.3g}" for v in r["delta_rms"]) + " |"
            )
        lines.append("")
    return "\n".join(lines)


def growth(rows, mup: bool, layer: int = -1) -> float:
    """Ratio of a layer's update size at the widest vs narrowest width (last step)."""
    last = max(r["step"] for r in rows)
    rs = sorted(
        (r for r in rows if r["mup"] == mup and r["step"] == last), key=lambda r: r["width"]
    )
    return rs[-1]["delta_rms"][layer] / rs[0]["delta_rms"][layer]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["coord", "sweep"], required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--widths", type=int, nargs="+", default=None)
    ap.add_argument("--log2-lrs", type=float, nargs="+", default=list(range(-14, -5)))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--mup-attn", choices=["sqrt", "d"], default="sqrt")
    ap.add_argument("--only", choices=["sp", "mup"], default=None, help="sweep one side only")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    data = load_corpus()
    path = os.path.join(a.out, f"{a.exp}.jsonl")
    recs = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    if a.exp == "coord":
        widths = a.widths or [128, 256, 512, 1024, 2048]
        if not recs:
            for mup in (False, True):
                recs += coord_check(
                    widths, mup, data[0], a.device, depth=a.depth, mup_attn=a.mup_attn
                )
            with open(path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
        text = report_coord(recs) + (
            f"\nLogit-update growth from width {min(widths)} to "
            f"{max(widths)}: SP ×{growth(recs, False):.1f}, "
            f"μP ×{growth(recs, True):.1f}.\n"
        )
    else:
        widths = a.widths or [128, 256, 512, 1024]
        done = {(*_variant(r), r["width"], r["lr"]) for r in recs}
        sides = {"sp": (False,), "mup": (True,), None: (False, True)}[a.only]
        for w in widths:  # narrow first: a cut-short session still yields complete small rows
            for mup in sides:
                for l2 in a.log2_lrs:
                    lr = 2.0**l2
                    if (mup, a.mup_attn if mup else None, w, lr) in done:
                        continue
                    r = run(w, mup, lr, data, a.steps, a.device, depth=a.depth, mup_attn=a.mup_attn)
                    with open(path, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    recs.append(r)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {'mup' if mup else 'sp '} w={w} "
                        f"lr=2^{l2:g} val={r['val_loss']} ({r['seconds']:.0f}s)",
                        flush=True,
                    )
        text = report_sweep(recs)
    with open(os.path.join(a.out, f"{a.exp}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
