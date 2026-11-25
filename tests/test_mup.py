import math

import numpy as np
import pytest
import torch

from mupx.model import GPT, Config, param_groups
from mupx.train import coord_check, growth, main, report_sweep, run

KW = dict(base_width=32, head_dim=16, depth=2)


def _model(width, mup):
    torch.manual_seed(0)
    return GPT(Config(width=width, mup=mup, ctx=32, **KW))


@pytest.mark.parametrize("mup", [False, True])
def test_param_groups_cover_every_parameter_once_with_the_right_lr(mup):
    m = _model(128, mup)  # m = 4
    groups = param_groups(m, 1e-2)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    assert groups[0]["lr"] == pytest.approx(1e-2 / 4 if mup else 1e-2)
    assert groups[1]["lr"] == 1e-2
    assert len(groups[0]["params"]) == 4 * 2  # four matrices per block


def test_mup_initialisation():
    sp, mu = _model(512, False), _model(512, True)  # m = 16
    w_sp, w_mu = sp.blocks[0].fc.weight, mu.blocks[0].fc.weight
    assert w_sp.std().item() == pytest.approx(0.02, rel=0.05)
    assert w_mu.std().item() == pytest.approx(0.02 / 4, rel=0.05)
    assert mu.head.weight.std().item() == pytest.approx(0.02, rel=0.05)  # readout unscaled
    assert mu.emb.weight.std().item() == pytest.approx(0.02, rel=0.05)  # embeddings unscaled
    torch.manual_seed(0)
    zero = GPT(Config(width=512, mup=True, ctx=32, mup_readout="zero", **KW))
    assert zero.head.weight.abs().max() == 0
    assert zero(torch.randint(0, 256, (2, 8))).abs().max() == 0  # logits start at exactly 0


def test_mup_initial_logits_shrink_with_width():
    """With the 1/m multiplier, a std-σ readout gives initial logits with std ∝ 1/√m."""
    x = torch.randint(0, 256, (4, 16))
    std = {w: _model(w, True)(x).std().item() for w in (32, 512)}  # m = 1, 16
    assert std[512] / std[32] == pytest.approx(0.25, rel=0.25)


def test_mup_at_base_width_is_sp_up_to_the_readout():
    """At m = 1 every μP multiplier is 1, so with the same readout the two models agree."""
    sp, mu = _model(32, False), _model(32, True)  # base_width = 32
    x = torch.randint(0, 256, (2, 8))
    torch.testing.assert_close(mu(x), sp(x))  # same seed, same init, same forward
    assert [g["lr"] for g in param_groups(mu, 1e-3)] == [g["lr"] for g in param_groups(sp, 1e-3)]
    # the 1/d_head variant is a different model even at m = 1
    torch.manual_seed(0)
    mu_d = GPT(Config(width=32, mup=True, ctx=32, mup_attn="d", **KW))
    mu_d.head.weight.data.copy_(sp.head.weight)
    assert not torch.allclose(mu_d(x), sp(x))


def test_mup_logits_are_divided_by_width_multiplier():
    m = _model(128, True)
    torch.nn.init.normal_(m.head.weight, std=0.02)
    x = torch.randint(0, 256, (2, 8))
    raw = m.head(m.norm(m(x, return_acts=True)[1][-2]))
    torch.testing.assert_close(m(x), raw / 4)


def test_coordinate_check_mup_is_flat_and_sp_is_not():
    """Logit updates after a few Adam steps: SP grows with width, μP stays roughly flat."""
    data = np.random.default_rng(0).integers(0, 256, 50_000).astype(np.uint8)
    widths = [32, 128, 512]
    rows = []
    for mup in (False, True):
        rows += coord_check(widths, mup, data, "cpu", steps=3, lr=1e-2, bs=4, ctx=16, **KW)
    g_sp, g_mu = growth(rows, False), growth(rows, True)
    assert g_sp > 3.0  # 16× width: SP logit updates blow up
    assert 0.3 < g_mu < 2.0
    # hidden-layer updates too
    assert growth(rows, False, layer=1) > 1.5 * growth(rows, True, layer=1)


def test_run_and_report(tmp_path):
    g = np.random.default_rng(0)
    data = (g.integers(0, 256, 20_000).astype(np.uint8), g.integers(0, 256, 5_000).astype(np.uint8))
    r = run(64, True, 2.0**-8, data, 10, "cpu", bs=4, ctx=16, **KW)
    assert math.isfinite(r["val_loss"]) and r["params"] > 0
    text = report_sweep([r, {**r, "lr": 2.0**-7, "val_loss": None}])
    assert "μP" in text and "✗" in text and "**" in text
    assert (r["mup_attn"], r["mup_readout"]) == ("sqrt", "normal")
    assert "q·k/d_head" in report_sweep([{**r, "mup_attn": "d", "mup_readout": "zero"}])


def test_sweep_cli_resumes(tmp_path):
    args = [
        "--exp",
        "sweep",
        "--device",
        "cpu",
        "--out",
        str(tmp_path),
        "--widths",
        "64",
        "--log2-lrs",
        "-9",
        "-8",
        "--steps",
        "10",
        "--depth",
        "1",
    ]
    main(args)
    n = len((tmp_path / "sweep.jsonl").read_text().splitlines())
    assert n == 4
    main(args)
    assert len((tmp_path / "sweep.jsonl").read_text().splitlines()) == n
