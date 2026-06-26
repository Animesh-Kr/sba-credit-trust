"""Smoke tests for the DL models + focal loss (forward + one optimisation step)."""
import numpy as np
import torch

from src.models import tabular_dl


def _batch(n=64, n_num=7, cards=(5, 8)):
    x_num = torch.randn(n, n_num)
    x_cat = torch.stack([torch.randint(0, c, (n,)) for c in cards], dim=1)
    y = (torch.rand(n) < 0.2).float()
    return x_num, x_cat, y


def test_focal_loss_positive_and_finite():
    loss = tabular_dl.FocalLoss()
    logits = torch.randn(100)
    targets = (torch.rand(100) < 0.2).float()
    val = loss(logits, targets)
    assert val.item() >= 0 and torch.isfinite(val)


def test_mlp_forward_shape_and_step():
    cards = [5, 8]
    net = tabular_dl.build("mlp", n_numeric=7, cardinalities=cards, hidden=(32, 16))
    x_num, x_cat, y = _batch(cards=tuple(cards))
    out = net(x_num, x_cat)
    assert out.shape == (64,)
    loss = tabular_dl.FocalLoss()(out, y)
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())


def test_ft_transformer_forward_shape_and_step():
    cards = [5, 8]
    net = tabular_dl.build("ft_transformer", n_numeric=7, cardinalities=cards, d_token=16, n_blocks=1, n_heads=2)
    x_num, x_cat, y = _batch(cards=tuple(cards))
    out = net(x_num, x_cat)
    assert out.shape == (64,)
    loss = tabular_dl.FocalLoss()(out, y)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_predict_proba_in_unit_interval():
    from src.models import trainer

    cards = [5, 8]
    net = tabular_dl.build("mlp", n_numeric=7, cardinalities=cards, hidden=(16,))
    x_num = np.random.randn(50, 7).astype(np.float32)
    x_cat = np.stack([np.random.randint(0, c, 50) for c in cards], axis=1)
    p = trainer.predict_proba(net, x_num, x_cat)
    assert p.shape == (50,)
    assert (p >= 0).all() and (p <= 1).all()
