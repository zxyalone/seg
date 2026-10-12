#!/usr/bin/env python3
"""SegResNetWithFeatures: same output as the bare MONAI SegResNet, decoder
features exposed finest first, and the CFC loss reaches the encoder.

    PYTHONPATH=nnUNet python experiments/test_segresnet.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nnUNet'))
from nnunetv2.training.nnUNetTrainer.phaser.lowcnr import contrast_consistency_loss  # noqa: E402
from nnunetv2.training.nnUNetTrainer.phaser.segresnet import SegResNetWithFeatures  # noqa: E402

torch.manual_seed(0)
net = SegResNetWithFeatures(1, 3, 3)
x = torch.randn(1, 1, 64, 64, 64)
y = net(x)
assert y.shape == (1, 3, 64, 64, 64)
ref = net.net(x)
assert torch.allclose(y, ref), 'wrapper changed the output'
f = net._last_dec_features
print('decoder features (finest first):', [tuple(t.shape) for t in f])
assert f[0].shape[2:] == x.shape[2:] and f[1].shape[2:] == (32, 32, 32)
with torch.no_grad():
    net(x)
teacher = [t.detach() for t in net._last_dec_features[:2]]
x2 = x.clone(); x2[..., 20:40, 20:40, 20:40] -= 0.5
net(x2)
region = torch.zeros(1, 1, 64, 64, 64); region[..., 14:46, 14:46, 14:46] = 1
loss = contrast_consistency_loss(net._last_dec_features[:2], teacher, region, 2)
loss.backward()
g = net.net.convInit.conv.weight.grad
print(f'CFC loss {loss.item():.4f}; first conv |grad| {g.abs().max().item():.2e}')
assert loss.item() > 0 and g.abs().max() > 0
# gradient checkpointing must not change the gradients
torch.manual_seed(1)
xa = torch.randn(1, 1, 48, 48, 48)
grads = []
for ckpt in (False, True):
    net.zero_grad(); net.checkpointing = ckpt
    net(xa).square().mean().backward()
    grads.append(net.net.convInit.conv.weight.grad.clone())
print(f'checkpointing grad max |diff| {(grads[0] - grads[1]).abs().max().item():.2e}')
assert torch.allclose(grads[0], grads[1], atol=1e-6)
print('全部通过')
