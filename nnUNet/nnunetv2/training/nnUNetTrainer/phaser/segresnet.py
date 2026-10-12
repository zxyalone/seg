"""MONAI SegResNet in the TriALS benchmark configuration, for the
architecture-transfer experiment of the detectability-guided training
(plan §11, 2026-10-13).

The network is MONAI's SegResNet unchanged (init_filters 32, blocks_down
[1, 2, 2, 4], blocks_up [1, 1, 1], no deep supervision). The wrapper re-states
MONAI's encode/decode loop so that
  * the decoder stage outputs are exposed, finest first, as
    `_last_dec_features` (the attribute the trainer's CFC loss reads), and
  * during training each encoder/decoder block runs under gradient
    checkpointing: with batch 2 and 128^3 patches the network does not fit in
    12 GB otherwise (TriALS trained it on 24 GB). Checkpointing recomputes
    activations in the backward pass and does not change the arithmetic.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class SegResNetWithFeatures(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3, checkpointing: bool = True):
        super().__init__()
        from monai.networks.nets import SegResNet
        self.net = SegResNet(spatial_dims=spatial_dims, init_filters=32, in_channels=in_channels,
                             out_channels=out_channels, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1])
        self.checkpointing = checkpointing
        self._last_dec_features = None
        self._spectral_grad_scales = 0      # read by the trainer; unused here

    def _run(self, block, x):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        net = self.net
        x = net.convInit(x)
        if net.dropout_prob is not None:
            x = net.dropout(x)
        down_x = []
        for down in net.down_layers:
            x = self._run(down, x)
            down_x.append(x)
        down_x.reverse()
        feats = []
        for i, (up, upl) in enumerate(zip(net.up_samples, net.up_layers)):
            x = up(x) + down_x[i + 1]
            x = self._run(upl, x)
            feats.append(x)
        self._last_dec_features = feats[::-1]          # finest first
        if net.use_conv_final:
            x = net.conv_final(x)
        return x
