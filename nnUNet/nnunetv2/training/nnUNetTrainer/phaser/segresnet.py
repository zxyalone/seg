"""MONAI SegResNet in the TriALS benchmark configuration, for the
architecture-transfer experiment of the detectability-guided training
(plan §11, 2026-10-13).

The wrapper only adds what the trainer's CFC loss needs: the outputs of the
decoder stages are recorded with forward hooks and exposed, finest first, as
`_last_dec_features` (the attribute PhaserUNet uses). The network itself is
unchanged: init_filters 32, blocks_down [1, 2, 2, 4], blocks_up [1, 1, 1], no
deep supervision.
"""
from torch import nn


class SegResNetWithFeatures(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3):
        super().__init__()
        from monai.networks.nets import SegResNet
        self.net = SegResNet(spatial_dims=spatial_dims, init_filters=32, in_channels=in_channels,
                             out_channels=out_channels, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1])
        self._last_dec_features = None
        self._spectral_grad_scales = 0      # read by the trainer; unused here
        self._stage_outputs = {}
        for i, layer in enumerate(self.net.up_layers):
            layer.register_forward_hook(self._record(i))

    def _record(self, i):
        def hook(_module, _inputs, output):
            self._stage_outputs[i] = output
        return hook

    def forward(self, x):
        self._stage_outputs = {}
        y = self.net(x)
        n = len(self.net.up_layers)
        self._last_dec_features = [self._stage_outputs[n - 1 - k] for k in range(n)]   # finest first
        return y
