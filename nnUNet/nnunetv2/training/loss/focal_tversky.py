import torch
from torch import nn


class MultiClassFocalTverskyLoss(nn.Module):
    """
    Multi-class focal Tversky loss on logits.
    target must be shape [B, 1, ...] (class indices) like nnUNet default.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1e-5,
        include_background: bool = False,
        ignore_label: int | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background
        self.ignore_label = ignore_label

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] != 1:
            raise ValueError("target must have shape [B, 1, ...] for MultiClassFocalTverskyLoss")

        probs = torch.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        target_idx = target[:, 0].long()
        valid_mask = torch.ones_like(target_idx, dtype=torch.bool)
        if self.ignore_label is not None:
            valid_mask = target_idx != self.ignore_label
            target_idx = torch.where(valid_mask, target_idx, torch.zeros_like(target_idx))

        target_1h = torch.nn.functional.one_hot(target_idx, num_classes=num_classes)
        # probs shape: [B, C, ...]. Reduce over batch + spatial dims, keep class dim.
        dims = [0] + list(range(2, probs.ndim))
        target_1h = target_1h.permute(0, -1, *range(1, target_1h.ndim - 1)).to(probs.dtype)

        valid_mask = valid_mask.unsqueeze(1).to(probs.dtype)
        probs = probs * valid_mask
        target_1h = target_1h * valid_mask

        tp = (probs * target_1h).sum(dim=dims)
        fp = (probs * (1.0 - target_1h)).sum(dim=dims)
        fn = ((1.0 - probs) * target_1h).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)

        if not self.include_background and num_classes > 1:
            focal_tversky = focal_tversky[1:]

        return focal_tversky.mean()

