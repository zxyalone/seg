import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_200epochs(nnUNetTrainer):
    """Stock nnU-Net trainer with the 200-epoch budget used by every PhASER
    ablation group, for same-budget external baselines (e.g. ResEnc, E9)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200
