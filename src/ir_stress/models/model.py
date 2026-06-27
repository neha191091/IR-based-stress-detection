"""RppgModel composes a backbone with an ST-rPPG head."""

import torch
import torch.nn as nn

from ir_stress.models.backbone import Backbone
from ir_stress.models.lejepa import LeJEPABackbone
from ir_stress.models.physnet import PhysNet
from ir_stress.models.physnet_lite import PhysNetLite
from ir_stress.models.strppg_head import STRppgHead

_BACKBONES: dict[str, type[Backbone]] = {
    "physnet": PhysNet,
    "physnet_lite": PhysNetLite,
    "lejepa": LeJEPABackbone,
}

_GRAD_CHECKPOINT_BACKBONES = frozenset({"physnet", "physnet_lite"})


class RppgModel(nn.Module):
    """Full rPPG model: backbone encoder + ST-rPPG head."""

    def __init__(self, backbone: Backbone, head: STRppgHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ST-rPPG block of shape (B, N, T)."""
        features = self.backbone.encode(self.backbone.normalize(x))
        return self.head(features)

    def predict_rppg(self, x: torch.Tensor) -> torch.Tensor:
        """Return the spatial-mean rPPG channel of shape (B, T)."""
        return self.forward(x)[:, -1, :]


def build_model(
    name: str, *, spatial_dim: int, in_ch: int, grad_checkpoint: bool = False
) -> RppgModel:
    """Construct an RppgModel from a backbone name and hyperparameters."""
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(_BACKBONES)}")
    if name in _GRAD_CHECKPOINT_BACKBONES:
        backbone = _BACKBONES[name](in_ch=in_ch, grad_checkpoint=grad_checkpoint)
    else:
        backbone = _BACKBONES[name](in_ch=in_ch)
    head = STRppgHead(in_channels=backbone.out_channels, spatial_dim=spatial_dim)
    return RppgModel(backbone, head)
