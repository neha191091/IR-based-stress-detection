"""LeJEPA backbone stub for future integration."""

import torch

from ir_stress.models.backbone import Backbone


class LeJEPABackbone(Backbone):
    """
    Placeholder for a LeJEPA-based video encoder.

  Subclass Backbone and implement encode() to integrate a LeJEPA encoder
  that outputs spatiotemporal features compatible with STRppgHead.
    """

    def __init__(self, in_ch: int = 1):
        super().__init__()
        self._in_ch = in_ch
        raise NotImplementedError(
            "LeJEPABackbone is not yet implemented. Use model: physnet in config."
        )

    @property
    def out_channels(self) -> int:
        return 64

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("LeJEPABackbone.encode() is not yet implemented.")
