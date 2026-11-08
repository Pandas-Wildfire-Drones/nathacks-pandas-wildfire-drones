"""PRIMARY verifier: a deliberately small CNN (~45k parameters, ~0.2 MB ONNX, ~1 ms per patch on a CPU core).

Small on purpose: the labelled radiometric data is small (FLAME 3), the decision is local ("is this blob a
fire?"), and it must run on the Jetson next to everything else at 9 Hz. Global MAX pooling is concatenated
with average pooling because a tiny hot spot can be only a few pixels of a 32 x 32 patch.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["TinyFireNet"]


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyFireNet(nn.Module):
    def __init__(self, in_channels: int = 2, width: int = 16, dropout: float = 0.2):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(in_channels, w), _block(w, w), nn.MaxPool2d(2),       # 32 -> 16
            _block(w, 2 * w), _block(2 * w, 2 * w), nn.MaxPool2d(2),     # 16 -> 8
            _block(2 * w, 4 * w),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, C, P, P) -> (N, 1) logits
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)
