"""
models/loss/cross_entropy.py

CrossEntropy Loss wrapper for semantic segmentation.

nn.CrossEntropyLoss는:
  - input : (B, C, H, W)  — logits (raw, before softmax)
  - target: (B, H, W)     — class index (int64)
  - ignore_index: 해당 pixel은 loss 계산에서 제외 (default 255 = void)

CamVid의 void/unlabeled pixel은 255로 표기되므로
ignore_index=255가 기본값으로 적합하다.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class CrossEntropyLoss(nn.Module):
    """
    Segmentation용 CrossEntropy Loss.

    Args:
        ignore_index (int):          무시할 class index. Default: 255 (CamVid void).
        weight       (Tensor|None):  클래스별 가중치 (C,). 클래스 불균형 보정 시 사용.
                                     None이면 uniform weight.
        label_smoothing (float):     Label smoothing 비율. Default: 0.0 (baseline).
    """

    def __init__(
        self,
        ignore_index: int = 255,
        weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits  : (B, num_classes, H, W)  float32 — raw logits
            targets : (B, H, W)               int64   — class index

        Returns:
            loss : scalar Tensor
        """
        # logits : (B, C, H, W)
        # targets: (B, H, W)  — nn.CrossEntropyLoss가 직접 처리
        return self.criterion(logits, targets)