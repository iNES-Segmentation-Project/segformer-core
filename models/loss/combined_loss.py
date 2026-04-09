"""
models/loss/combined_loss.py

Loss 조합 관리 모듈 — Experiments E3, E4, E5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[실험별 loss 구성]

  실험 │ Loss 구성                    │ 비고
  ─────┼──────────────────────────────┼──────────────────
  E0   │ CE                           │ baseline
  E1   │ CE                           │ decoder만 FPN으로 변경
  E2   │ Focal                        │ FocalLoss 단독
  E3   │ CE + Dice                    │ CombinedLoss(ce+dice)
  E4   │ CE + Boundary                │ CombinedLoss(ce+boundary)
  E5   │ FPN + best loss              │ 실험 후 결정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[가중치 설계 원칙]

  CE 계열 loss를 anchor(weight=1.0)로 두고
  보조 loss(Dice, Boundary)의 weight를 조정한다.
  두 loss의 스케일 차이가 크므로 weight로 보정이 필요하다.

  CE ≈ 0.5~2.0,  Dice ≈ 0.3~0.8,  Boundary ≈ 0.1~0.5
  기본값: ce_weight=1.0, aux_weight=1.0 (스케일 비교 후 조정 권장)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  각 개별 loss 파일의 Reference 참조.
"""

import torch.nn as nn
from torch import Tensor

from .cross_entropy import CrossEntropyLoss
from .dice_loss     import DiceLoss
from .boundary_loss import BoundaryLoss


class CombinedLoss(nn.Module):
    """
    CE + Dice, CE + Boundary, 또는 CE + Dice + Boundary 조합 Loss.

    forward()는 scalar Tensor를 반환한다.
    train.py의 기존 학습 루프와 완전히 호환.

    Args:
        mode          (str):   "ce+dice" | "ce+boundary" | "ce+dice+boundary"
        num_classes   (int):   클래스 수. DiceLoss에 필요.
        ignore_index  (int):   무시할 class index. Default: 255.
        ce_weight     (float): CE loss 가중치. Default: 1.0.
        aux_weight    (float): Dice loss 가중치 (ce+dice+boundary 포함). Default: 1.0.
        boundary_weight (float): Boundary loss 가중치 (ce+dice+boundary 전용). Default: 0.1.
    """

    VALID_MODES = ("ce+dice", "ce+boundary", "ce+dice+boundary")

    def __init__(
        self,
        mode: str,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        aux_weight: float = 1.0,
        boundary_weight: float = 0.1,
    ):
        super().__init__()

        assert mode in self.VALID_MODES, (
            f"mode must be one of {self.VALID_MODES}, got '{mode}'"
        )

        self.mode            = mode
        self.ce_weight       = ce_weight
        self.aux_weight      = aux_weight
        self.boundary_weight = boundary_weight

        # ── CE loss (공통) ────────────────────────────────────────────────────
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)

        # ── 보조 loss (mode에 따라 선택) ──────────────────────────────────────
        if mode in ("ce+dice", "ce+dice+boundary"):
            self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        if mode in ("ce+boundary", "ce+dice+boundary"):
            self.boundary = BoundaryLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits  : (B, num_classes, H, W)  float32 — raw logits
            targets : (B, H, W)               int64   — class index
        Returns:
            loss : scalar Tensor — backward() 대상
        """
        ce_loss = self.ce(logits, targets)

        if self.mode == "ce+dice":
            return self.ce_weight * ce_loss + self.aux_weight * self.dice(logits, targets)

        elif self.mode == "ce+boundary":
            return self.ce_weight * ce_loss + self.aux_weight * self.boundary(logits, targets)

        else:  # ce+dice+boundary
            # Total = 1.0*CE + 1.0*Dice + 0.1*Boundary (E5 설계 원칙)
            return (self.ce_weight       * ce_loss
                    + self.aux_weight      * self.dice(logits, targets)
                    + self.boundary_weight * self.boundary(logits, targets))