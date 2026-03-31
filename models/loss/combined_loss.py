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
[log_dict 반환 이유]

  total loss만 반환하면 CE/Dice/Boundary 각각의 기여를
  학습 중에 모니터링할 수 없다.
  log_dict를 함께 반환하여 WandB/CSV 로거에 바로 넘길 수 있게 한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  각 개별 loss 파일의 Reference 참조.
"""

import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

from .cross_entropy import CrossEntropyLoss
from .dice_loss     import DiceLoss
from .boundary_loss import BoundaryLoss


class CombinedLoss(nn.Module):
    """
    CE + Dice 또는 CE + Boundary 조합 Loss — E3, E4 실험용.

    forward()는 (total_loss, log_dict) 튜플을 반환한다.
    log_dict는 {"loss/ce": float, "loss/dice": float, "loss/total": float}
    형태로 WandB/CSV 로거에 바로 넘길 수 있다.

    Args:
        mode         (str):   "ce+dice" | "ce+boundary". 실험 구성 선택.
        num_classes  (int):   클래스 수. DiceLoss에 필요.
        ignore_index (int):   무시할 class index. Default: 255.
        ce_weight    (float): CE loss 가중치. Default: 1.0.
        aux_weight   (float): 보조 loss(Dice|Boundary) 가중치. Default: 1.0.
    """

    VALID_MODES = ("ce+dice", "ce+boundary")

    def __init__(
        self,
        mode: str,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        aux_weight: float = 1.0,
    ):
        super().__init__()

        assert mode in self.VALID_MODES, (
            f"mode must be one of {self.VALID_MODES}, got '{mode}'"
        )

        self.mode       = mode
        self.ce_weight  = ce_weight
        self.aux_weight = aux_weight

        # ── CE loss (공통) ────────────────────────────────────────────────────
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)

        # ── 보조 loss (mode에 따라 선택) ──────────────────────────────────────
        if mode == "ce+dice":
            self.aux = DiceLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            self._aux_key = "loss/dice"

        elif mode == "ce+boundary":
            self.aux = BoundaryLoss(ignore_index=ignore_index)
            self._aux_key = "loss/boundary"

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Args:
            logits  : (B, num_classes, H, W)  float32 — raw logits
            targets : (B, H, W)               int64   — class index
        Returns:
            total_loss : scalar Tensor  — backward() 대상
            log_dict   : Dict[str, float]
                           "loss/ce"              : CE loss 값
                           "loss/dice"|"loss/boundary" : 보조 loss 값
                           "loss/total"           : weighted 합계
        """
        # ── 개별 loss 계산 ────────────────────────────────────────────────────
        ce_loss  = self.ce(logits, targets)
        aux_loss = self.aux(logits, targets)

        # ── 가중합 ────────────────────────────────────────────────────────────
        total = self.ce_weight * ce_loss + self.aux_weight * aux_loss

        # ── 로깅용 dict ───────────────────────────────────────────────────────
        log_dict: Dict[str, float] = {
            "loss/ce":       ce_loss.item(),
            self._aux_key:   aux_loss.item(),
            "loss/total":    total.item(),
        }

        return total, log_dict