"""
models/loss/dice_loss.py

Soft Dice Loss for semantic segmentation — Experiment E3 (CE + Dice).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[배경]

  CrossEntropyLoss는 픽셀 단위 분류 손실로, 클래스 불균형 시
  rare class에 대한 gradient 기여가 적다.

  Dice Loss는 예측과 GT의 overlap 비율을 직접 최적화한다.
  클래스별로 독립적으로 계산하므로 픽셀 수가 적은 클래스도
  loss에 동등하게 기여한다.

  Dice = 2 * |P ∩ G| / (|P| + |G|)
  DiceLoss = 1 - Dice

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Soft Dice vs Hard Dice]

  Hard Dice: argmax 후 계산 → 미분 불가
  Soft Dice: softmax 확률로 계산 → 미분 가능, 학습에 사용

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[CrossEntropyLoss와의 구조 대응]

  CrossEntropyLoss          │ DiceLoss
  ──────────────────────────┼──────────────────────────────
  픽셀 단위 분류 손실        │ 클래스 단위 overlap 손실
  ignore_index 자동 처리    │ valid mask로 수동 처리
  클래스 불균형에 민감       │ 클래스 불균형에 강건

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[smooth 값 선택 근거]

  smooth=1.0: Laplace smoothing. zero-division 방지.
  GT와 예측이 모두 비어있는 클래스(희귀 클래스)를 1.0으로
  처리하여 loss가 0이 되는 것을 막는다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[단일 변수 원칙]

  E3 실험: CrossEntropyLoss + DiceLoss 조합 (CombinedLoss 경유)
  DiceLoss 단독 사용 시 학습 불안정 가능성이 있어 CE와 병용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  - V-Net (Milletari et al., 2016)
  - Generalised Dice Loss (Sudre et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiceLoss(nn.Module):
    """
    Soft Dice Loss — E3 실험에서 CrossEntropyLoss와 조합하여 사용.

    클래스별 Dice를 독립적으로 계산한 뒤 평균.
    ignore_index 픽셀은 valid mask로 제거 후 계산.

    Args:
        num_classes  (int):   클래스 수.
        ignore_index (int):   무시할 class index. Default: 255.
        smooth       (float): zero-division 방지용 smoothing 상수. Default: 1.0.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits  : (B, num_classes, H, W)  float32 — raw logits
            targets : (B, H, W)               int64   — class index
        Returns:
            loss : scalar Tensor
        """
        # ── valid mask: ignore_index 픽셀 제외 ────────────────────────────────
        valid_mask = (targets != self.ignore_index)          # (B, H, W) bool

        # ── ignore 픽셀을 0으로 치환 (one_hot 변환을 위한 임시 처리) ──────────
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                        # (B, H, W)

        # ── Softmax → 예측 확률 ───────────────────────────────────────────────
        # (B, C, H, W)
        prob = F.softmax(logits, dim=1)

        # ── One-hot encoding ──────────────────────────────────────────────────
        # (B, H, W) → (B, H, W, C) → (B, C, H, W)
        one_hot = F.one_hot(targets_safe, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()       # (B, C, H, W)

        # ── valid mask 적용: ignore 픽셀을 예측/GT 모두에서 제거 ──────────────
        # (B, H, W) → (B, 1, H, W) 로 브로드캐스트
        mask = valid_mask.unsqueeze(1).float()               # (B, 1, H, W)
        prob    = prob    * mask                             # (B, C, H, W)
        one_hot = one_hot * mask                             # (B, C, H, W)

        # ── 클래스별 Soft Dice 계산 ───────────────────────────────────────────
        # dim=(0,2,3): batch + spatial 방향으로 합산 → (C,)
        intersection = (prob * one_hot).sum(dim=(0, 2, 3))  # (C,)
        cardinality  = (prob + one_hot).sum(dim=(0, 2, 3))  # (C,)

        # Dice per class: (C,)
        dice_per_class = (2.0 * intersection + self.smooth) / \
                         (cardinality + self.smooth)

        # ── 클래스 평균 → DiceLoss ────────────────────────────────────────────
        return 1.0 - dice_per_class.mean()