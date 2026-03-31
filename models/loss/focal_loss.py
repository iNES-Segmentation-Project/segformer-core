"""
models/loss/focal_loss.py

Focal Loss for semantic segmentation — Experiment E2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[배경]

  CrossEntropyLoss는 easy example(잘 맞히는 픽셀)과
  hard example(틀리는 픽셀)에 동등한 gradient를 부여한다.
  클래스 불균형이 심한 경우(CamVid의 Bicyclist, Pedestrian 등)
  easy example이 loss를 지배하여 rare class 학습이 방해된다.

  Focal Loss는 (1 - p_t)^gamma 항으로 easy example의 가중치를
  낮추고, hard example에 집중한다.

  FL(p_t) = -(1 - p_t)^gamma * log(p_t)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[CrossEntropyLoss와의 구조 대응]

  CrossEntropyLoss          │ FocalLoss
  ──────────────────────────┼──────────────────────────────
  nn.CrossEntropyLoss 위임  │ 직접 log_softmax + gather
  focusing 없음             │ (1 - p_t)^gamma 곱
  ignore_index 자동 처리    │ valid mask로 수동 처리
  weight (class prior)      │ alpha (class prior, 동일 역할)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[단일 변수 원칙]

  E2 실험에서 변경되는 것: CrossEntropyLoss → FocalLoss
  고정 요소: Encoder(MiT-B0), Decoder(MLP), 학습 설정 전체

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[gamma 선택 근거]

  원논문(Lin et al., 2017) 권장값: gamma=2.0
  gamma=0 이면 CrossEntropyLoss와 수학적으로 동일.
  ablation 없이 단일 실험만 할 경우 gamma=2.0 사용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  - Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss — E2 실험용.

    gamma=0 이면 CrossEntropyLoss와 수학적으로 동일하므로,
    E0(CE)과의 공정한 비교를 위해 gamma=2.0을 기본값으로 사용.

    Args:
        gamma        (float):         focusing parameter. Default: 2.0.
        alpha        (Tensor|None):   클래스별 가중치 (C,). CrossEntropyLoss의
                                      weight와 동일한 역할. None이면 uniform.
        ignore_index (int):           무시할 class index. Default: 255.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        # alpha는 버퍼로 등록 — device 이동 시 자동으로 따라감
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits  : (B, num_classes, H, W)  float32 — raw logits
            targets : (B, H, W)               int64   — class index
        Returns:
            loss : scalar Tensor
        """
        # ── valid mask: ignore_index 픽셀 제외 ────────────────────────────────
        # (B, H, W) → bool mask
        valid_mask = targets != self.ignore_index            # (B, H, W)

        # ── ignore 픽셀을 0으로 치환 (gather를 위한 임시 처리) ────────────────
        # gather 전에 ignore_index 값(255)이 index로 사용되면 out-of-range 에러
        # valid_mask로 나중에 제거하므로 0으로 치환해도 loss에 영향 없음
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                        # (B, H, W)

        # ── log_softmax → p_t 계산 ───────────────────────────────────────────
        # (B, C, H, W)
        log_prob = F.log_softmax(logits, dim=1)

        # 정답 클래스의 log 확률만 추출
        # targets_safe: (B, H, W) → unsqueeze → (B, 1, H, W)
        log_pt = log_prob.gather(
            dim=1,
            index=targets_safe.unsqueeze(1),
        ).squeeze(1)                                         # (B, H, W)

        pt = log_pt.exp()                                    # (B, H, W)

        # ── Focal weight: (1 - p_t)^gamma ────────────────────────────────────
        focal_weight = (1.0 - pt) ** self.gamma              # (B, H, W)

        # ── alpha weighting (optional) ────────────────────────────────────────
        if self.alpha is not None:
            # alpha: (C,) → targets_safe 인덱스로 픽셀별 alpha 추출
            alpha_t = self.alpha[targets_safe]               # (B, H, W)
            focal_weight = focal_weight * alpha_t

        # ── Focal Loss 계산 ───────────────────────────────────────────────────
        # FL = -(1 - p_t)^gamma * log(p_t)
        loss = -focal_weight * log_pt                        # (B, H, W)

        # ── ignore_index 픽셀 마스킹 후 mean ─────────────────────────────────
        # valid 픽셀 수로 나눠야 ignore 비율에 따라 loss 스케일이 달라지지 않음
        loss = loss * valid_mask.float()
        loss = loss.sum() / valid_mask.float().sum().clamp(min=1.0)

        return loss