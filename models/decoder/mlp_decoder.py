"""
mlp_decoder.py

SegFormer All-MLP Decoder — Faithful Baseline (E0).

이 구현은 NVlabs/SegFormer 공식 코드(segformer_head.py)와
MMSegmentation 구현을 기준으로, 구조적으로 동일하게 재현한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[공식 구현과의 구조 대응]

  NVlabs 원본                     │ 이 구현
  ────────────────────────────────┼──────────────────────────────
  MLP(Linear)                     │ LinearProjection(Conv2d 1×1)
  bilinear resize to x[0].size()  │ F.interpolate to c1.shape
  torch.cat                       │ torch.cat
  ConvModule(1×1, BN, ReLU)       │ Conv2d(1×1) + BN + ReLU
  Dropout2d                       │ Dropout2d
  nn.Conv2d(1×1) seg head         │ nn.Conv2d(1×1) seg head

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[NVlabs 원본 MLP 클래스 동작 방식]

  class MLP(nn.Module):
      def __init__(self, input_dim, embed_dim):
          self.proj = nn.Linear(input_dim, embed_dim)  # Linear만, BN/ReLU 없음

      def forward(self, x):
          x = x.flatten(2).transpose(1, 2)  # (B,C,H,W) → (B,N,C)
          x = self.proj(x)                   # (B,N,embed_dim)
          return x
  # 이후 reshape: (B,N,C) → (B,C,H,W)

  Linear vs Conv2d(1×1): 수학적으로 동일.
  본 구현은 2D spatial map 흐름 유지를 위해 Conv2d(1×1) 채택.
  (BN/ReLU는 proj 단계에서 절대 포함하지 않는다.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[이전 구현과의 핵심 차이]

  항목               │ 이전 구현 (비충실)          │ 이 구현 (논문 일치)
  ───────────────────┼────────────────────────────┼──────────────────────
  Projection         │ Conv2d(1×1) + BN + ReLU    │ Conv2d(1×1) only
  Fusion conv kernel │ 3×3                        │ 1×1  ← 논문 명시
  BN/ReLU 위치       │ proj 4회 + fusion 1회       │ fusion 1회만
  비선형성 횟수      │ 5회                        │ 1회

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  - SegFormer paper (Xie et al., NeurIPS 2021), Section 3.2
  - NVlabs/SegFormer: mmseg/models/decode_heads/segformer_head.py
  - MMSegmentation: mmseg/models/decode_heads/segformer_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from .base_decoder import BaseDecoder


class LinearProjection(nn.Module):
    """
    Per-stage channel projection: C_i → embed_dim.

    NVlabs 원본의 MLP 클래스에 대응.
    nn.Linear(C_i, embed_dim)와 수학적으로 동일한 Conv2d(1×1)을 사용.

    BN, ReLU, Dropout 없음.
    projection 단계는 선형 변환만 수행한다.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        # bias=True: nn.Linear의 기본값과 동일하게 맞춤
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, C_i, H_i, W_i)
        Returns:
            x : (B, embed_dim, H_i, W_i)
        """
        return self.proj(x)  # linear projection only, no BN/ReLU


class MLPDecoder(BaseDecoder):
    """
    SegFormer All-MLP Decoder — NVlabs 원본 충실 재현.

    Pipeline:
      Step 1 │ LinearProjection (per stage) : C_i → embed_dim  [BN/ReLU 없음]
      Step 2 │ Bilinear upsample            : H_i,W_i → H/4,W/4
      Step 3 │ Concatenate                  : 4 × (B,embed_dim,H/4,W/4)
      Step 4 │ Fusion Conv2d(1×1)+BN+ReLU   : 4*embed_dim → embed_dim
      Step 5 │ Dropout2d
      Step 6 │ Seg head Conv2d(1×1)         : embed_dim → num_classes

    Args:
        in_channels (List[int]): 각 stage 출력 채널 [c1,c2,c3,c4].
                                 MiT-B0: [32, 64, 160, 256].
        embed_dim   (int):       통합 채널 수. B0 기준 256.
        num_classes (int):       분류 클래스 수.
        dropout     (float):     seg head 직전 Dropout2d 비율. Default: 0.1
    """

    def __init__(
        self,
        in_channels: List[int],
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__(num_classes=num_classes)

        assert len(in_channels) == 4, (
            f"in_channels must have 4 entries for [c1,c2,c3,c4], got {len(in_channels)}"
        )

        self.embed_dim = embed_dim

        # ── Step 1: Per-stage linear projection ───────────────────────────────
        # BN/ReLU 없음 — 논문 원본 MLP 클래스와 동일한 선형 변환만 수행
        #
        #   c1 : (B,  32, H/4,  W/4 ) → (B, embed_dim, H/4,  W/4 )
        #   c2 : (B,  64, H/8,  W/8 ) → (B, embed_dim, H/8,  W/8 )
        #   c3 : (B, 160, H/16, W/16) → (B, embed_dim, H/16, W/16)
        #   c4 : (B, 256, H/32, W/32) → (B, embed_dim, H/32, W/32)
        self.proj_layers = nn.ModuleList([
            LinearProjection(in_ch, embed_dim)
            for in_ch in in_channels
        ])

        # ── Step 4: Fusion conv ────────────────────────────────────────────────
        # NVlabs 원본: ConvModule(4*embed_dim, embed_dim, kernel_size=1, norm_cfg=BN)
        #
        # kernel=1×1 (논문 원본, 3×3 아님)
        # BN + ReLU는 오직 이 fusion 단계에서만 사용
        #
        #   (B, 4*embed_dim, H/4, W/4) → (B, embed_dim, H/4, W/4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                4 * embed_dim,
                embed_dim,
                kernel_size=1,   # 1×1: 논문 원본과 일치
                bias=False,      # BN이 뒤에 오므로 bias 불필요
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # ── Step 5 & 6: Dropout + Segmentation head ───────────────────────────
        # NVlabs 원본: self.dropout(x) → self.linear_pred(x)
        self.dropout = nn.Dropout2d(p=dropout)
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Args:
            features : [c1, c2, c3, c4]
                c1 : (B,  32, H/4,  W/4 )
                c2 : (B,  64, H/8,  W/8 )
                c3 : (B, 160, H/16, W/16)
                c4 : (B, 256, H/32, W/32)

        Returns:
            logits : (B, num_classes, H/4, W/4)
            ※ 최종 H 복원 upsample은 segformer.py에서 수행
        """
        self._check_features(features)

        # c1의 공간 해상도 = target (H/4, W/4)
        target_H, target_W = features[0].shape[2], features[0].shape[3]

        # ── Step 1 & 2: Project + Upsample (per stage) ────────────────────────
        projected = []
        for feat, proj in zip(features, self.proj_layers):

            # Step 1: Linear projection only (BN/ReLU 없음)
            # (B, C_i, H_i, W_i) → (B, embed_dim, H_i, W_i)
            x = proj(feat)

            # Step 2: Bilinear upsample to c1 resolution
            # c1은 이미 target이므로 skip; c2,c3,c4만 실제 upsample 발생
            # (B, embed_dim, H_i, W_i) → (B, embed_dim, H/4, W/4)
            if x.shape[2] != target_H or x.shape[3] != target_W:
                x = F.interpolate(
                    x,
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )

            projected.append(x)  # (B, embed_dim, H/4, W/4)

        # ── Step 3: Concatenate ────────────────────────────────────────────────
        # 4 × (B, embed_dim, H/4, W/4) → (B, 4*embed_dim, H/4, W/4)
        x = torch.cat(projected, dim=1)

        # ── Step 4: Fusion Conv(1×1) + BN + ReLU ──────────────────────────────
        # (B, 4*embed_dim, H/4, W/4) → (B, embed_dim, H/4, W/4)
        x = self.fusion_conv(x)

        # ── Step 5: Dropout2d ─────────────────────────────────────────────────
        # (B, embed_dim, H/4, W/4)
        x = self.dropout(x)

        # ── Step 6: Segmentation head ──────────────────────────────────────────
        # (B, embed_dim, H/4, W/4) → (B, num_classes, H/4, W/4)
        logits = self.seg_head(x)

        return logits  # (B, num_classes, H/4, W/4)