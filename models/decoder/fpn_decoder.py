"""
fpn_decoder.py

FPN-style Decoder for SegFormer — Custom Decoder (E1, E5).

MLP Decoder(E0)와의 단일 변수 비교를 위해 Encoder는 완전히 동일하게 유지하고,
Decoder만 FPN 구조로 교체한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[MLP Decoder(E0) vs FPN Decoder(E1) 구조 대응]

  MLP Decoder (E0)                │ FPN Decoder (E1)
  ────────────────────────────────┼──────────────────────────────
  LinearProjection × 4 (1×1)     │ LateralConv × 4 (1×1)
  독립 upsample → concat          │ top-down pathway (add) → concat
  stage 간 상호작용 없음           │ F4→F3→F2→F1 순서로 의미 전파
  fusion Conv(1×1) + BN + ReLU    │ OutputConv(3×3) + BN + ReLU × 4
  Dropout2d                       │ fusion Conv(1×1) + BN + ReLU
  seg head Conv(1×1)              │ Dropout2d → seg head Conv(1×1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[FPN 4단계 파이프라인]

  Step 1 │ LateralConv (1×1, per stage)    : C_i → fpn_dim   [BN/ReLU 없음]
  Step 2 │ Top-down pathway                : laterals[i] += upsample(laterals[i+1])
           ※ F4 → F3 → F2 → F1 순서로 역방향 순회
  Step 3 │ OutputConv (3×3, per level)     : fpn_dim → fpn_dim, BN + ReLU
  Step 4 │ Upsample all to c1 + Concatenate: 4 × (B, fpn_dim, H/4, W/4)
  Step 5 │ Fusion Conv(1×1) + BN + ReLU   : 4*fpn_dim → fpn_dim
  Step 6 │ Dropout2d
  Step 7 │ Seg head Conv(1×1)             : fpn_dim → num_classes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LateralConv 설계 근거]

  MLP Decoder의 LinearProjection과 동일하게 BN/ReLU 없는 1×1 Conv를 사용.
  - lateral connection의 역할은 채널 통일(C_i → fpn_dim)에 국한
  - 비선형성은 top-down add 이후 Step 3의 OutputConv에서 한 번만 적용
  - MLP의 LinearProjection과 대칭적 설계 → 변수 통제(단일 변수 원칙) 준수

  Linear vs Conv2d(1×1): 수학적으로 동일.
  공간 맵 흐름 유지를 위해 Conv2d(1×1) 채택 (MLPDecoder와 동일한 이유).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OutputConv에 3×3을 사용하는 이유]

  top-down add 이후 두 feature를 단순 합산하면 aliasing artifact가 발생할 수 있음.
  3×3 conv는 주변 픽셀을 참조하여 이를 정제하고, 공간 context를 보완한다.
  MLP Decoder의 fusion conv(1×1)와 역할이 다름 → 3×3 의도적 선택.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[E0 vs E1 단일 변수 통제 요약]

  고정 요소 : Encoder(MiT-B0), Loss(CE), 학습률, 배치크기, 에폭, 시드
  변경 요소 : Decoder (MLPDecoder → FPNDecoder)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  - Feature Pyramid Networks for Object Detection (Lin et al., CVPR 2017)
  - SegFormer paper (Xie et al., NeurIPS 2021), Section 3.2
  - MMDetection: mmdet/models/necks/fpn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from .base_decoder import BaseDecoder


class LateralConv(nn.Module):
    """
    Per-stage lateral connection: C_i → fpn_dim.

    MLPDecoder의 LinearProjection에 대응.
    채널 수를 통일하여 top-down pathway의 element-wise add를 가능하게 한다.

    BN, ReLU 없음.
    lateral 단계는 채널 변환만 수행한다 (LinearProjection과 동일한 설계 원칙).
    """

    def __init__(self, in_channels: int, fpn_dim: int):
        super().__init__()
        # bias=True: LinearProjection(nn.Linear 기본값)과 대칭
        self.proj = nn.Conv2d(in_channels, fpn_dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, C_i, H_i, W_i)
        Returns:
            x : (B, fpn_dim, H_i, W_i)
        """
        return self.proj(x)  # lateral projection only, no BN/ReLU


class OutputConv(nn.Module):
    """
    Per-level output refinement: fpn_dim → fpn_dim.

    top-down add 이후 발생할 수 있는 aliasing artifact를 정제하고
    공간적 context를 보완한다.

    3×3 Conv + BN + ReLU.
    비선형성은 오직 이 단계와 fusion conv에서만 사용된다.
    """

    def __init__(self, fpn_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                fpn_dim,
                fpn_dim,
                kernel_size=3,
                padding=1,
                bias=False,  # BN이 뒤에 오므로 bias 불필요
            ),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, fpn_dim, H_i, W_i)
        Returns:
            x : (B, fpn_dim, H_i, W_i)
        """
        return self.conv(x)


class FPNDecoder(BaseDecoder):
    """
    FPN-style Decoder — E1/E5 실험용 커스텀 Decoder.

    MLP Decoder(E0)와 Encoder를 완전히 공유하고 Decoder만 교체한다.
    top-down pathway와 lateral connection으로 stage 간 feature interaction을 강화한다.

    Pipeline:
      Step 1 │ LateralConv (1×1, per stage)  : C_i → fpn_dim  [BN/ReLU 없음]
      Step 2 │ Top-down pathway               : F4→F3→F2→F1 순서, element-wise add
      Step 3 │ OutputConv (3×3, per level)    : fpn_dim → fpn_dim, BN + ReLU
      Step 4 │ Upsample all → H/4 + Cat       : 4 × (B, fpn_dim, H/4, W/4)
      Step 5 │ Fusion Conv(1×1) + BN + ReLU  : 4*fpn_dim → fpn_dim
      Step 6 │ Dropout2d
      Step 7 │ Seg head Conv(1×1)             : fpn_dim → num_classes

    Args:
        in_channels (List[int]): 각 stage 출력 채널 [c1,c2,c3,c4].
                                 MiT-B0: [32, 64, 160, 256].
        fpn_dim     (int):       FPN 내부 통합 채널 수. Default: 256.
        num_classes (int):       분류 클래스 수.
        dropout     (float):     seg head 직전 Dropout2d 비율. Default: 0.1.
    """

    def __init__(
        self,
        in_channels: List[int],
        fpn_dim: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
    ):
        super().__init__(num_classes=num_classes)

        assert len(in_channels) == 4, (
            f"in_channels must have 4 entries for [c1,c2,c3,c4], got {len(in_channels)}"
        )

        self.fpn_dim = fpn_dim

        # ── Step 1: Lateral conv (per stage) ─────────────────────────────────
        # BN/ReLU 없음 — LinearProjection(E0)과 대칭적 설계
        #
        #   c1 : (B,  32, H/4,  W/4 ) → (B, fpn_dim, H/4,  W/4 )
        #   c2 : (B,  64, H/8,  W/8 ) → (B, fpn_dim, H/8,  W/8 )
        #   c3 : (B, 160, H/16, W/16) → (B, fpn_dim, H/16, W/16)
        #   c4 : (B, 256, H/32, W/32) → (B, fpn_dim, H/32, W/32)
        self.lateral_convs = nn.ModuleList([
            LateralConv(in_ch, fpn_dim)
            for in_ch in in_channels
        ])

        # ── Step 3: Output conv (per level) ──────────────────────────────────
        # top-down add 이후 각 레벨을 독립적으로 정제
        # 인덱스는 lateral_convs와 1:1 대응
        #
        #   P1 : (B, fpn_dim, H/4,  W/4 ) → (B, fpn_dim, H/4,  W/4 )
        #   P2 : (B, fpn_dim, H/8,  W/8 ) → (B, fpn_dim, H/8,  W/8 )
        #   P3 : (B, fpn_dim, H/16, W/16) → (B, fpn_dim, H/16, W/16)
        #   P4 : (B, fpn_dim, H/32, W/32) → (B, fpn_dim, H/32, W/32)
        self.output_convs = nn.ModuleList([
            OutputConv(fpn_dim)
            for _ in in_channels
        ])

        # ── Step 5: Fusion conv ───────────────────────────────────────────────
        # MLPDecoder의 fusion_conv와 동일한 구조 (1×1, BN, ReLU)
        # MLPDecoder: 4*embed_dim → embed_dim
        # FPNDecoder: 4*fpn_dim   → fpn_dim   (대칭)
        #
        #   (B, 4*fpn_dim, H/4, W/4) → (B, fpn_dim, H/4, W/4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                4 * fpn_dim,
                fpn_dim,
                kernel_size=1,   # 1×1: MLPDecoder fusion_conv와 동일
                bias=False,      # BN이 뒤에 오므로 bias 불필요
            ),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )

        # ── Step 6 & 7: Dropout + Segmentation head ──────────────────────────
        # MLPDecoder와 완전히 동일한 구조
        self.dropout = nn.Dropout2d(p=dropout)
        self.seg_head = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

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

        # ── Step 1: Lateral projection (per stage) ────────────────────────────
        # BN/ReLU 없는 선형 변환으로 각 stage 채널을 fpn_dim으로 통일
        #
        #   c1 → L1 : (B, fpn_dim, H/4,  W/4 )
        #   c2 → L2 : (B, fpn_dim, H/8,  W/8 )
        #   c3 → L3 : (B, fpn_dim, H/16, W/16)
        #   c4 → L4 : (B, fpn_dim, H/32, W/32)
        laterals: List[Tensor] = [
            proj(feat)
            for feat, proj in zip(features, self.lateral_convs)
        ]

        # ── Step 2: Top-down pathway ──────────────────────────────────────────
        # 깊은 stage(L4)의 의미론적 정보를 얕은 stage(L1) 방향으로 전파.
        # i = 3, 2, 1 순서로 역방향 순회 → L4가 L3을 보강, L3이 L2를 보강, ...
        #
        #   iteration i=3: upsample(L4, size=L3) → L3 += upsampled_L4
        #   iteration i=2: upsample(L3, size=L2) → L2 += upsampled_L3
        #   iteration i=1: upsample(L2, size=L1) → L1 += upsampled_L2
        #
        # element-wise add가 가능한 이유: Step 1에서 채널을 fpn_dim으로 통일했기 때문
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],  # (H_{i-1}, W_{i-1})
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled  # in-place add

        # ── Step 3: Output conv (per level) ──────────────────────────────────
        # top-down add로 발생하는 aliasing artifact를 3×3 conv로 정제
        # 비선형성(BN+ReLU)이 처음으로 적용되는 단계
        #
        #   L1 → P1 : (B, fpn_dim, H/4,  W/4 )
        #   L2 → P2 : (B, fpn_dim, H/8,  W/8 )
        #   L3 → P3 : (B, fpn_dim, H/16, W/16)
        #   L4 → P4 : (B, fpn_dim, H/32, W/32)
        fpn_outs: List[Tensor] = [
            out_conv(lat)
            for lat, out_conv in zip(laterals, self.output_convs)
        ]

        # ── Step 4: Upsample all to c1 resolution + Concatenate ───────────────
        # 모든 레벨을 P1의 해상도(H/4, W/4)로 upsample 후 채널 방향으로 concat.
        # P1은 이미 target 해상도이므로 skip; P2, P3, P4만 실제 upsample 발생.
        #
        #   P1 :          (B, fpn_dim, H/4,  W/4 )  → (B, fpn_dim, H/4, W/4) [skip]
        #   P2 : ×2 up    (B, fpn_dim, H/8,  W/8 )  → (B, fpn_dim, H/4, W/4)
        #   P3 : ×4 up    (B, fpn_dim, H/16, W/16)  → (B, fpn_dim, H/4, W/4)
        #   P4 : ×8 up    (B, fpn_dim, H/32, W/32)  → (B, fpn_dim, H/4, W/4)
        #   cat → (B, 4*fpn_dim, H/4, W/4)
        target_H, target_W = fpn_outs[0].shape[2], fpn_outs[0].shape[3]

        aligned: List[Tensor] = []
        for p in fpn_outs:
            if p.shape[2] != target_H or p.shape[3] != target_W:
                p = F.interpolate(
                    p,
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(p)

        # (B, 4*fpn_dim, H/4, W/4)
        x = torch.cat(aligned, dim=1)

        # ── Step 5: Fusion Conv(1×1) + BN + ReLU ─────────────────────────────
        # MLPDecoder의 fusion_conv와 동일한 구조로 채널 압축
        # (B, 4*fpn_dim, H/4, W/4) → (B, fpn_dim, H/4, W/4)
        x = self.fusion_conv(x)

        # ── Step 6: Dropout2d ─────────────────────────────────────────────────
        # (B, fpn_dim, H/4, W/4)
        x = self.dropout(x)

        # ── Step 7: Segmentation head ─────────────────────────────────────────
        # (B, fpn_dim, H/4, W/4) → (B, num_classes, H/4, W/4)
        logits = self.seg_head(x)

        return logits  # (B, num_classes, H/4, W/4)