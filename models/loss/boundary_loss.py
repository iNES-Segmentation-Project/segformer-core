"""
models/loss/boundary_loss.py

Boundary-aware Loss for semantic segmentation — Experiment E4 (CE + Boundary).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[배경]

  CrossEntropyLoss는 경계(boundary) 픽셀과 내부(interior) 픽셀을
  동등하게 취급한다. Segmentation 품질은 경계 정확도에 크게 좌우되므로,
  경계 픽셀에 높은 가중치를 주면 boundary IoU 향상을 기대할 수 있다.

  GT mask에서 경계 픽셀을 추출하여 해당 위치에 theta배의 가중치를 부여한
  weighted CrossEntropy를 계산한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[경계 픽셀 추출 방법]

  Laplacian 필터 방식 사용.
  GT mask를 float로 변환 후 3×3 Laplacian을 적용.
  인접 픽셀 간 class가 바뀌는 위치에서 0이 아닌 값이 나온다.

  Laplacian kernel (3×3):
      -1 -1 -1
      -1  8 -1
      -1 -1 -1

  dilate_kernel을 통해 경계 주변 픽셀까지 확장(dilate)한다.
  경계가 1px인 경우 GT noise에 취약하므로, dilate로 인접 픽셀도 포함.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[CrossEntropyLoss와의 구조 대응]

  CrossEntropyLoss          │ BoundaryLoss
  ──────────────────────────┼──────────────────────────────
  픽셀 단위 균등 손실        │ 경계 픽셀에 theta배 가중치 부여
  ignore_index 자동 처리    │ valid mask로 수동 처리
  경계/내부 구분 없음        │ Laplacian으로 경계 픽셀 추출

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[theta, dilate_kernel_size 선택 근거]

  theta=3.0: 경계 픽셀이 내부 픽셀 대비 3배 가중치.
             너무 크면 경계 외 클래스 학습이 방해됨.
  dilate_kernel_size=3: 경계 1px를 3px로 확장.
                        경계 레이블 noise에 대한 내성 확보.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[단일 변수 원칙]

  E4 실험: CrossEntropyLoss + BoundaryLoss 조합 (CombinedLoss 경유)
  단독 사용 시 내부 픽셀 학습이 부족할 수 있어 CE와 병용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  - Boundary Loss for Remote Sensing (Bokhovkin & Burnaev, 2019)
  - Detail-Sensitive Loss (Yuan et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BoundaryLoss(nn.Module):
    """
    Boundary-aware Loss — E4 실험에서 CrossEntropyLoss와 조합하여 사용.

    GT mask에서 Laplacian으로 경계 픽셀을 추출하고,
    경계 위치에 theta배 가중치를 부여한 weighted NLL을 계산한다.

    Args:
        ignore_index       (int):   무시할 class index. Default: 255.
        theta              (float): 경계 픽셀 가중치 배수. Default: 3.0.
        dilate_kernel_size (int):   경계 팽창 커널 크기. Default: 3.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        theta: float = 3.0,
        dilate_kernel_size: int = 3,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.theta        = theta

        # ── Laplacian 커널 (경계 검출용, 학습 안 됨) ──────────────────────────
        # 3×3 Laplacian: 중심값 8, 주변 -1
        # (1, 1, 3, 3) shape: Conv2d 입력 형식
        laplacian = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

        # ── Dilate 커널 (경계 팽창용, 학습 안 됨) ────────────────────────────
        # max_pool2d로 경계 영역을 dilate_kernel_size만큼 확장
        self.dilate_kernel_size = dilate_kernel_size

    def _extract_boundary(self, targets: Tensor) -> Tensor:
        """
        GT mask에서 경계 픽셀 맵을 추출한다.

        Args:
            targets : (B, H, W) int64 — ignore_index 포함 가능
        Returns:
            boundary : (B, H, W) float32 — 경계=1.0, 내부=0.0
                       ignore_index 위치는 0.0
        """
        # ── ignore 픽셀을 인접 클래스와 구분되지 않도록 처리 ─────────────────
        # Laplacian 전에 ignore_index → 0으로 치환
        # (실제 class 0과 구분이 안 되지만, valid_mask로 이후에 제거)
        valid_mask  = (targets != self.ignore_index)         # (B, H, W) bool
        targets_f   = targets.clone().float()
        targets_f[~valid_mask] = 0.0

        # ── Laplacian 적용 ────────────────────────────────────────────────────
        # (B, H, W) → (B, 1, H, W) : Conv2d 입력 형식
        t = targets_f.unsqueeze(1)                           # (B, 1, H, W)

        # padding=1로 공간 크기 유지
        # .to(t.device): 모듈이 .cuda() 호출을 받지 못한 경우에도 device를 맞춤
        edge = F.conv2d(t, self.laplacian.to(t.device), padding=1)  # (B, 1, H, W)

        # 0이 아닌 위치 = 경계 픽셀
        boundary = (edge.abs() > 0).float()                  # (B, 1, H, W)

        # ── Dilate: 경계 영역을 kernel_size만큼 팽창 ─────────────────────────
        # max_pool2d로 인접 픽셀까지 boundary=1로 확장
        if self.dilate_kernel_size > 1:
            pad      = self.dilate_kernel_size // 2
            boundary = F.max_pool2d(
                boundary,
                kernel_size=self.dilate_kernel_size,
                stride=1,
                padding=pad,
            )                                                # (B, 1, H, W)

        # ── ignore_index 위치 제거 ────────────────────────────────────────────
        boundary = boundary.squeeze(1)                       # (B, H, W)
        boundary = boundary * valid_mask.float()             # ignore 위치 = 0

        return boundary                                      # (B, H, W) float32

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits  : (B, num_classes, H, W)  float32 — raw logits
            targets : (B, H, W)               int64   — class index
        Returns:
            loss : scalar Tensor
        """
        # ── valid mask ────────────────────────────────────────────────────────
        valid_mask = (targets != self.ignore_index)          # (B, H, W) bool

        # ── 경계 픽셀 추출 ────────────────────────────────────────────────────
        # boundary: (B, H, W), 경계=1.0, 내부=0.0
        boundary = self._extract_boundary(targets)

        # ── 픽셀별 가중치 맵 ──────────────────────────────────────────────────
        # 내부 픽셀: 1.0 / 경계 픽셀: theta
        # boundary가 0~1 사이이므로: 1.0 + (theta - 1.0) * boundary
        weight_map = 1.0 + (self.theta - 1.0) * boundary    # (B, H, W)

        # ── ignore 픽셀을 0으로 치환 (NLL 계산을 위한 임시 처리) ─────────────
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                        # (B, H, W)

        # ── NLL 계산: -log(p_t) ───────────────────────────────────────────────
        log_prob = F.log_softmax(logits, dim=1)              # (B, C, H, W)
        nll = -log_prob.gather(
            dim=1,
            index=targets_safe.unsqueeze(1),
        ).squeeze(1)                                         # (B, H, W)

        # ── 가중치 적용 + ignore 마스킹 ──────────────────────────────────────
        loss = nll * weight_map * valid_mask.float()         # (B, H, W)

        # ── valid 픽셀 수로 정규화 ────────────────────────────────────────────
        loss = loss.sum() / valid_mask.float().sum().clamp(min=1.0)

        return loss