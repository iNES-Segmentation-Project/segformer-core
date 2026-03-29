"""
data/transforms.py

Segmentation용 transform pipeline.

핵심 원칙:
  - image와 mask는 항상 동일한 spatial transform을 받아야 한다.
  - mask resize는 반드시 nearest interpolation (class index가 뭉개지면 안 됨).
  - image resize는 bilinear interpolation.
  - augmentation은 제외 (baseline E0 목적).

사용 예:
    from data.transforms import SegTransform

    train_tf = SegTransform(size=(360, 480))
    dataset  = CamVidDataset(..., transforms=train_tf)
"""

import numpy as np
from PIL import Image
from typing import Tuple

import torch
from torch import Tensor


# ── ImageNet 기본 mean / std ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class SegTransform:
    """
    Segmentation용 통합 transform.

    적용 순서:
      1. Resize   : image → bilinear,  mask → nearest
      2. ToTensor : image PIL → float32 Tensor (3,H,W), [0,1]
                    mask PIL  → int64   Tensor (H,W)
      3. Normalize: image Tensor을 ImageNet mean/std로 정규화

    Args:
        size      (tuple): (H, W) 출력 크기.
        mean      (list):  채널별 mean. Default: ImageNet mean.
        std       (list):  채널별 std.  Default: ImageNet std.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        mean: list = IMAGENET_MEAN,
        std:  list = IMAGENET_STD,
    ):
        self.size = size   # (H, W)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)

    def __call__(
        self,
        image: Image.Image,
        mask:  Image.Image,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            image : PIL Image (H_orig, W_orig, 3)  RGB
            mask  : PIL Image mode "I"  (H_orig, W_orig)  class index int32

        Returns:
            image : Tensor (3, H, W)  float32, normalized
            mask  : Tensor (H, W)     int64,   values ∈ {0..10, 255}
        """
        H, W = self.size

        # ── Step 1: Resize ────────────────────────────────────────────────────
        # image: bilinear (시각적 품질 유지)
        image = image.resize((W, H), resample=Image.BILINEAR)

        # mask: nearest (class index가 보간되면 안 됨)
        mask = mask.resize((W, H), resample=Image.NEAREST)

        # ── Step 2: ToTensor ──────────────────────────────────────────────────
        # image: (H, W, 3) uint8 → (3, H, W) float32, [0, 1]
        image_np = np.array(image, dtype=np.float32) / 255.0   # (H, W, 3)
        image_t  = torch.from_numpy(
            image_np.transpose(2, 0, 1)                         # (3, H, W)
        )

        # mask: PIL mode "I" (int32) → (H, W) int64
        mask_np = np.array(mask, dtype=np.int64)                # (H, W)
        mask_t  = torch.from_numpy(mask_np)                     # (H, W)

        # ── Step 3: Normalize image ───────────────────────────────────────────
        # (3, H, W) - mean / std, both (3, 1, 1)
        image_t = (image_t - self.mean) / self.std              # (3, H, W)

        return image_t, mask_t  # (3,H,W) float32, (H,W) int64