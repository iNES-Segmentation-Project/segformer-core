"""
data/transforms.py

Segmentation용 transform pipeline.

제공 클래스/함수:
  - SegTransform       : 기존 transform (하위 호환용, augmentation 없음)
  - PaperlikeTransform : paper-like augmentation (train split에만 랜덤 증강 적용)
  - build_transform    : augmentation_type 문자열로 transform을 선택하는 팩토리 함수

augmentation_type:
  - "basic"      : Resize → ToTensor → Normalize (기존 SegTransform과 동일)
  - "paperlike"  : train split → RandomResize + Pad + RandomCrop + HFlip + ColorJitter
                              → ToTensor → Normalize
                   val split  → Resize → ToTensor → Normalize (basic과 동일)

Interpolation 규칙:
  - image : 모든 resize에서 BILINEAR
  - mask  : 모든 resize에서 NEAREST  ← class index 깨짐 방지

Spatial transform (resize / crop / flip)은 image / mask에 반드시 동일 파라미터 적용.
Photometric transform (ColorJitter)은 image에만 적용.
"""

import random
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple

import torch
from torch import Tensor


# ── ImageNet 기본 mean / std ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_IGNORE_INDEX = 255   # mask padding 값 (Void)


# =============================================================================
# ── 기존 transform (하위 호환 유지 — 변경 금지) ──────────────────────────────
# =============================================================================

class SegTransform:
    """
    Segmentation용 통합 transform.

    적용 순서:
      1. Resize   : image → BILINEAR,  mask → NEAREST
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


# =============================================================================
# ── Paper-like augmentation ───────────────────────────────────────────────────
# =============================================================================

class PaperlikeTransform:
    """
    Paper-like augmentation (SegFormer / MMSeg 설정 참고).

    train split 적용 순서:
      1. RandomResize    : scale ∈ [scale_min, scale_max] — image BILINEAR, mask NEAREST
      2. Pad             : crop size보다 작으면 패딩 — image 0(black), mask IGNORE_INDEX
      3. RandomCrop      : (H, W) 크기 — image / mask 동일 위치
      4. RandomHFlip     : p=flip_prob — image / mask 동일 방향
      5. ColorJitter     : brightness / contrast jitter — image에만 적용
      6. ToTensor        : image → float32 (3,H,W),  mask → int64 (H,W)
      7. Normalize       : image에만 (ImageNet mean/std)

    val split 적용 순서:
      Resize → ToTensor → Normalize   (BasicTransform과 동일)

    Args:
        size        (tuple): (H, W) 출력 크기.
        split       (str):   "train" | "val". train일 때만 랜덤 증강 적용.
        scale_range (tuple): RandomResize scale 범위. Default: (0.5, 2.0).
        flip_prob   (float): HorizontalFlip 확률. Default: 0.5.
        brightness  (float): ColorJitter brightness jitter 강도. Default: 0.2.
        contrast    (float): ColorJitter contrast jitter 강도. Default: 0.2.
        saturation  (float): ColorJitter saturation jitter 강도. Default: 0.2.
                             SegFormer 논문 PhotoMetricDistortion의 saturation에 대응.
                             PIL ImageEnhance.Color 사용 (torchvision 불필요).
        mean        (list):  Normalize mean. Default: ImageNet mean.
        std         (list):  Normalize std.  Default: ImageNet std.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        split: str = "train",
        scale_range: Tuple[float, float] = (0.5, 2.0),
        flip_prob: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        mean: list = IMAGENET_MEAN,
        std:  list = IMAGENET_STD,
    ):
        self.size        = size          # (H, W)
        self.split       = split
        self.scale_range = scale_range
        self.flip_prob   = flip_prob
        self.brightness  = brightness
        self.contrast    = contrast
        self.saturation  = saturation
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)

    def __call__(
        self,
        image: Image.Image,
        mask:  Image.Image,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            image : PIL Image RGB
            mask  : PIL Image mode "I"  class index int32

        Returns:
            image : Tensor (3, H, W)  float32, normalized
            mask  : Tensor (H, W)     int64,   values ∈ {0..10, 255}
        """
        H, W = self.size

        if self.split != "train":
            # val / test: resize only (basic과 동일)
            image = image.resize((W, H), resample=Image.BILINEAR)
            mask  = mask.resize((W, H),  resample=Image.NEAREST)

        else:
            # ── Step 1: RandomResize ──────────────────────────────────────────
            # scale을 샘플링하고 image / mask 동일 크기로 리사이즈
            image, mask = self._random_resize(image, mask)

            # ── Step 2: Pad if needed ─────────────────────────────────────────
            # crop size보다 작으면 패딩 (scale < 1.0 경우 발생)
            # image: 0(black),  mask: IGNORE_INDEX(255)
            image, mask = self._pad_if_needed(image, mask)

            # ── Step 3: RandomCrop ────────────────────────────────────────────
            # image / mask 동일 (top, left) 좌표에서 crop
            image, mask = self._random_crop(image, mask)

            # ── Step 4: RandomHorizontalFlip ──────────────────────────────────
            # image / mask 동일 방향으로 flip
            if random.random() < self.flip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # ── Step 5: ColorJitter (image only) ─────────────────────────────
            # mask에는 적용하지 않음
            image = self._color_jitter(image)

        # ── Step 6: ToTensor ──────────────────────────────────────────────────
        image_np = np.array(image, dtype=np.float32) / 255.0   # (H, W, 3)
        image_t  = torch.from_numpy(image_np.transpose(2, 0, 1))  # (3, H, W)

        mask_np = np.array(mask, dtype=np.int64)                # (H, W)
        mask_t  = torch.from_numpy(mask_np)

        # ── Step 7: Normalize (image only) ───────────────────────────────────
        image_t = (image_t - self.mean) / self.std

        return image_t, mask_t   # (3,H,W) float32, (H,W) int64

    # ── 내부 헬퍼 메서드 ──────────────────────────────────────────────────────

    def _random_resize(
        self,
        image: Image.Image,
        mask:  Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        target size에 scale을 곱한 크기로 리사이즈.
        image: BILINEAR,  mask: NEAREST
        """
        H, W = self.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        mask  = mask.resize((new_w, new_h),  resample=Image.NEAREST)
        return image, mask

    def _pad_if_needed(
        self,
        image: Image.Image,
        mask:  Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        crop size(H, W)보다 작은 경우 우/하단에 패딩.
        image: RGB(0,0,0),  mask: IGNORE_INDEX
        """
        H, W = self.size
        img_w, img_h = image.size   # PIL.size = (width, height)
        pad_h = max(0, H - img_h)
        pad_w = max(0, W - img_w)
        if pad_h > 0 or pad_w > 0:
            new_w = img_w + pad_w
            new_h = img_h + pad_h
            new_image = Image.new("RGB", (new_w, new_h), 0)
            new_image.paste(image, (0, 0))
            new_mask = Image.new("I", (new_w, new_h), _IGNORE_INDEX)
            new_mask.paste(mask, (0, 0))
            return new_image, new_mask
        return image, mask

    def _random_crop(
        self,
        image: Image.Image,
        mask:  Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        image / mask를 동일한 (top, left) 좌표에서 (H, W) 크기로 crop.
        """
        H, W = self.size
        img_w, img_h = image.size
        top  = random.randint(0, img_h - H)
        left = random.randint(0, img_w - W)
        image = image.crop((left, top, left + W, top + H))
        mask  = mask.crop((left, top, left + W, top + H))
        return image, mask

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """
        Brightness / Contrast / Saturation jitter — PIL ImageEnhance 사용.
        torchvision 의존성 없음. mask에는 적용하지 않음.
        factor = 1.0 + Uniform(-strength, +strength)

        SegFormer 논문 PhotoMetricDistortion 대응:
          brightness  → ImageEnhance.Brightness
          contrast    → ImageEnhance.Contrast
          saturation  → ImageEnhance.Color  (색 채도)
        """
        # 적용 순서를 랜덤으로 섞어 더 다양한 augmentation 생성
        ops = []
        if self.brightness > 0:
            ops.append(('brightness', self.brightness))
        if self.contrast > 0:
            ops.append(('contrast', self.contrast))
        if self.saturation > 0:
            ops.append(('saturation', self.saturation))
        random.shuffle(ops)

        for op, strength in ops:
            factor = 1.0 + random.uniform(-strength, strength)
            factor = max(0.0, factor)
            if op == 'brightness':
                image = ImageEnhance.Brightness(image).enhance(factor)
            elif op == 'contrast':
                image = ImageEnhance.Contrast(image).enhance(factor)
            elif op == 'saturation':
                image = ImageEnhance.Color(image).enhance(factor)
        return image


# =============================================================================
# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
# =============================================================================

def build_transform(
    augmentation_type: str,
    size: Tuple[int, int],
    split: str = "train",
) -> object:
    """
    augmentation_type과 split에 맞는 transform을 반환.

    Args:
        augmentation_type : "basic" | "paperlike"
        size              : (H, W) 출력 해상도
        split             : "train" | "val"

    Returns:
        callable : (PIL.Image, PIL.Image) → (Tensor, Tensor)

    Note:
        "basic" 은 split에 관계없이 항상 동일하게 Resize + Normalize.
        "paperlike" 는 split="train" 일 때만 랜덤 증강 적용.
    """
    if augmentation_type == "basic":
        return SegTransform(size=size)
    elif augmentation_type == "paperlike":
        return PaperlikeTransform(size=size, split=split)
    else:
        raise ValueError(
            f"Unknown augmentation_type: '{augmentation_type}'. "
            "Choose 'basic' or 'paperlike'."
        )
