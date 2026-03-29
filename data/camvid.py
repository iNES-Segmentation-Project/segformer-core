"""
data/camvid.py

CamVid Dataset loader.

CamVid 디렉토리 구조 (실제 다운로드 기준):
    datasets/CamVid/
    ├── train/          *.png   (이미지)
    ├── train_labels/   *.png   (RGB palette mask, 이미지와 동일 파일명)
    ├── val/            *.png
    ├── val_labels/     *.png
    ├── test/           *.png
    └── test_labels/    *.png

CamVid 11 classes (void 제외):
    0  Sky
    1  Building
    2  Pole
    3  Road
    4  Pavement
    5  Tree
    6  SignSymbol
    7  Fence
    8  Car
    9  Pedestrian
    10 Bicyclist
    255 Void / ignore

RGB palette mask → class index 변환:
    CamVid의 label 이미지는 RGB 컬러로 저장되어 있다.
    각 pixel의 RGB 값을 CLASS_MAP으로 lookup해서 class index로 변환한다.

class_dict.csv:
    다운로드 시 포함된 class_dict.csv는 색상표 참고용이며,
    본 코드에서는 CAMVID_COLORMAP에 직접 하드코딩되어 있으므로 사용하지 않아도 된다.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset


# ── CamVid 11-class RGB palette ───────────────────────────────────────────────
# (R, G, B) → class index
# 출처: SegNet/CamVid 공식 color map
CAMVID_COLORMAP: List[Tuple[int, int, int]] = [
    (128, 128, 128),   # 0  Sky
    (128,   0,   0),   # 1  Building
    (192, 192, 128),   # 2  Pole
    (128,  64, 128),   # 3  Road
    (60,  40, 222),    # 4  Pavement
    (128, 128,   0),   # 5  Tree
    (192, 128, 128),   # 6  SignSymbol
    ( 64,  64, 128),   # 7  Fence
    ( 64,   0, 128),   # 8  Car
    ( 64,  64,   0),   # 9  Pedestrian
    (  0, 128, 192),   # 10 Bicyclist
]

NUM_CLASSES = 11
IGNORE_INDEX = 255


def _build_color_lookup() -> np.ndarray:
    """
    RGB → class index lookup table.
    shape: (256, 256, 256) — uint8, 값: class index 또는 IGNORE_INDEX

    메모리 절약을 위해 실제로는 (R//8, G//8, B//8) 축소 테이블을 쓰지 않고
    dictionary lookup 방식을 사용한다.
    """
    table = {}
    for idx, rgb in enumerate(CAMVID_COLORMAP):
        table[rgb] = idx
    return table


# 모듈 로드 시 1회만 생성
_COLOR_TABLE = _build_color_lookup()


def _rgb_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """
    RGB mask (H, W, 3) → class index mask (H, W), dtype=int64.
    알 수 없는 색상은 IGNORE_INDEX(255)로 처리.
    """
    H, W, _ = mask_rgb.shape
    index_mask = np.full((H, W), IGNORE_INDEX, dtype=np.int64)

    for rgb, cls_idx in _COLOR_TABLE.items():
        # 각 채널이 모두 일치하는 pixel을 찾아 class index 할당
        match = (
            (mask_rgb[:, :, 0] == rgb[0]) &
            (mask_rgb[:, :, 1] == rgb[1]) &
            (mask_rgb[:, :, 2] == rgb[2])
        )
        index_mask[match] = cls_idx

    return index_mask  # (H, W), values ∈ {0..10, 255}


class CamVidDataset(Dataset):
    """
    CamVid Semantic Segmentation Dataset.

    Args:
        image_dir  (str): 이미지 폴더 경로 (e.g., "camvid/images/train")
        label_dir  (str): 라벨 폴더 경로 (e.g., "camvid/labels/train")
        transforms (callable, optional): data/transforms.py의 SegTransform 인스턴스.
                   None이면 PIL Image 그대로 반환하지 않고 기본 ToTensor 적용.
        img_suffix  (str): 이미지 파일 확장자. Default: ".png"
        mask_suffix (str): 마스크 파일 확장자. Default: ".png"

    Returns (per __getitem__):
        image : Tensor (3, H, W)  float32, [0,1] 또는 normalized
        mask  : Tensor (H, W)     int64,   values ∈ {0..10, 255}
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transforms: Optional[Callable] = None,
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        # 이미지 파일 목록 수집 및 정렬 (재현성을 위해 정렬)
        self.image_paths = sorted(
            self.image_dir.glob(f"*{img_suffix}")
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir} with suffix {img_suffix}"
            )

        # 대응하는 mask 경로 생성
        # 실제 CamVid 다운로드 구조:
        #   train/0001TP_009210.png  →  train_labels/0001TP_009210_L.png
        # 이미지 stem에 "_L" 접미사가 붙은 형태.
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_name = img_path.stem + "_L" + mask_suffix   # e.g. 0001TP_009210_L.png
            mask_path = self.label_dir / mask_name
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                raise FileNotFoundError(
                    f"Mask not found for {img_path.name} in {label_dir}\n"
                    f"  Expected: {mask_path}"
                )

        assert len(self.image_paths) == len(self.mask_paths), \
            "image / mask 수가 일치하지 않습니다."

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            image : Tensor (3, H, W)  float32
            mask  : Tensor (H, W)     int64
        """
        # ── Load image ────────────────────────────────────────────────────────
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # ── Load mask ─────────────────────────────────────────────────────────
        mask_pil = Image.open(self.mask_paths[idx]).convert("RGB")
        # RGB PIL Image (H, W, 3) → numpy
        mask_rgb = np.array(mask_pil, dtype=np.uint8)    # (H, W, 3)
        # RGB → class index (H, W), int64
        mask_np  = _rgb_mask_to_index(mask_rgb)          # (H, W)

        # PIL Image로 다시 감싸서 transforms에서 resize 가능하게 함
        # dtype 보존을 위해 int32 PIL mode "I" 사용
        mask_pil_idx = Image.fromarray(mask_np.astype(np.int32), mode="I")

        # ── Apply transforms ──────────────────────────────────────────────────
        if self.transforms is not None:
            image, mask_pil_idx = self.transforms(image, mask_pil_idx)
            # transforms가 Tensor로 변환해서 반환하는 경우
            if isinstance(image, Tensor):
                mask = mask_pil_idx
                if not isinstance(mask, Tensor):
                    mask = torch.from_numpy(
                        np.array(mask_pil_idx, dtype=np.int64)
                    )
                return image, mask  # (3,H,W), (H,W)

        # transforms 없을 때: 기본 변환
        # image: PIL → float32 Tensor (3, H, W), [0, 1]
        image_np = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
        image_tensor = torch.from_numpy(
            image_np.transpose(2, 0, 1)                        # (3, H, W)
        )
        mask_tensor = torch.from_numpy(
            np.array(mask_pil_idx, dtype=np.int64)             # (H, W)
        )

        return image_tensor, mask_tensor  # (3,H,W) float32, (H,W) int64

    def get_class_names(self) -> List[str]:
        return [
            "Sky", "Building", "Pole", "Road", "Pavement",
            "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
        ]