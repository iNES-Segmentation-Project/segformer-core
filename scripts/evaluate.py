"""
scripts/evaluate.py

best checkpoint를 로드해서 test set으로 최종 평가를 수행한다.
학습 없이 평가만 실행하는 스크립트.

실행:
    python scripts/evaluate.py --config configs/e0_internal.yaml
    python scripts/evaluate.py --config configs/e1_internal.yaml
    ...
"""

import sys
import os
import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.camvid import CamVidDataset, NUM_CLASSES, IGNORE_INDEX
from data.transforms import build_transform
from scripts.train import (
    load_config,
    build_model,
    build_criterion,
    validate,
    MeanIoU,
)


def main():
    parser = argparse.ArgumentParser(description="SegFormer-B0 Test Evaluation")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to yaml config file (e.g. configs/e0_internal.yaml)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config : {args.config}")
    print(f"  exp_name={cfg['exp_name']}  model={cfg['model_type']}  loss={cfg['loss_type']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── best checkpoint 경로 확인 ──────────────────────────────────────────────
    best_path = os.path.join(cfg["save_dir"], f"{cfg['exp_name']}_best.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"best checkpoint 없음: {best_path}\n"
            "  weights/ 디렉토리와 exp_name을 확인하세요."
        )
    print(f"Checkpoint: {best_path}")

    # ── test 경로 확인 ────────────────────────────────────────────────────────
    test_img_dir = cfg.get("test_img_dir", "")
    test_lbl_dir = cfg.get("test_lbl_dir", "")
    if not test_img_dir or not test_lbl_dir:
        raise ValueError(
            "config에 test_img_dir / test_lbl_dir 가 없습니다.\n"
            "  configs/*.yaml 파일에 경로를 추가하세요."
        )

    # ── Model 로드 ────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    miou_val = ckpt.get('miou', None)
    miou_str = f"{miou_val:.4f}" if miou_val is not None else "?"
    print(f"  best checkpoint epoch: {ckpt.get('epoch', '?')}  val_mIoU: {miou_str}")

    # ── Test Dataset ──────────────────────────────────────────────────────────
    aug_type = cfg.get("augmentation_type", "basic")
    test_transform = build_transform(aug_type, cfg["input_size"], split="val")
    test_dataset = CamVidDataset(
        image_dir=test_img_dir,
        label_dir=test_lbl_dir,
        transforms=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    print(f"Test   : {len(test_dataset)} samples")

    # ── 평가 ─────────────────────────────────────────────────────────────────
    criterion = build_criterion(cfg)
    test_loss, test_miou, test_per_class, test_mpa = validate(
        model, test_loader, criterion, device,
        cfg["num_classes"], cfg["ignore_index"],
    )

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    class_names = test_dataset.get_class_names()

    print("\n" + "=" * 60)
    print(f"[Test Result] {cfg['exp_name']}")
    print(f"  test_loss : {test_loss:.4f}")
    print(f"  test_mIoU : {test_miou:.4f}")
    print(f"  test_mPA  : {test_mpa:.4f}")
    print("  Per-class IoU:")
    for name, iou_val in zip(class_names, test_per_class):
        print(f"    {name:<12s}: {iou_val:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
