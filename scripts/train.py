"""
scripts/train.py

SegFormer-B0 학습 스크립트 — config 기반.

실행:
    python scripts/train.py --config configs/e0_internal.yaml
    python scripts/train.py --config configs/e0_paperlike.yaml

디렉토리 구조 :
    segformer-core/
    ├── configs/
    │   ├── e0_internal.yaml
    │   └── e0_paperlike.yaml
    ├── data/
    │   ├── Camvid/
    │   │   ├── images/
    │   │   │   ├── train/
    │   │   │   └── val/
    │   │   └── labels/
    │   │       ├── train/
    │   │       └── val/
    ├── models/
    ├── scripts/
    │   └── train.py   ← 이 파일
    └── weights/       ← checkpoint 저장 위치
"""

import sys
import os
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 프로젝트 루트를 sys.path에 추가 (scripts/ 내부에서 실행 시) ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.camvid import CamVidDataset, NUM_CLASSES, IGNORE_INDEX
from data.transforms import build_transform
from models.segformer import build_segformer_b0, build_segformer_b0_fpn
from models.loss.cross_entropy import CrossEntropyLoss
from models.loss.focal_loss import FocalLoss
from models.loss.combined_loss import CombinedLoss


# =============================================================================
# ── Config 로드 ───────────────────────────────────────────────────────────────
# =============================================================================

def load_config(config_path: str) -> dict:
    """yaml 파일을 읽어 dict로 반환. 경로는 ROOT 기준 상대경로 허용."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 데이터/가중치 경로를 절대경로로 변환
    for key in ("train_img_dir", "train_lbl_dir", "val_img_dir", "val_lbl_dir",
                "test_img_dir", "test_lbl_dir", "save_dir", "pretrained_path"):
        if key in cfg and cfg[key]:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str(ROOT / p)

    # input_size: list → tuple
    if "input_size" in cfg:
        cfg["input_size"] = tuple(cfg["input_size"])

    # 기본값 보완
    cfg.setdefault("num_classes",  NUM_CLASSES)
    cfg.setdefault("ignore_index", IGNORE_INDEX)
    cfg.setdefault("embed_dim",    256)
    cfg.setdefault("dropout",      0.1)
    cfg.setdefault("num_workers",  2)
    cfg.setdefault("warmup_ratio", 0.1)

    return cfg


# =============================================================================
# ── 팩토리 함수 ───────────────────────────────────────────────────────────────
# =============================================================================

def load_pretrained_encoder(model: nn.Module, weight_path: str) -> None:
    """
    encoder에만 pretrained 가중치를 로드.
    decoder / segmentation head는 로드하지 않음.

    지원 형식:
      1. 우리 full model 체크포인트 (train.py 저장 형식)
         → ckpt["model"] 에서 "encoder.*" 키만 추출 후 prefix 제거
      2. encoder-only state_dict
         → keys가 "stages.*" 로 시작하면 그대로 사용

    MMSeg backbone 형식 ("backbone.*")은 미지원 — 키 변환 필요.
    """
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(
            f"[pretrained] 가중치 파일 없음: {path}\n"
            "  config의 pretrained_path 경로를 확인하세요."
        )

    ckpt = torch.load(str(path), map_location="cpu")

    # ── checkpoint wrapper 처리 ────────────────────────────────────────────────
    # train.py가 저장하는 형식: {"epoch": ..., "model": {...}, "optimizer": ...}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # 이미 state_dict

    # ── key 형식 감지 및 정규화 (전체 키 기준) ───────────────────────────────
    all_keys = list(state_dict.keys())

    has_encoder_prefix  = any(k.startswith("encoder.")  for k in all_keys)
    has_backbone_prefix = any(k.startswith("backbone.") for k in all_keys)

    if has_encoder_prefix:
        # 형식 1: full model state_dict → encoder 키만 추출 후 prefix 제거
        state_dict = {
            k[len("encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
    elif has_backbone_prefix:
        raise ValueError(
            "[pretrained] MMSeg backbone 형식('backbone.*')은 미지원.\n"
            "  지원 형식: (1) 우리 체크포인트의 'encoder.*' 키\n"
            "             (2) encoder-only state_dict의 'stages.*' 키"
        )
    # else: 형식 2 — 이미 encoder-only ('stages.*'), 그대로 사용

    # ── encoder에만 로드 (decoder 키는 state_dict에 없으므로 자동 제외) ────────
    missing, unexpected = model.encoder.load_state_dict(state_dict, strict=False)

    print(f"  [pretrained] 로드 완료: {path.name}")
    if not missing and not unexpected:
        print("  [pretrained] 모든 encoder 키 정상 로드.")
    if missing:
        print(f"  [pretrained] Missing keys  ({len(missing)}): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  [pretrained] Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")


def build_model(cfg: dict) -> nn.Module:
    """model_type에 따라 SegFormer 모델을 생성. pretrained=true 이면 encoder에만 가중치 로드."""
    model_type = cfg.get("model_type", "mlp")

    if model_type == "mlp":
        model = build_segformer_b0(
            num_classes=cfg["num_classes"],
            embed_dim=cfg["embed_dim"],
            dropout=cfg["dropout"],
        )
    elif model_type == "fpn":
        model = build_segformer_b0_fpn(
            num_classes=cfg["num_classes"],
            fpn_dim=cfg["embed_dim"],
            dropout=cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'mlp' or 'fpn'.")

    # ── pretrained encoder 로드 (E5 / paperlike 전용) ─────────────────────────
    # pretrained=false 이면 이 블록 전체를 건너뜀 → E0~E4에 영향 없음
    if cfg.get("pretrained", False):
        pretrained_path = cfg.get("pretrained_path", "")
        if not pretrained_path:
            raise ValueError(
                "[pretrained] pretrained=true이지만 pretrained_path가 비어 있습니다.\n"
                "  config에 pretrained_path: <path/to/weights.pth> 를 추가하세요."
            )
        load_pretrained_encoder(model, pretrained_path)

    return model


def build_criterion(cfg: dict) -> nn.Module:
    """loss_type에 따라 loss 함수를 생성."""
    loss_type    = cfg.get("loss_type", "ce")
    ignore_index = cfg["ignore_index"]
    num_classes  = cfg["num_classes"]

    if loss_type == "ce":
        return CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "focal":
        return FocalLoss(ignore_index=ignore_index)
    elif loss_type == "ce_dice":
        return CombinedLoss(mode="ce+dice", num_classes=num_classes, ignore_index=ignore_index)
    elif loss_type == "ce_boundary":
        return CombinedLoss(mode="ce+boundary", num_classes=num_classes, ignore_index=ignore_index)
    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            "Choose 'ce', 'focal', 'ce_dice', or 'ce_boundary'."
        )


def build_scheduler(
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    total_iters: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """scheduler_type에 따라 LR scheduler를 생성."""
    scheduler_type = cfg.get("scheduler_type", "poly")

    if scheduler_type == "poly":
        def poly_lr(iteration):
            return (1 - iteration / total_iters) ** 0.9
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    elif scheduler_type == "warmup_poly":
        warmup_iters = int(total_iters * cfg["warmup_ratio"])

        def warmup_poly_lr(iteration):
            if iteration < warmup_iters:
                return (iteration + 1) / warmup_iters
            progress = (iteration - warmup_iters) / max(total_iters - warmup_iters, 1)
            return (1 - progress) ** 0.9

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_poly_lr)

    else:
        raise ValueError(
            f"Unknown scheduler_type: '{scheduler_type}'. "
            "Choose 'poly' or 'warmup_poly'."
        )


# =============================================================================
# ── mIoU 계산 ─────────────────────────────────────────────────────────────────
# =============================================================================

class MeanIoU:
    """
    Incremental mIoU calculator.
    매 batch마다 confusion matrix를 누적하고, epoch 끝에 한 번에 계산.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.reset()

    def reset(self):
        # (num_classes, num_classes): rows=GT, cols=Pred
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   : (B, H, W)  int64 — argmax 결과
            targets : (B, H, W)  int64 — GT class index
        """
        # ignore_index 제거
        valid = targets != self.ignore_index           # (B, H, W) bool
        preds_v   = preds[valid]                       # (N,)
        targets_v = targets[valid]                     # (N,)

        # valid range 체크
        valid_range = (targets_v >= 0) & (targets_v < self.num_classes)
        preds_v   = preds_v[valid_range]
        targets_v = targets_v[valid_range]

        # confusion matrix 누적
        # gt * num_classes + pred → 1D index → bincount → reshape
        idx = targets_v * self.num_classes + preds_v
        self.confusion += torch.bincount(
            idx, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Returns:
            miou        : float  — mean IoU over valid classes
            per_class   : list   — per-class IoU (NaN if class absent)
        """
        conf = self.confusion.float()
        intersection = conf.diag()                     # (C,)
        union = conf.sum(1) + conf.sum(0) - intersection  # (C,)

        iou = intersection / union.clamp(min=1e-6)    # (C,)

        # GT에 한 번도 등장하지 않은 class는 제외
        valid_classes = conf.sum(1) > 0
        miou = iou[valid_classes].mean().item()

        return miou, iou.tolist()


# =============================================================================
# ── 학습 1 epoch ──────────────────────────────────────────────────────────────
# =============================================================================

def train_one_epoch(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scheduler_type: str,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Returns:
        avg_loss : float

    Scheduler stepping:
        - "poly"        : epoch 기반 → main 루프에서 epoch 종료 후 step()
        - "warmup_poly" : iteration 기반 → 여기서 매 batch 후 step()
    """
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        # ── Forward ──────────────────────────────────────────────────────────
        logits = model(images)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss = criterion(logits, masks)

        # ── Backward ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # warmup_poly: iteration 단위로 step (validation과 무관하게 batch마다)
        if scheduler_type == "warmup_poly":
            scheduler.step()

        total_loss += loss.item()

        # 진행상황 출력 (10 batch마다)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  [Train] Epoch {epoch:3d} "
                f"[{batch_idx+1:3d}/{n_batches}] "
                f"loss: {loss.item():.4f}"
            )

    return total_loss / n_batches


# =============================================================================
# ── 검증 1 epoch ──────────────────────────────────────────────────────────────
# =============================================================================

@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> tuple:
    """
    Returns:
        avg_loss : float
        miou     : float
    """
    model.eval()
    total_loss = 0.0
    miou_calc  = MeanIoU(num_classes=num_classes, ignore_index=ignore_index)

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)

        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        miou_calc.update(preds.cpu(), masks.cpu())

    avg_loss = total_loss / len(loader)
    miou, per_class_iou = miou_calc.compute()

    return avg_loss, miou, per_class_iou


# =============================================================================
# ── Checkpoint 저장 ───────────────────────────────────────────────────────────
# =============================================================================

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    miou: float,
    save_dir: str,
    exp_name: str,
    is_best: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)

    state = {
        "epoch":      epoch,
        "miou":       miou,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }

    # 매 epoch 저장 (덮어쓰기)
    last_path = os.path.join(save_dir, f"{exp_name}_last.pth")
    torch.save(state, last_path)

    # best 모델 별도 저장
    if is_best:
        best_path = os.path.join(save_dir, f"{exp_name}_best.pth")
        torch.save(state, best_path)
        print(f"  ★ Best model saved → {best_path}  (mIoU: {miou:.4f})")


# =============================================================================
# ── Main ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    # ── argparse ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="SegFormer-B0 Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to yaml config file (e.g. configs/e0_internal.yaml)"
    )
    args = parser.parse_args()

    # ── Config 로드 ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"  exp_name={cfg['exp_name']}  model={cfg['model_type']}  "
          f"loss={cfg['loss_type']}  scheduler={cfg['scheduler_type']}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Transform ─────────────────────────────────────────────────────────────
    aug_type = cfg.get("augmentation_type", "basic")
    train_transform = build_transform(aug_type, cfg["input_size"], split="train")
    val_transform   = build_transform(aug_type, cfg["input_size"], split="val")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dataset = CamVidDataset(
        image_dir=cfg["train_img_dir"],
        label_dir=cfg["train_lbl_dir"],
        transforms=train_transform,
    )
    val_dataset = CamVidDataset(
        image_dir=cfg["val_img_dir"],
        label_dir=cfg["val_lbl_dir"],
        transforms=val_transform,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val  : {len(val_dataset)} samples")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_criterion(cfg)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg["lr"] * 10},
            {"params": decoder_params, "lr": cfg["lr"] * 10},
        ],
        weight_decay=cfg["weight_decay"],
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    total_iters    = cfg["epochs"] * len(train_loader)
    scheduler_type = cfg.get("scheduler_type", "poly")
    scheduler      = build_scheduler(cfg, optimizer, total_iters)

    # stepping 기준 안내
    # - poly        : epoch 단위 (main 루프에서 1회/epoch) — E0~E4 baseline과 동일
    # - warmup_poly : iteration 단위 (train_one_epoch 내 batch마다) — E5 전용

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    best_miou = 0.0

    print("\n" + "=" * 60)
    print(f"Training: {cfg['exp_name']}  |  {cfg['epochs']} epochs")
    print("=" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scheduler_type, device, epoch,
        )

        # poly: epoch 단위 step (validation 전에 수행, validation과 섞이지 않음)
        if scheduler_type != "warmup_poly":
            scheduler.step()

        # Validate
        val_loss, miou, per_class_iou = validate(
            model, val_loader, criterion, device,
            cfg["num_classes"], cfg["ignore_index"],
        )

        elapsed = time.time() - t0

        # 출력
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch [{epoch:3d}/{cfg['epochs']}] "
            f"| train_loss: {train_loss:.4f} "
            f"| val_loss: {val_loss:.4f} "
            f"| mIoU: {miou:.4f} "
            f"| lr: {current_lr:.2e} "
            f"| time: {elapsed:.1f}s"
        )

        # Per-class IoU 출력 (5 epoch마다)
        if epoch % 5 == 0:
            class_names = train_dataset.get_class_names()
            print("  Per-class IoU:")
            for name, iou_val in zip(class_names, per_class_iou):
                print(f"    {name:<12s}: {iou_val:.4f}")

        # Checkpoint 저장
        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        save_checkpoint(
            model, optimizer, epoch, miou,
            cfg["save_dir"], cfg["exp_name"],
            is_best=is_best,
        )

    print("\n" + "=" * 60)
    print(f"Training complete. Best val mIoU: {best_miou:.4f}")
    print("=" * 60)

    # ── Test evaluation (best checkpoint 기준, 1회) ───────────────────────────
    test_img_dir = cfg.get("test_img_dir", "")
    test_lbl_dir = cfg.get("test_lbl_dir", "")

    if test_img_dir and test_lbl_dir:
        best_path = os.path.join(cfg["save_dir"], f"{cfg['exp_name']}_best.pth")
        if not os.path.exists(best_path):
            print(f"\n[Test] best checkpoint 없음: {best_path} — test 생략")
        else:
            print(f"\n[Test] best checkpoint 로드: {best_path}")
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])

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
            print(f"[Test] {len(test_dataset)} samples")

            test_loss, test_miou, test_per_class = validate(
                model, test_loader, criterion, device,
                cfg["num_classes"], cfg["ignore_index"],
            )

            print("\n" + "=" * 60)
            print(f"[Test Result] {cfg['exp_name']}")
            print(f"  test_loss : {test_loss:.4f}")
            print(f"  test_mIoU : {test_miou:.4f}")
            print("  Per-class IoU:")
            class_names = train_dataset.get_class_names()
            for name, iou_val in zip(class_names, test_per_class):
                print(f"    {name:<12s}: {iou_val:.4f}")
            print("=" * 60)
    else:
        print("\n[Test] test_img_dir / test_lbl_dir 미설정 — test 생략")


if __name__ == "__main__":
    main()
