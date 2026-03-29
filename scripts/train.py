"""
scripts/train.py

SegFormer-B0 baseline (E0: MLP decoder + CrossEntropy) — CamVid 학습 스크립트.

실행:
    python scripts/train.py

디렉토리 구조 :
    segformer-core/
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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 프로젝트 루트를 sys.path에 추가 (scripts/ 내부에서 실행 시) ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.camvid import CamVidDataset, NUM_CLASSES, IGNORE_INDEX
from data.transforms import SegTransform
from models.segformer import build_segformer_b0
from models.loss.cross_entropy import CrossEntropyLoss


# =============================================================================
# ── 하드코딩 설정값 (argparse 없이 직접 수정해서 사용) ──────────────────────
# =============================================================================

CFG = {
    # ── 데이터 경로 ────────────────────────────────────────────────────────────
    # 실제 폴더 구조: datasets/camVid/train/, datasets/camVid/train_labels/, ...
    "train_img_dir":   str(ROOT / "datasets" / "CamVid" / "train"),
    "train_lbl_dir":   str(ROOT / "datasets" / "CamVid" / "train_labels"),
    "val_img_dir":     str(ROOT / "datasets" / "CamVid" / "val"),
    "val_lbl_dir":     str(ROOT / "datasets" / "CamVid" / "val_labels"),

    # ── 입력 해상도 (H, W) ─────────────────────────────────────────────────────
    # CamVid 원본: 720×960 → 학습 시 절반으로 줄이는 것이 일반적
    "input_size":      (360, 480),

    # ── 학습 설정 ──────────────────────────────────────────────────────────────
    "num_classes":     NUM_CLASSES,   # 11
    "ignore_index":    IGNORE_INDEX,  # 255
    "epochs":          40,
    "batch_size":      4,
    "num_workers":     2,
    "lr":              6e-5,          # SegFormer 논문 권장값
    "weight_decay":    0.01,

    # ── 모델 설정 ──────────────────────────────────────────────────────────────
    "embed_dim":       256,           # MLP decoder 채널 수
    "dropout":         0.1,

    # ── 저장 경로 ──────────────────────────────────────────────────────────────
    "save_dir":        str(ROOT / "weights"),
    "exp_name":        "e0_mlp_ce",   # 실험 이름 (E0 baseline)
}


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
        # IoU_c = TP_c / (TP_c + FP_c + FN_c)
        # TP_c = conf[c, c]
        # FP_c = sum(conf[:, c]) - conf[c, c]
        # FN_c = sum(conf[c, :]) - conf[c, c]
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
    criterion: CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Returns:
        avg_loss : float
    """
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        # images : (B, 3, H, W)  float32
        # masks  : (B, H, W)     int64
        images = images.to(device)
        masks  = masks.to(device)

        # ── Forward ──────────────────────────────────────────────────────────
        # logits: (B, num_classes, H, W)
        logits = model(images)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss = criterion(logits, masks)

        # ── Backward ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    criterion: CrossEntropyLoss,
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
        # images : (B, 3, H, W)
        # masks  : (B, H, W)
        images = images.to(device)
        masks  = masks.to(device)

        # logits : (B, num_classes, H, W)
        logits = model(images)

        # loss
        loss = criterion(logits, masks)
        total_loss += loss.item()

        # mIoU update
        # preds : (B, H, W)  class index
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
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Transform ─────────────────────────────────────────────────────────────
    transform = SegTransform(size=CFG["input_size"])

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dataset = CamVidDataset(
        image_dir=CFG["train_img_dir"],
        label_dir=CFG["train_lbl_dir"],
        transforms=transform,
    )
    val_dataset = CamVidDataset(
        image_dir=CFG["val_img_dir"],
        label_dir=CFG["val_lbl_dir"],
        transforms=transform,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val  : {len(val_dataset)} samples")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=True,    # batch가 incomplete하면 BN이 불안정해질 수 있음
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_segformer_b0(
        num_classes=CFG["num_classes"],
        embed_dim=CFG["embed_dim"],
        dropout=CFG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = CrossEntropyLoss(ignore_index=CFG["ignore_index"])

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # SegFormer 논문: AdamW, lr=6e-5, weight_decay=0.01
    # encoder / decoder lr 분리 (encoder는 더 낮은 lr — pretrain 보호)
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": CFG["lr"]},
            {"params": decoder_params, "lr": CFG["lr"] * 10},  # decoder는 10배
        ],
        weight_decay=CFG["weight_decay"],
    )

    # ── LR Scheduler: poly decay (SegFormer 논문 권장) ───────────────────────
    total_iters = CFG["epochs"] * len(train_loader)

    def poly_lr(iteration):
        return (1 - iteration / total_iters) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    best_miou = 0.0

    print("\n" + "=" * 60)
    print(f"Training: {CFG['exp_name']}  |  {CFG['epochs']} epochs")
    print("=" * 60)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        scheduler.step()

        # Validate
        val_loss, miou, per_class_iou = validate(
            model, val_loader, criterion, device,
            CFG["num_classes"], CFG["ignore_index"],
        )

        elapsed = time.time() - t0

        # 출력
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch [{epoch:3d}/{CFG['epochs']}] "
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
            CFG["save_dir"], CFG["exp_name"],
            is_best=is_best,
        )

    print("\n" + "=" * 60)
    print(f"Training complete. Best mIoU: {best_miou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()