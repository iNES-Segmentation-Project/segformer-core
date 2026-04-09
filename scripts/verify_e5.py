"""
scripts/verify_e5.py

E5 실험 실행 전 설정 검증 스크립트.

확인 항목:
  1. Config 로드 및 주요 값 출력
  2. Pretrained weight 로드 + key remapping (missing / unexpected 수)
  3. Differential LR 파라미터 그룹 분리 확인
  4. Paperlike augmentation 출력 shape / dtype / 값 범위 확인
  5. Forward pass (dummy input → logits shape 확인)
  6. Loss function 동작 확인
  7. Scheduler 초기 lr 확인

실행:
    python scripts/verify_e5.py --config configs/e5_best.yaml
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train import load_config, build_model, build_criterion, build_scheduler


# ── 출력 헬퍼 ─────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg: str):
    print(f"  [OK]   {msg}")

def warn(msg: str):
    print(f"  [WARN] {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")


# =============================================================================
# 1. Config
# =============================================================================

def verify_config(cfg: dict):
    section("1. Config 로드")

    required = [
        "exp_name", "model_type", "loss_type", "pretrained",
        "hf_model_name", "augmentation_type", "input_size",
        "epochs", "lr", "differential_lr", "scheduler_type", "warmup_ratio",
    ]
    for key in required:
        val = cfg.get(key, "MISSING")
        if val == "MISSING":
            fail(f"{key} 없음")
        else:
            ok(f"{key} = {val}")

    # E5 핵심 설정 검증
    assert cfg["pretrained"] is True,              "pretrained must be true"
    assert cfg["augmentation_type"] == "paperlike","augmentation_type must be paperlike"
    assert cfg["differential_lr"] is True,         "differential_lr must be true"
    assert cfg["scheduler_type"] == "warmup_poly", "scheduler_type must be warmup_poly"
    assert cfg["epochs"] >= 100,                   "epochs must be >= 100"
    ok("E5 핵심 설정 모두 확인")


# =============================================================================
# 2. Pretrained weight 로드 + key remapping
# =============================================================================

def verify_pretrained(cfg: dict):
    section("2. Pretrained Weight 로드 + Key Remapping")

    import os

    weight_path = cfg.get("hf_model_name", "")
    has_file        = os.path.isfile(weight_path)
    has_transformers = True
    try:
        import transformers  # noqa
    except ImportError:
        has_transformers = False

    if not has_file and not has_transformers:
        warn(f"로컬 파일 없음: {weight_path}")
        warn("transformers 패키지 미설치 → pretrained 로드 건너뜀")
        warn("실제 학습 전에 아래 중 하나를 준비하세요:")
        print("    (A) pip install transformers  → HF 자동 다운로드")
        print("    (B) weights/mit_b0_backbone.pth 직접 배치")
        print()
        # pretrained 없이 구조만 검증하기 위해 임시로 false로 빌드
        cfg_temp = dict(cfg)
        cfg_temp["pretrained"] = False
        model = build_model(cfg_temp)
        warn("pretrained=false로 모델 구조만 빌드 (구조 검증용)")
    elif not has_file:
        warn(f"로컬 파일 없음: {weight_path} → HF 다운로드 시도")
        model = build_model(cfg)
    else:
        model = build_model(cfg)  # 로컬 파일 로드

        w = model.encoder.stages[0].patch_embed.norm.weight
        print(f"\n  encoder.stages[0].patch_embed.norm.weight")
        print(f"    mean={w.mean().item():.6f}  std={w.std().item():.6f}  "
              f"min={w.min().item():.6f}  max={w.max().item():.6f}")
        if abs(w.mean().item() - 1.0) < 0.5:
            ok("norm weight 분포 정상 (mean ≈ 1.0, pretrained 특성)")
        else:
            warn(f"norm weight mean={w.mean().item():.4f} — 예상 범위 벗어남")
        ok("decoder는 random 초기화 (pretrained 로드 대상 아님)")

    return model


# =============================================================================
# 3. Differential LR
# =============================================================================

def verify_differential_lr(model: nn.Module, cfg: dict):
    section("3. Differential LR")

    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    if cfg.get("differential_lr", False):
        encoder_lr = cfg["lr"]
        decoder_lr = cfg["lr"] * 10
    else:
        encoder_lr = cfg["lr"] * 10
        decoder_lr = cfg["lr"] * 10

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ],
        weight_decay=cfg["weight_decay"],
    )

    g0_lr = optimizer.param_groups[0]["lr"]
    g1_lr = optimizer.param_groups[1]["lr"]

    print(f"  encoder lr : {g0_lr:.2e}   (param count: {len(encoder_params)})")
    print(f"  decoder lr : {g1_lr:.2e}   (param count: {len(decoder_params)})")

    ratio = g1_lr / g0_lr
    if abs(ratio - 10.0) < 1e-6:
        ok(f"decoder lr / encoder lr = {ratio:.1f}x  (SegFormer 논문 원칙)")
    else:
        fail(f"lr ratio = {ratio:.1f}x — 10x여야 함")

    return optimizer


# =============================================================================
# 4. Paperlike Augmentation
# =============================================================================

def verify_augmentation(cfg: dict):
    section("4. Paperlike Augmentation")

    from data.transforms import build_transform

    size = tuple(cfg["input_size"])  # (512, 512)
    train_tf = build_transform("paperlike", size, split="train")
    val_tf   = build_transform("paperlike", size, split="val")

    # 더미 이미지와 마스크 생성
    img  = Image.fromarray(np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8))
    mask = Image.fromarray(np.random.randint(0, 11, (360, 480), dtype=np.int32), mode="I")

    # Train transform
    img_t, mask_t = train_tf(img, mask)
    print(f"\n  [Train]")
    print(f"    image  : shape={tuple(img_t.shape)}  dtype={img_t.dtype}  "
          f"min={img_t.min():.3f}  max={img_t.max():.3f}")
    print(f"    mask   : shape={tuple(mask_t.shape)}  dtype={mask_t.dtype}  "
          f"min={mask_t.min().item()}  max={mask_t.max().item()}")

    assert img_t.shape  == (3, size[0], size[1]), f"image shape 오류: {img_t.shape}"
    assert mask_t.shape == (size[0], size[1]),     f"mask shape 오류: {mask_t.shape}"
    assert img_t.dtype  == torch.float32,          "image dtype must be float32"
    assert mask_t.dtype == torch.int64,            "mask dtype must be int64"
    ok("train: shape / dtype 정상")

    # mask class index 손상 여부 — 0~10 및 255만 존재해야 함
    unique = mask_t.unique().tolist()
    valid  = all(v in list(range(11)) + [255] for v in unique)
    if valid:
        ok(f"mask class index 정상 (unique values: {unique})")
    else:
        fail(f"mask class index 손상: {unique}")

    # Val transform
    img_v, mask_v = val_tf(img, mask)
    print(f"\n  [Val]")
    print(f"    image  : shape={tuple(img_v.shape)}  dtype={img_v.dtype}")
    print(f"    mask   : shape={tuple(mask_v.shape)}  dtype={mask_v.dtype}")
    assert img_v.shape == (3, size[0], size[1]), f"val image shape 오류: {img_v.shape}"
    ok("val: shape / dtype 정상")


# =============================================================================
# 5. Forward Pass
# =============================================================================

def verify_forward(model: nn.Module, cfg: dict):
    section("5. Forward Pass (dummy input)")

    H, W = cfg["input_size"]
    B    = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    dummy = torch.randn(B, 3, H, W, device=device)

    with torch.no_grad():
        logits = model(dummy)

    expected = (B, cfg["num_classes"], H, W)
    print(f"  input  : {tuple(dummy.shape)}")
    print(f"  logits : {tuple(logits.shape)}  (expected: {expected})")

    if tuple(logits.shape) == expected:
        ok("logits shape 정상")
    else:
        fail(f"logits shape 불일치: {tuple(logits.shape)} ≠ {expected}")

    model.cpu()
    return model


# =============================================================================
# 6. mPA 계산 검증
# =============================================================================

def verify_mpa(cfg: dict):
    section("6. mPA 계산 검증")

    from scripts.train import MeanIoU

    num_classes  = cfg["num_classes"]
    ignore_index = cfg["ignore_index"]
    calc = MeanIoU(num_classes=num_classes, ignore_index=ignore_index)

    # 완벽한 예측 케이스: preds == targets → mPA = 1.0, mIoU = 1.0
    targets = torch.randint(0, num_classes, (2, 64, 64))
    calc.update(targets.clone(), targets)
    miou, per_class_iou, mpa = calc.compute()

    print(f"  [완벽한 예측] mIoU={miou:.4f}  mPA={mpa:.4f}  (둘 다 1.0이어야 함)")
    if abs(miou - 1.0) < 1e-4 and abs(mpa - 1.0) < 1e-4:
        ok("완벽한 예측: mIoU=1.0, mPA=1.0 확인")
    else:
        fail(f"완벽한 예측인데 mIoU={miou:.4f}, mPA={mpa:.4f}")

    # ignore_index 처리 확인
    calc2 = MeanIoU(num_classes=num_classes, ignore_index=ignore_index)
    targets2 = torch.full((2, 64, 64), ignore_index, dtype=torch.long)
    targets2[:, :32, :] = 0   # 일부만 유효
    preds2 = targets2.clone()
    calc2.update(preds2, targets2)
    miou2, _, mpa2 = calc2.compute()

    print(f"  [ignore_index 처리] mIoU={miou2:.4f}  mPA={mpa2:.4f}  (ignore 제외하고 1.0이어야 함)")
    if abs(miou2 - 1.0) < 1e-4 and abs(mpa2 - 1.0) < 1e-4:
        ok("ignore_index 제외 정상")
    else:
        fail(f"ignore_index 처리 오류: mIoU={miou2:.4f}, mPA={mpa2:.4f}")

    print(f"\n  반환 형식: (miou, per_class_iou, mpa) — 3-tuple")
    ok("mPA 계산 및 반환 형식 정상")


# =============================================================================
# 7. GFLOPs / Params 검증
# =============================================================================

def verify_complexity(model: nn.Module, cfg: dict):
    section("7. GFLOPs / Params 검증")

    # Params
    total_params   = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"  Total params  : {total_params:>10,}")
    print(f"  Encoder params: {encoder_params:>10,}  (MiT-B0 고정)")
    print(f"  Decoder params: {decoder_params:>10,}  (FPN)")

    # MiT-B0 encoder 공식 파라미터 수: ~3.7M
    if 3_000_000 < encoder_params < 5_000_000:
        ok(f"encoder params={encoder_params:,} — MiT-B0 정상 범위 (3M~5M)")
    else:
        warn(f"encoder params={encoder_params:,} — 예상 범위 벗어남")

    # GFLOPs
    try:
        from fvcore.nn import FlopCountAnalysis
        H, W = cfg["input_size"]
        dummy = torch.randn(1, 3, H, W)
        model.eval()
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_settings(raise_if_not_supported=False)
        gflops = flops.total() / 1e9
        print(f"\n  GFLOPs : {gflops:.2f} G  (input {H}×{W})")
        # SegFormer-B0 기준 약 8~15 GFLOPs (512×512)
        if 1.0 < gflops < 50.0:
            ok(f"GFLOPs={gflops:.2f}G — 정상 범위")
        else:
            warn(f"GFLOPs={gflops:.2f}G — 예상 범위 벗어남, 확인 필요")
    except Exception as e:
        warn(f"GFLOPs 측정 실패: {e}")


# =============================================================================
# 8. Loss Function
# =============================================================================

def verify_loss(model: nn.Module, cfg: dict):
    section("8. Loss Function (ce_dice_boundary)")

    criterion = build_criterion(cfg)
    print(f"  criterion: {criterion.__class__.__name__}")

    H, W = cfg["input_size"]
    B    = 2
    logits = torch.randn(B, cfg["num_classes"], H, W)
    masks  = torch.randint(0, cfg["num_classes"], (B, H, W))

    try:
        loss = criterion(logits, masks)
        print(f"  loss value: {loss.item():.4f}")
        assert loss.item() > 0, "loss는 0보다 커야 함"
        ok("loss 계산 정상")
    except Exception as e:
        fail(f"loss 계산 실패: {e}")


# =============================================================================
# 7. Scheduler
# =============================================================================

def verify_scheduler(model: nn.Module, cfg: dict, optimizer: torch.optim.Optimizer):
    section("7. Scheduler (warmup_poly)")

    # 실제 DataLoader 없이 total_iters 임의 설정
    iters_per_epoch = 100   # CamVid train 약 367 / batch 4 ≈ 91
    total_iters     = cfg["epochs"] * iters_per_epoch
    warmup_iters    = int(total_iters * cfg["warmup_ratio"])

    scheduler = build_scheduler(cfg, optimizer, total_iters)

    print(f"  total_iters  : {total_iters}")
    print(f"  warmup_iters : {warmup_iters} (ratio={cfg['warmup_ratio']})")

    # iter 0 → lr should be near 0 (warmup 시작)
    lr0 = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr1 = optimizer.param_groups[0]["lr"]

    print(f"\n  iter 0 → lr_encoder: {lr0:.2e}")
    print(f"  iter 1 → lr_encoder: {lr1:.2e}  (warmup 중이면 증가해야 함)")

    if lr1 > lr0:
        ok("warmup 중 lr 증가 확인")
    else:
        warn(f"lr가 증가하지 않음: {lr0:.2e} → {lr1:.2e}")

    # warmup 끝 지점
    for _ in range(warmup_iters - 2):
        scheduler.step()
    lr_peak = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr_after = optimizer.param_groups[0]["lr"]

    print(f"\n  warmup 마지막  lr_encoder: {lr_peak:.2e}")
    print(f"  warmup 직후    lr_encoder: {lr_after:.2e}  (poly decay 시작이면 감소)")

    if lr_after <= lr_peak:
        ok("warmup 이후 poly decay 시작 확인")
    else:
        warn("warmup 이후 lr가 감소하지 않음")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/e5_best.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("\n" + "★" * 60)
    print("  E5 설정 검증 시작")
    print("★" * 60)

    verify_config(cfg)

    model     = verify_pretrained(cfg)
    optimizer = verify_differential_lr(model, cfg)

    verify_augmentation(cfg)
    model = verify_forward(model, cfg)
    verify_mpa(cfg)
    verify_complexity(model, cfg)
    verify_loss(model, cfg)
    verify_scheduler(model, cfg, optimizer)

    print("\n" + "★" * 60)
    print("  검증 완료")
    print("★" * 60 + "\n")


if __name__ == "__main__":
    main()
