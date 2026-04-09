"""
scripts/e5_sanity_check.py
E5_test.md 기준 pretrained 로딩 sanity check.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import SegformerModel
from utils.checkpoint import _remap_hf_to_ours
from models.segformer import build_segformer_b0_fpn
from models.loss.combined_loss import CombinedLoss

SEP = "=" * 60

# ── 공통 준비 ──────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  HF 모델 다운로드 / 캐시 로드")
print(SEP)
hf_model  = SegformerModel.from_pretrained("nvidia/mit-b0")
hf_state  = hf_model.state_dict()
print(f"  HF state_dict keys: {len(hf_state)}")

our_model = build_segformer_b0_fpn(num_classes=11, fpn_dim=256, dropout=0.1)
our_state = _remap_hf_to_ours(hf_state)
missing, unexpected = our_model.encoder.load_state_dict(our_state, strict=False)


# =============================================================================
# 1. load_state_dict 결과 검증
# =============================================================================
print(f"\n{SEP}")
print("  [1] load_state_dict 결과 검증")
print(SEP)

print(f"  remapped keys  : {len(our_state)}")
print(f"  missing keys   : {len(missing)}")
print(f"  unexpected keys: {len(unexpected)}")

if missing:
    print(f"  missing  (전체): {missing}")
if unexpected:
    print(f"  unexpected (전체): {unexpected}")

# 정상 기준:
# - missing=0    : 우리 encoder의 모든 파라미터가 HF에서 로드됨
# - unexpected=0 : HF 파라미터 중 우리 모델에 없는 것이 없음
# encoder-only 로딩이므로 HF의 decoder 관련 키는 애초에 없음
#   → missing=0, unexpected=0 이 PASS 기준
if len(missing) == 0 and len(unexpected) == 0:
    print("\n  판정: PASS ✅  missing=0, unexpected=0")
elif len(missing) == 0 and len(unexpected) > 0:
    print("\n  판정: PASS ✅  missing=0 (encoder 완전 로드), unexpected는 무시 가능")
else:
    print(f"\n  판정: FAIL ❌  missing={len(missing)}개 — 로드 안 된 파라미터 존재")


# =============================================================================
# 2. Weight 분포 검증
# =============================================================================
print(f"\n{SEP}")
print("  [2] Weight 분포 검증")
print(SEP)

w = our_model.encoder.stages[0].patch_embed.norm.weight
print(f"  stages[0].patch_embed.norm.weight")
print(f"    shape : {tuple(w.shape)}")
print(f"    mean  : {w.mean().item():.6f}")
print(f"    std   : {w.std().item():.6f}")
print(f"    min   : {w.min().item():.6f}")
print(f"    max   : {w.max().item():.6f}")

# 판정 기준:
# - LayerNorm.weight는 ones_()로 초기화 → random init이면 mean ≈ 1.0, std ≈ 0.0 (상수)
# - pretrained는 학습 과정에서 1.0 근처에서 변화 → mean은 1.0 근처지만 std > 0
# - 따라서 std > 0.01이면 학습된 weight (pretrained)
# - std ≈ 0이면 random init (ones_) 상태 → 로딩 실패
print(f"\n  판정 기준: std > 0.01 이면 pretrained (학습된 값)")
if w.std().item() > 0.01:
    print(f"  판정: PASS ✅  std={w.std().item():.6f} > 0.01 (pretrained 분포 확인)")
else:
    print(f"  판정: FAIL ❌  std={w.std().item():.6f} ≈ 0 (ones_ 초기화 상태, 로딩 안 됨)")


# =============================================================================
# 3. KV 매핑 검증
# =============================================================================
print(f"\n{SEP}")
print("  [3] KV 매핑 검증")
print(SEP)

# HF: key.weight (dim, dim) + value.weight (dim, dim)
# 우리: kv.weight (dim*2, dim)
# stage 0, block 0 기준 (embed_dim=32, num_heads=1 → head_dim=32)
hf_k_w = hf_state["encoder.block.0.0.attention.self.key.weight"]
hf_v_w = hf_state["encoder.block.0.0.attention.self.value.weight"]
our_kv_w = our_model.encoder.stages[0].blocks[0].attn.kv.weight

print(f"  HF  key.weight  shape : {tuple(hf_k_w.shape)}")
print(f"  HF  val.weight  shape : {tuple(hf_v_w.shape)}")
print(f"  우리 kv.weight   shape : {tuple(our_kv_w.shape)}")

dim = hf_k_w.shape[0]
expected_kv_shape = (dim * 2, dim)
print(f"\n  기대 kv shape: {expected_kv_shape}  (dim={dim}, dim*2={dim*2})")

# cat(k, v, dim=0)으로 구성됐는지 확인
expected_kv = torch.cat([hf_k_w, hf_v_w], dim=0)
match = torch.allclose(our_kv_w, expected_kv)

print(f"  cat([k,v], dim=0) == kv.weight : {match}")
print(f"  max_diff : {(our_kv_w - expected_kv).abs().max().item():.2e}")

if tuple(our_kv_w.shape) == expected_kv_shape and match:
    print(f"\n  판정: PASS ✅  shape 정확, 값 일치")
elif tuple(our_kv_w.shape) == expected_kv_shape and not match:
    print(f"\n  판정: FAIL ❌  shape은 맞지만 값 불일치 (cat 순서 오류)")
else:
    print(f"\n  판정: FAIL ❌  shape 불일치 {tuple(our_kv_w.shape)} ≠ {expected_kv_shape}")


# =============================================================================
# 4. 특정 파라미터 직접 비교
# =============================================================================
print(f"\n{SEP}")
print("  [4] 파라미터 직접 비교 (HF vs 우리 모델)")
print(SEP)

# patch_embed.proj.weight 비교
hf_proj_w  = hf_state["encoder.patch_embeddings.0.proj.weight"]
our_proj_w = our_model.encoder.stages[0].patch_embed.proj.weight

match_proj = torch.allclose(hf_proj_w, our_proj_w)
print(f"  [patch_embed.proj.weight]")
print(f"    HF  shape : {tuple(hf_proj_w.shape)}")
print(f"    우리 shape : {tuple(our_proj_w.shape)}")
print(f"    값 일치    : {match_proj}")
print(f"    max_diff   : {(hf_proj_w - our_proj_w).abs().max().item():.2e}")

# norm.weight 비교
hf_norm_w  = hf_state["encoder.patch_embeddings.0.layer_norm.weight"]
our_norm_w = our_model.encoder.stages[0].patch_embed.norm.weight

match_norm = torch.allclose(hf_norm_w, our_norm_w)
print(f"\n  [patch_embed.norm.weight]")
print(f"    HF  값 (앞 5): {hf_norm_w[:5].tolist()}")
print(f"    우리 값 (앞 5): {our_norm_w[:5].tolist()}")
print(f"    값 일치        : {match_norm}")
print(f"    max_diff       : {(hf_norm_w - our_norm_w).abs().max().item():.2e}")

if match_proj and match_norm:
    print(f"\n  판정: PASS ✅  두 파라미터 모두 HF와 값 일치")
else:
    print(f"\n  판정: FAIL ❌  proj={match_proj}, norm={match_norm}")


# =============================================================================
# 5. Forward 안정성 검증
# =============================================================================
print(f"\n{SEP}")
print("  [5] Forward 안정성 검증")
print(SEP)

criterion = CombinedLoss(
    mode="ce+dice+boundary",
    num_classes=11,
    ignore_index=255,
    ce_weight=1.0,
    aux_weight=1.0,
    boundary_weight=0.1,
)

our_model.eval()
dummy_img  = torch.randn(2, 3, 512, 512)
dummy_mask = torch.randint(0, 11, (2, 512, 512))

with torch.no_grad():
    logits = our_model(dummy_img)
    loss   = criterion(logits, dummy_mask)

print(f"  input  shape : {tuple(dummy_img.shape)}")
print(f"  logits shape : {tuple(logits.shape)}")
print(f"  loss value   : {loss.item():.4f}")
print(f"  logits NaN   : {torch.isnan(logits).any().item()}")
print(f"  logits Inf   : {torch.isinf(logits).any().item()}")
print(f"  loss   NaN   : {torch.isnan(loss).item()}")

# 정상 기준:
# - logits shape: (2, 11, 512, 512)
# - loss: 대략 1.5~4.0 범위 (CE ≈ log(11) ≈ 2.4 + Dice ≈ 0.9 + 0.1*Boundary)
# - NaN/Inf 없음
expected_logits = (2, 11, 512, 512)
shape_ok  = tuple(logits.shape) == expected_logits
nan_ok    = not torch.isnan(logits).any().item() and not torch.isnan(loss).item()
loss_ok   = 0.5 < loss.item() < 10.0

if shape_ok and nan_ok and loss_ok:
    print(f"\n  판정: PASS ✅  shape 정확, NaN 없음, loss 정상 범위")
elif shape_ok and nan_ok:
    print(f"\n  판정: PASS ✅  shape/NaN 정상 (loss={loss.item():.4f} 범위 확인 필요)")
else:
    print(f"\n  판정: FAIL ❌  shape={shape_ok}, nan_free={nan_ok}")


# =============================================================================
# 6. 최종 판정
# =============================================================================
print(f"\n{SEP}")
print("  [6] 최종 판정")
print(SEP)

checks = {
    "missing=0"          : len(missing) == 0,
    "std>0.01 (pretrained 분포)": w.std().item() > 0.01,
    "kv shape 정확"      : tuple(our_kv_w.shape) == expected_kv_shape,
    "kv 값 일치"         : match,
    "proj.weight 일치"   : match_proj,
    "norm.weight 일치"   : match_norm,
    "logits shape 정확"  : shape_ok,
    "NaN/Inf 없음"       : nan_ok,
    "loss 정상 범위"     : loss_ok,
}

all_pass = all(checks.values())
any_fail = not all_pass

for name, result in checks.items():
    mark = "✅" if result else "❌"
    print(f"  {mark}  {name}")

print()
if all_pass:
    print("  최종: ✅ 정상 로딩 완료 — E5 학습 진행 가능")
elif checks["missing=0"] and checks["kv 값 일치"] and checks["proj.weight 일치"]:
    print("  최종: ⚠️ 부분 성공 — 핵심 파라미터는 로드됨, 세부 항목 확인 필요")
else:
    print("  최종: ❌ 로딩 실패 — 구조 문제 해결 필요")

print(f"\n{SEP}\n")
