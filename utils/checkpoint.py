"""
utils/checkpoint.py

MiT-B0 ImageNet1K pretrained weight 로드 유틸리티.

HuggingFace nvidia/mit-b0 실제 key 구조 (확인 완료):
    encoder.patch_embeddings.{i}.proj.weight
    encoder.patch_embeddings.{i}.layer_norm.weight
    encoder.block.{i}.{j}.layer_norm_1.weight
    encoder.block.{i}.{j}.attention.self.query.weight
    encoder.block.{i}.{j}.attention.self.key.weight
    encoder.block.{i}.{j}.attention.self.value.weight
    encoder.block.{i}.{j}.attention.self.sr.weight
    encoder.block.{i}.{j}.attention.self.layer_norm.weight
    encoder.block.{i}.{j}.attention.output.dense.weight
    encoder.block.{i}.{j}.layer_norm_2.weight
    encoder.block.{i}.{j}.mlp.dense1.weight
    encoder.block.{i}.{j}.mlp.dense2.weight
    encoder.block.{i}.{j}.mlp.dwconv.dwconv.weight
    encoder.layer_norm.{i}.weight

이 프로젝트 key 구조:
    stages.{i}.patch_embed.proj.weight
    stages.{i}.patch_embed.norm.weight
    stages.{i}.blocks.{j}.norm1.weight
    stages.{i}.blocks.{j}.attn.q.weight
    stages.{i}.blocks.{j}.attn.kv.weight        ← k + v를 cat해서 합침
    stages.{i}.blocks.{j}.attn.sr.weight
    stages.{i}.blocks.{j}.attn.sr_norm.weight
    stages.{i}.blocks.{j}.attn.proj.weight
    stages.{i}.blocks.{j}.norm2.weight
    stages.{i}.blocks.{j}.ffn.fc1.weight
    stages.{i}.blocks.{j}.ffn.fc2.weight
    stages.{i}.blocks.{j}.ffn.dw_conv.weight
    stages.{i}.norm.weight
"""

import re
import torch
import torch.nn as nn
from typing import Dict


# =============================================================================
# ── HuggingFace → 프로젝트 key remapping ─────────────────────────────────────
# =============================================================================

_SUFFIX_MAP = {
    # patch embed
    "proj.weight":                   "patch_embed.proj.weight",
    "proj.bias":                     "patch_embed.proj.bias",
    "layer_norm.weight":             "patch_embed.norm.weight",
    "layer_norm.bias":               "patch_embed.norm.bias",
    # block: layer norm
    "layer_norm_1.weight":           "norm1.weight",
    "layer_norm_1.bias":             "norm1.bias",
    "layer_norm_2.weight":           "norm2.weight",
    "layer_norm_2.bias":             "norm2.bias",
    # block: attention - query
    "attention.self.query.weight":   "attn.q.weight",
    "attention.self.query.bias":     "attn.q.bias",
    # block: attention - spatial reduction
    "attention.self.sr.weight":      "attn.sr.weight",
    "attention.self.sr.bias":        "attn.sr.bias",
    # block: attention - output projection
    "attention.output.dense.weight": "attn.proj.weight",
    "attention.output.dense.bias":   "attn.proj.bias",
    # block: ffn
    "mlp.dense1.weight":             "ffn.fc1.weight",
    "mlp.dense1.bias":               "ffn.fc1.bias",
    "mlp.dense2.weight":             "ffn.fc2.weight",
    "mlp.dense2.bias":               "ffn.fc2.bias",
    "mlp.dwconv.dwconv.weight":      "ffn.dw_conv.weight",
    "mlp.dwconv.dwconv.bias":        "ffn.dw_conv.bias",
}

_RE_PATCH = re.compile(r"^encoder\.patch_embeddings\.(\d+)\.(.+)$")
_RE_BLOCK = re.compile(r"^encoder\.block\.(\d+)\.(\d+)\.(.+)$")
_RE_NORM  = re.compile(r"^encoder\.layer_norm\.(\d+)\.(.+)$")


def _remap_hf_to_ours(hf_state: Dict) -> Dict:
    new_state = {}
    skipped   = []

    # k, v weight/bias를 따로 모아 나중에 cat으로 합침
    # { "stages.{i}.blocks.{j}": {"k_w": tensor, "v_w": tensor, ...} }
    kv_buffer = {}

    for hf_key, v in hf_state.items():

        # ── patch_embeddings.{i}.* ────────────────────────────────────────────
        m = _RE_PATCH.match(hf_key)
        if m:
            i, suffix = m.group(1), m.group(2)
            if suffix in _SUFFIX_MAP:
                new_state[f"stages.{i}.{_SUFFIX_MAP[suffix]}"] = v
            else:
                skipped.append(hf_key)
            continue

        # ── block.{i}.{j}.* ──────────────────────────────────────────────────
        m = _RE_BLOCK.match(hf_key)
        if m:
            i, j, suffix = m.group(1), m.group(2), m.group(3)
            block_prefix = f"stages.{i}.blocks.{j}"

            # k, v는 kv_buffer에 수집 → 나중에 cat
            if suffix == "attention.self.key.weight":
                kv_buffer.setdefault(block_prefix, {})["k_w"] = v
                continue
            if suffix == "attention.self.key.bias":
                kv_buffer.setdefault(block_prefix, {})["k_b"] = v
                continue
            if suffix == "attention.self.value.weight":
                kv_buffer.setdefault(block_prefix, {})["v_w"] = v
                continue
            if suffix == "attention.self.value.bias":
                kv_buffer.setdefault(block_prefix, {})["v_b"] = v
                continue

            # attention.self.layer_norm → attn.sr_norm
            if suffix == "attention.self.layer_norm.weight":
                new_state[f"{block_prefix}.attn.sr_norm.weight"] = v
                continue
            if suffix == "attention.self.layer_norm.bias":
                new_state[f"{block_prefix}.attn.sr_norm.bias"] = v
                continue

            if suffix in _SUFFIX_MAP:
                new_state[f"{block_prefix}.{_SUFFIX_MAP[suffix]}"] = v
            else:
                skipped.append(hf_key)
            continue

        # ── layer_norm.{i}.* (stage-level norm) ──────────────────────────────
        m = _RE_NORM.match(hf_key)
        if m:
            i, suffix = m.group(1), m.group(2)
            new_state[f"stages.{i}.norm.{suffix}"] = v
            continue

        skipped.append(hf_key)

    # ── kv_buffer: k, v를 cat해서 kv Linear weight로 합치기 ──────────────────
    # self.kv = nn.Linear(dim, dim*2)  → weight shape: (dim*2, dim)
    # cat([k_w, v_w], dim=0): (dim, dim) + (dim, dim) → (dim*2, dim) ✓
    for block_prefix, buf in kv_buffer.items():
        if "k_w" in buf and "v_w" in buf:
            new_state[f"{block_prefix}.attn.kv.weight"] = torch.cat(
                [buf["k_w"], buf["v_w"]], dim=0
            )
        if "k_b" in buf and "v_b" in buf:
            new_state[f"{block_prefix}.attn.kv.bias"] = torch.cat(
                [buf["k_b"], buf["v_b"]], dim=0
            )

    if skipped:
        print(f"[Checkpoint] Remapping skipped ({len(skipped)} keys): {skipped[:5]}")

    return new_state


# =============================================================================
# ── Pretrained encoder 로드 ───────────────────────────────────────────────────
# =============================================================================

def load_pretrained_encoder(
    model: nn.Module,
    hf_model_name: str = "nvidia/mit-b0",
    strict: bool = False,
) -> nn.Module:
    """
    HuggingFace pretrained weight를 model.encoder에 로드한다.

    checkpoint resume 시에는 호출하지 말 것.
    train.py에서 start_epoch == 0일 때만 호출한다.

    Args:
        model         : SegFormer 전체 모델
        hf_model_name : HuggingFace model ID 또는 로컬 .pth 경로
                        - "nvidia/mit-b0"              → HF에서 다운로드 (캐시 활용)
                        - "weights/mit_b0_backbone.pth" → 로컬 파일 직접 로드
        strict        : key 불일치 시 에러 여부 (default: False)

    Returns:
        model : pretrained encoder weight가 로드된 모델
    """
    import os
    if os.path.isfile(hf_model_name):
        # 로컬 .pth 파일 — torch.save(hf_model.state_dict(), path) 형식
        print(f"[Checkpoint] Loading pretrained encoder from local file: {hf_model_name}")
        hf_state = torch.load(hf_model_name, map_location="cpu")
    else:
        # HuggingFace model ID
        from transformers import SegformerModel
        print(f"[Checkpoint] Loading pretrained encoder from HuggingFace: {hf_model_name}")
        hf_state = SegformerModel.from_pretrained(hf_model_name).state_dict()

    our_state = _remap_hf_to_ours(hf_state)

    missing, unexpected = model.encoder.load_state_dict(our_state, strict=strict)

    print(f"[Checkpoint] Remapped keys  : {len(our_state)}")
    print(f"[Checkpoint] Missing keys   : {len(missing)}")
    print(f"[Checkpoint] Unexpected keys: {len(unexpected)}")

    if missing:
        print(f"  Missing   (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")

    if len(missing) == 0:
        print("[Checkpoint] Pretrained encoder loaded successfully.")
    else:
        print("[Checkpoint] WARNING: some keys were not loaded. "
              "Check encoder key names against _SUFFIX_MAP.")

    return model
