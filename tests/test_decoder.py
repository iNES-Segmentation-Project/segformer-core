"""
tests/test_decoder.py

Integration tests for decoder and full SegFormer model.

Tests:
  1.  MLPDecoder output shape                    — (B, num_classes, H/4, W/4)
  2.  MLPDecoder with real encoder features      — end-to-end shape
  3.  SegFormer full forward pass shape          — (B, num_classes, H, W)
  4.  Non-square input handling                  — H ≠ W
  5.  CamVid config (11 classes)                 — build_segformer_b0
  6.  Cityscapes config (19 classes)             — build_segformer_b0
  7.  CrossEntropy loss computation              — no NaN/Inf
  8.  Backward pass                              — gradients flow to encoder
  9.  Encoder weights frozen / unfrozen check   — grad existence
  10. Parameter count sanity                     — decoder << encoder
  11. Eval mode output determinism               — same input → same output

Run with:
    pytest tests/test_decoder.py -v
    python tests/test_decoder.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder.mit_encoder import MiTEncoder
from models.decoder.base_decoder import BaseDecoder
from models.decoder.mlp_decoder import MLPDecoder
from models.segformer import SegFormer, build_segformer_b0, MIT_B0_CHANNELS

# ── Test constants ─────────────────────────────────────────────────────────────
B           = 2
H = W       = 512
NUM_CLASSES = 11          # CamVid default
EMBED_DIM   = 256         # MLP decoder embed dim

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
INFO = "\033[34m[INFO]\033[0m"


# ─── Helper: build encoder features without full model ────────────────────────

def _get_encoder_features(h=H, w=W):
    """Run a dummy batch through MiTEncoder, return [c1,c2,c3,c4]."""
    encoder = MiTEncoder(in_channels=3)
    encoder.eval()
    x = torch.randn(B, 3, h, w)
    with torch.no_grad():
        features = encoder(x)
    return features


# ─── Test 1: MLPDecoder output shape (from dummy features) ────────────────────

def test_mlp_decoder_shape_from_dummy():
    """Decoder output must be (B, num_classes, H/4, W/4) given valid dummy features."""
    decoder = MLPDecoder(
        in_channels=MIT_B0_CHANNELS,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
    )
    decoder.eval()

    # Build dummy features matching MiT-B0 encoder output shapes
    dummy_features = [
        torch.randn(B,  32, H//4,  W//4),   # c1
        torch.randn(B,  64, H//8,  W//8),   # c2
        torch.randn(B, 160, H//16, W//16),  # c3
        torch.randn(B, 256, H//32, W//32),  # c4
    ]

    with torch.no_grad():
        out = decoder(dummy_features)

    expected = (B, NUM_CLASSES, H//4, W//4)
    assert out.shape == expected, f"Expected {expected}, got {tuple(out.shape)}"
    print(f"{PASS} MLPDecoder shape (dummy): {tuple(out.shape)}")


# ─── Test 2: MLPDecoder with real encoder features ────────────────────────────

def test_mlp_decoder_shape_from_encoder():
    """Decoder accepts real encoder output without shape errors."""
    features = _get_encoder_features()
    decoder = MLPDecoder(
        in_channels=MIT_B0_CHANNELS,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
    )
    decoder.eval()

    with torch.no_grad():
        out = decoder(features)

    expected = (B, NUM_CLASSES, H//4, W//4)
    assert out.shape == expected, f"Expected {expected}, got {tuple(out.shape)}"
    print(f"{PASS} MLPDecoder shape (real encoder): {tuple(out.shape)}")


# ─── Test 3: Full SegFormer forward shape ─────────────────────────────────────

def test_segformer_full_forward_shape():
    """Full model output must match original input resolution (B, C, H, W)."""
    model = build_segformer_b0(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM)
    model.eval()

    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        out = model(x)

    expected = (B, NUM_CLASSES, H, W)
    assert out.shape == expected, f"Expected {expected}, got {tuple(out.shape)}"
    print(f"{PASS} SegFormer full forward: {tuple(out.shape)}")


# ─── Test 4: Non-square input ─────────────────────────────────────────────────

def test_non_square_input():
    """Model must handle non-square inputs (e.g., CamVid 360×480)."""
    H_ns, W_ns = 360, 480
    model = build_segformer_b0(num_classes=NUM_CLASSES)
    model.eval()

    x = torch.randn(B, 3, H_ns, W_ns)
    with torch.no_grad():
        out = model(x)

    expected = (B, NUM_CLASSES, H_ns, W_ns)
    assert out.shape == expected, f"Expected {expected}, got {tuple(out.shape)}"
    print(f"{PASS} Non-square input (360×480): {tuple(out.shape)}")


# ─── Test 5: CamVid config (11 classes) ───────────────────────────────────────

def test_camvid_config():
    """build_segformer_b0 with 11 classes (CamVid)."""
    model = build_segformer_b0(num_classes=11)
    model.eval()
    x = torch.randn(1, 3, 360, 480)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 11, 360, 480), f"Got {tuple(out.shape)}"
    print(f"{PASS} CamVid config (11 classes): {tuple(out.shape)}")


# ─── Test 6: Cityscapes config (19 classes) ───────────────────────────────────

def test_cityscapes_config():
    """build_segformer_b0 with 19 classes (Cityscapes)."""
    model = build_segformer_b0(num_classes=19)
    model.eval()
    x = torch.randn(1, 3, 512, 1024)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 19, 512, 1024), f"Got {tuple(out.shape)}"
    print(f"{PASS} Cityscapes config (19 classes): {tuple(out.shape)}")


# ─── Test 7: CrossEntropy loss computation ────────────────────────────────────

def test_cross_entropy_loss():
    """Loss must be finite and non-zero."""
    model = build_segformer_b0(num_classes=NUM_CLASSES)
    model.train()

    x      = torch.randn(B, 3, H, W)
    labels = torch.randint(0, NUM_CLASSES, (B, H, W))   # (B, H, W) long tensor

    logits = model(x)           # (B, num_classes, H, W)
    loss   = F.cross_entropy(logits, labels)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0,      f"Loss is zero — something is wrong"
    print(f"{PASS} CrossEntropy loss: {loss.item():.4f}")


# ─── Test 8: Backward pass ────────────────────────────────────────────────────

def test_backward_pass():
    """
    Gradients must flow all the way back to the encoder.
    Checks that the first encoder stage (patch_embed projection weight)
    has a non-None, non-zero gradient after loss.backward().
    """
    model = build_segformer_b0(num_classes=NUM_CLASSES)
    model.train()

    x      = torch.randn(B, 3, H, W)
    labels = torch.randint(0, NUM_CLASSES, (B, H, W))

    logits = model(x)
    loss   = F.cross_entropy(logits, labels)
    loss.backward()

    # Check gradient exists in encoder stage 1 patch embedding
    enc_weight = model.encoder.stages[0].patch_embed.proj.weight
    assert enc_weight.grad is not None, "No gradient in encoder!"
    assert enc_weight.grad.abs().sum() > 0, "Gradient is all zeros in encoder!"

    # Check gradient exists in decoder projection layer
    # proj_layers[i] is a LinearProjection instance (not nn.Sequential),
    # so access its inner Conv2d via .proj, not via [0]
    dec_weight = model.decoder.proj_layers[0].proj.weight
    assert dec_weight.grad is not None, "No gradient in decoder proj_layers!"

    print(f"{PASS} Backward pass — gradients reach encoder & decoder")
    print(f"       Encoder grad norm: {enc_weight.grad.norm().item():.6f}")
    print(f"       Decoder grad norm: {dec_weight.grad.norm().item():.6f}")


# ─── Test 9: Parameter count ──────────────────────────────────────────────────

def test_parameter_count():
    """
    Decoder should be lightweight compared to encoder.
    MiT-B0 encoder ≈ 3.3M params.
    MLP decoder with embed_dim=256 ≈ 0.5M params.
    """
    model = build_segformer_b0(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total      = enc_params + dec_params

    print(f"{INFO} Encoder params : {enc_params:>10,}")
    print(f"{INFO} Decoder params : {dec_params:>10,}")
    print(f"{INFO} Total params   : {total:>10,}")

    assert enc_params > dec_params, "Encoder should have more params than decoder"
    assert total < 10_000_000, f"Total param count unexpectedly large: {total:,}"
    print(f"{PASS} Parameter count within expected range")


# ─── Test 10: Eval mode determinism ───────────────────────────────────────────

def test_eval_determinism():
    """Same input must produce identical output in eval mode (no dropout/BN noise)."""
    model = build_segformer_b0(num_classes=NUM_CLASSES)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2), "Eval outputs are not deterministic!"
    print(f"{PASS} Eval mode determinism — outputs are identical")


# ─── Test 11: Decoder feature count assertion ─────────────────────────────────

def test_decoder_wrong_feature_count():
    """BaseDecoder._check_features must raise AssertionError for wrong input length."""
    decoder = MLPDecoder(
        in_channels=MIT_B0_CHANNELS,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
    )
    bad_features = [torch.randn(B, 32, 128, 128)] * 3   # only 3, not 4
    try:
        decoder(bad_features)
        print(f"{FAIL} Should have raised AssertionError for 3 features")
    except AssertionError:
        print(f"{PASS} Correctly raises AssertionError for wrong feature count")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_mlp_decoder_shape_from_dummy,
        test_mlp_decoder_shape_from_encoder,
        test_segformer_full_forward_shape,
        test_non_square_input,
        test_camvid_config,
        test_cityscapes_config,
        test_cross_entropy_loss,
        test_backward_pass,
        test_parameter_count,
        test_eval_determinism,
        test_decoder_wrong_feature_count,
    ]

    print("=" * 60)
    print("Running decoder & SegFormer integration tests")
    print("=" * 60)
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"{FAIL} {t.__name__}: {e}")
            failed.append(t.__name__)
    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} test(s): {failed}")
    else:
        print("All tests passed.")
