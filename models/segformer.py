"""
segformer.py

Full SegFormer model: encoder + decoder assembly.

Responsibilities:
  1. Run MiTEncoder to get multi-scale features [c1, c2, c3, c4]
  2. Run decoder to get logits at (H/4, W/4) resolution
  3. Upsample logits back to original input resolution (H, W)

Design notes:
  - Encoder and decoder are fully decoupled; swapping either requires
    no changes to this file beyond the constructor arguments.
  - Final interpolation is done HERE, not inside the decoder, so the
    decoder always outputs at a fixed (H/4, W/4) scale regardless of
    which decoder variant is used.
  - align_corners=False is standard for segmentation inference.

Supported configurations (experiment matrix):
  E0: MiTEncoder + MLPDecoder + CrossEntropyLoss   ← this file targets E0
  E1: MiTEncoder + FPNDecoder + CrossEntropyLoss
  E2–E5: swap decoder/loss via constructor args
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Type

from .encoder.mit_encoder import MiTEncoder
from .decoder.base_decoder import BaseDecoder
from .decoder.mlp_decoder import MLPDecoder


# ── SegFormer-B0 encoder output channels (fixed, must not change) ─────────────
MIT_B0_CHANNELS = [32, 64, 160, 256]


class SegFormer(nn.Module):
    """
    SegFormer: encoder + decoder + final upsample.

    Args:
        num_classes  (int):          Number of segmentation output classes.
        decoder      (BaseDecoder):  Any decoder implementing BaseDecoder.
                                     Pass an already-constructed instance.
        attn_drop    (float):        Attention dropout for encoder. Default: 0.0
        proj_drop    (float):        Projection dropout for encoder. Default: 0.0

    Example:
        decoder = MLPDecoder(
            in_channels=[32, 64, 160, 256],
            embed_dim=256,
            num_classes=11,
        )
        model = SegFormer(num_classes=11, decoder=decoder)
    """

    def __init__(
        self,
        num_classes: int,
        decoder: BaseDecoder,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        # Encoder: frozen structure — MiT-B0, never modified
        self.encoder = MiTEncoder(
            in_channels=3,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # Decoder: swappable — any BaseDecoder subclass
        self.decoder = decoder

        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, 3, H, W)   — input image batch

        Returns:
            logits : (B, num_classes, H, W)   — full-resolution segmentation logits
        """
        # Remember input resolution for final upsampling
        input_H, input_W = x.shape[2], x.shape[3]

        # ── Step 1: Encoder ────────────────────────────────────────────────────
        # (B, 3, H, W) → [c1, c2, c3, c4]
        #   c1 : (B,  32, H/4,  W/4 )
        #   c2 : (B,  64, H/8,  W/8 )
        #   c3 : (B, 160, H/16, W/16)
        #   c4 : (B, 256, H/32, W/32)
        features = self.encoder(x)

        # ── Step 2: Decoder ────────────────────────────────────────────────────
        # [c1, c2, c3, c4] → (B, num_classes, H/4, W/4)
        logits = self.decoder(features)

        # ── Step 3: Final upsample to original resolution ──────────────────────
        # (B, num_classes, H/4, W/4) → (B, num_classes, H, W)
        logits = F.interpolate(
            logits,
            size=(input_H, input_W),
            mode="bilinear",
            align_corners=False,
        )

        return logits  # (B, num_classes, H, W)


# ── Factory function for E0 baseline ──────────────────────────────────────────

def build_segformer_b0(
    num_classes: int,
    embed_dim: int = 256,
    dropout: float = 0.1,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
) -> SegFormer:
    """
    Build the E0 baseline: SegFormer-B0 + MLP Decoder.

    Args:
        num_classes (int):   Number of target classes.
                             CamVid=11, Cityscapes=19
        embed_dim   (int):   Decoder unified channel dim. Default: 256.
        dropout     (float): Decoder dropout before seg head.

    Returns:
        SegFormer model ready for training.
    """
    decoder = MLPDecoder(
        in_channels=MIT_B0_CHANNELS,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    return SegFormer(
        num_classes=num_classes,
        decoder=decoder,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
    )