"""
base_decoder.py

Abstract base class for all decoder implementations in this project.

Design contract:
  - Every decoder MUST accept a list of 4 feature maps [c1, c2, c3, c4]
  - Every decoder MUST return segmentation logits (B, num_classes, H', W')
  - Final upsampling to original image size is handled by segformer.py, NOT here

This enforces a unified interface so encoders and decoders remain
fully decoupled and swappable for the experiment matrix (E0–E5).
"""

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class BaseDecoder(ABC, nn.Module):
    """
    Abstract base class for all segmentation decoders.

    Subclasses must implement forward() with the following contract:

        Input : List[Tensor]
            features[0] = c1 : (B, C1, H/4,  W/4 )
            features[1] = c2 : (B, C2, H/8,  W/8 )
            features[2] = c3 : (B, C3, H/16, W/16)
            features[3] = c4 : (B, C4, H/32, W/32)

        Output : Tensor
            logits : (B, num_classes, H/4, W/4)

    Note:
        Final upsampling from (H/4, W/4) to the original image resolution
        is the responsibility of segformer.py, not the decoder.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Args:
            features : List of 4 tensors [c1, c2, c3, c4]
                       from MiTEncoder in order of increasing depth.

        Returns:
            logits : (B, num_classes, H/4, W/4)
        """
        raise NotImplementedError

    def _check_features(self, features: List[Tensor]) -> None:
        """
        Optional runtime shape assertion for debugging.
        Call this at the top of forward() during development.
        """
        assert len(features) == 4, (
            f"Expected 4 feature maps [c1,c2,c3,c4], got {len(features)}"
        )