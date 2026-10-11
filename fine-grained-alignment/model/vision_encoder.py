"""
Frozen CLIP Vision Encoder for Fine-Grained Patch Alignment.
Uses ViT-B/32 (openai pretrained) to extract patch-level token embeddings (49 patches, 768-dim).
All parameters are frozen (requires_grad = False).
"""

import torch
import torch.nn as nn
import open_clip

class FrozenCLIPVisionEncoder(nn.Module):
    """
    Frozen OpenCLIP ViT-B/32 Vision Encoder.
    Extracts spatial patch tokens (shape: B, 49, 768) rather than just pooled tokens.
    """
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cpu"):
        super().__init__()
        self.device = device
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = clip_model.visual
        self.visual.output_tokens = True

        # Completely freeze all visual encoder parameters
        for param in self.visual.parameters():
            param.requires_grad = False

        self.eval()
        self.to(device)

    def forward(self, images: torch.Tensor, reshape_grid: bool = False) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            reshape_grid: If True, returns (B, 7, 7, 768); else (B, 49, 768)
        Returns:
            patch_tokens: Tensor of patch embeddings (B, 49, 768) or (B, 7, 7, 768)
        """
        with torch.no_grad():
            images = images.to(self.device)
            pooled, tokens = self.visual(images)
            # tokens is shape (B, 49, 768) - the 49 visual patches excluding the CLS token
            if reshape_grid:
                b, n, d = tokens.shape
                grid = int(n ** 0.5)
                tokens = tokens.view(b, grid, grid, d)
            return tokens

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
