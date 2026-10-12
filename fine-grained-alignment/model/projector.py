"""
Trainable 2-Layer MLP Projector for Fine-Grained Patch Alignment.
Follows LLaVA's architectural convention:
    Linear(in_dim, hidden_dim) -> GELU -> Linear(hidden_dim, out_dim)
Projects visual patch tokens into the language embedding space.
This is the ONLY trainable module in the prototype.
"""

import torch
import torch.nn as nn

class PatchProjector(nn.Module):
    """
    2-Layer MLP Projector (LLaVA-style connector).
    Maps CLIP patch embeddings (768-d) to Language Model embedding space (768-d).
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights with standard truncated normal / Xavier for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual patch tokens of shape (..., in_dim), e.g. (B, 49, 768)
        Returns:
            Projected patch tokens of shape (..., out_dim), e.g. (B, 49, 768)
        """
        return self.net(x)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
