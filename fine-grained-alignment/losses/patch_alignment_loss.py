"""
Patch Alignment Loss (Eq. 5) and Beta Schedule from NeurIPS 2025.
Paper: "Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models"
(arXiv:2505.17316)

Computes masked average pooling over visual patch embeddings for each object instance,
calculates cosine similarity against the target label word embedding,
and returns L_patch = 1 - cos_sim(v_k, w_k) averaged over all instances.
"""

from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAlignmentLoss(nn.Module):
    """
    Computes the Patch Alignment Loss (L_patch, Eq. 5).
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def pool_instance_patches(self, patch_tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """
        Pools patch tokens covering an object instance.
        Args:
            patch_tokens: (N, D) where N is number of patches (e.g. 49), D is feature dimension (768).
            patch_mask: (grid, grid) or (N,) binary tensor indicating instance coverage.
        Returns:
            v_k: (D,) pooled representation of the object.
        """
        flat_mask = patch_mask.view(-1).float()  # (N,)
        mask_sum = flat_mask.sum()

        if mask_sum == 0:
            # Fallback if mask is somehow empty: global average
            return patch_tokens.mean(dim=0)

        # Weighted average over active patches: (N, 1) * (N, D) -> sum -> (D,)
        pooled = (flat_mask.unsqueeze(-1) * patch_tokens).sum(dim=0) / mask_sum
        return pooled

    def forward(
        self,
        projected_patches: torch.Tensor,
        batch_instances: List[List[Dict]],
        target_embeddings: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            projected_patches: (B, N, D) projected patch tokens from the MLP projector.
            batch_instances: List of length B, where each element is a list of instance dicts
                             containing 'patch_mask' (7, 7).
            target_embeddings: List of length B, where each element is a list of target
                               word embedding tensors (D,) corresponding to each instance.
        Returns:
            loss: scalar torch.Tensor representing L_patch.
            num_instances: total number of object instances processed in the batch.
        """
        b, n, d = projected_patches.shape
        device = projected_patches.device

        pooled_list = []
        target_list = []

        for b_idx in range(b):
            img_patches = projected_patches[b_idx]  # (49, 768)
            instances = batch_instances[b_idx]
            embeds = target_embeddings[b_idx]

            for inst, target_emb in zip(instances, embeds):
                mask = inst["patch_mask"].to(device)  # (7, 7)
                v_k = self.pool_instance_patches(img_patches, mask)  # (768,)
                w_k = target_emb.to(device)  # (768,)

                pooled_list.append(v_k)
                target_list.append(w_k)

        if len(pooled_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        # Stack into matrices of shape (K, D)
        V = torch.stack(pooled_list, dim=0)  # (K, D)
        W = torch.stack(target_list, dim=0)  # (K, D)

        # Cosine similarity: (V . W) / (||V|| * ||W||)
        cos_sim = F.cosine_similarity(V, W, dim=-1, eps=self.eps)  # (K,)

        # L_patch = mean(1 - cos_sim)
        instance_losses = 1.0 - cos_sim
        loss = instance_losses.mean()

        return loss, len(pooled_list)


class BetaScheduler:
    """
    Linear Beta(step) schedule from beta_start to beta_end over total_steps.
    In the NeurIPS 2025 paper's ablation, ramping beta from 0 to 5 yielded optimal alignment.
    """
    def __init__(self, beta_start: float = 0.0, beta_end: float = 5.0, total_steps: int = 1000):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = max(1, total_steps)

    def get_beta(self, current_step: int) -> float:
        """Computes current beta scalar via linear interpolation."""
        progress = min(1.0, max(0.0, current_step / self.total_steps))
        return self.beta_start + (self.beta_end - self.beta_start) * progress
