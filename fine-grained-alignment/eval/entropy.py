"""
Von Neumann Entropy Evaluation (Eq. 1 in NeurIPS 2025).
Paper: "Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models"

Measures the effective dimensionality and spectral diversity of patch embeddings
before the projector (raw CLIP ViT-B/32) vs. after the projector (LLaVA-style MLP).
"""

from typing import Dict
import torch
import numpy as np
from torch.utils.data import DataLoader

def compute_von_neumann_entropy(patches: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Von Neumann entropy of patch representations (N x D).
    rho = (X @ X.T) / Tr(X @ X.T)
    S(rho) = - sum_i lambda_i * ln(lambda_i)
    Args:
        patches: Tensor of shape (N, D), e.g. (49, 768)
    Returns:
        entropy: float in [0.0, ln(N)]
    """
    # Center or normalize
    X = patches.float()  # (N, D)
    gram = torch.matmul(X, X.T)  # (N, N)
    trace = torch.trace(gram)

    if trace <= eps:
        return 0.0

    rho = gram / trace  # Normalized density matrix

    # Eigenvalues of symmetric positive semi-definite matrix
    eigenvalues = torch.linalg.eigvalsh(rho)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    # Re-normalize to ensure sum is strictly 1.0
    s = eigenvalues.sum()
    if s > 0:
        eigenvalues = eigenvalues / s

    # - sum(lambda * log(lambda))
    positive_eigs = eigenvalues[eigenvalues > eps]
    entropy = -torch.sum(positive_eigs * torch.log(positive_eigs)).item()
    return float(entropy)

def evaluate_von_neumann_entropy(
    vision_encoder,
    projector,
    val_loader: DataLoader,
    device: str = "mps"
) -> Dict[str, float]:
    """
    Computes average Von Neumann entropy across images in the validation set,
    before the projector (CLIP) and after the projector.
    """
    vision_encoder.eval()
    projector.eval()

    entropies_before = []
    entropies_after = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)  # (B, 3, 224, 224)
            raw_patches = vision_encoder(images)  # (B, 49, 768)
            projected_patches = projector(raw_patches)  # (B, 49, 768)

            b = images.shape[0]
            for i in range(b):
                h_before = compute_von_neumann_entropy(raw_patches[i])
                h_after = compute_von_neumann_entropy(projected_patches[i])
                entropies_before.append(h_before)
                entropies_after.append(h_after)

    mean_h_before = float(np.mean(entropies_before)) if len(entropies_before) > 0 else 0.0
    mean_h_after = float(np.mean(entropies_after)) if len(entropies_after) > 0 else 0.0
    compression_ratio = (mean_h_after / mean_h_before) if mean_h_before > 0 else 1.0

    # Theoretical maximum for 49 patches: ln(49) ≈ 3.8918
    max_theoretical_entropy = round(float(np.log(49)), 4)

    return {
        "entropy_before_projector": round(mean_h_before, 4),
        "entropy_after_projector": round(mean_h_after, 4),
        "entropy_compression_ratio": round(compression_ratio, 4),
        "theoretical_max_entropy": max_theoretical_entropy,
        "num_images_evaluated": len(entropies_before)
    }
