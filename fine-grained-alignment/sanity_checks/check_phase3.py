"""
Sanity check script for Phase 3: Trainable Projector.
Verifies the 2-layer MLP projector, parameter freezing across the pipeline,
trainable parameter count, and gradient isolation.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from model.vision_encoder import FrozenCLIPVisionEncoder
from model.word_embeddings import FrozenWordEmbeddings
from model.projector import PatchProjector

def run_phase3_checks():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Running Phase 3 Sanity Checks on Device: {device} ===\n")

    # 1. Instantiate Modules
    vision_enc = FrozenCLIPVisionEncoder(device=device)
    word_enc = FrozenWordEmbeddings(device=device)
    projector = PatchProjector(in_dim=768, hidden_dim=768, out_dim=768).to(device)

    # 2. Parameter Count Breakdown
    vision_total = vision_enc.total_params
    vision_trainable = vision_enc.trainable_params

    word_total = word_enc.total_params
    word_trainable = word_enc.trainable_params

    proj_total = projector.total_params
    proj_trainable = projector.trainable_params

    print("--- Parameter Counts & Freezing Audit ---")
    print(f"Vision Encoder:   Total = {vision_total:,} | Trainable = {vision_trainable} (requires_grad={vision_trainable > 0})")
    print(f"Word Embeddings:  Total = {word_total:,} | Trainable = {word_trainable} (requires_grad={word_trainable > 0})")
    print(f"MLP Projector:    Total = {proj_total:,} | Trainable = {proj_trainable:,} (requires_grad={proj_trainable > 0})")
    print("-" * 55)
    print(f"Overall System:   Total = {vision_total + word_total + proj_total:,}")
    print(f"Total Frozen:     {vision_total + word_total:,}")
    print(f"Total Trainable:  {proj_trainable:,}")

    # Assertions
    assert vision_trainable == 0, "Vision encoder must be completely frozen!"
    assert word_trainable == 0, "Word embeddings must be completely frozen!"
    assert proj_trainable == proj_total == 1_181_184, f"Projector parameter mismatch: got {proj_trainable}, expected 1,181,184"

    # Verify every individual parameter
    for name, param in vision_enc.named_parameters():
        assert not param.requires_grad, f"Found un-frozen vision parameter: {name}"
    for name, param in word_enc.named_parameters():
        assert not param.requires_grad, f"Found un-frozen word parameter: {name}"
    for name, param in projector.named_parameters():
        assert param.requires_grad, f"Found non-trainable projector parameter: {name}"

    print("\n✓ Confirmed: ONLY the 2-layer MLP projector has requires_grad=True.")

    # 3. Forward Pass & Output Dimensions
    print("\n--- Forward Pass & Output Dimensions ---")
    dummy_images = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        patch_tokens = vision_enc(dummy_images)  # (2, 49, 768)
    projected_patches = projector(patch_tokens)  # (2, 49, 768)

    print(f"Input image shape:             {dummy_images.shape}")
    print(f"CLIP patch tokens shape:       {patch_tokens.shape}")
    print(f"Projected patch tokens shape:  {projected_patches.shape} (Expected: [2, 49, 768])")
    assert projected_patches.shape == (2, 49, 768), "Projector output shape mismatch!"

    # 4. Gradient Isolation Test
    print("\n--- Gradient Isolation & Backward Pass Test ---")
    dummy_labels = ["cat", "dog"]
    target_embeds = word_enc.encode_labels(dummy_labels).to(device)  # (2, 768)

    # Simple dummy loss: cosine distance between mean projected patch and label embedding
    pooled_projected = projected_patches.mean(dim=1)  # (2, 768)
    cos_sim = nn.functional.cosine_similarity(pooled_projected, target_embeds, dim=-1)
    dummy_loss = (1.0 - cos_sim).mean()

    projector.zero_grad()
    dummy_loss.backward()

    # Verify projector gradients are computed and non-zero
    for name, param in projector.named_parameters():
        assert param.grad is not None, f"Projector parameter {name} has no gradient!"
        grad_norm = param.grad.norm().item()
        print(f"  Projector {name} grad norm: {grad_norm:.6f}")
        assert grad_norm > 0, f"Projector parameter {name} has zero gradient!"

    print("\n✓ Backward pass successful: gradients flow exclusively to the projector.")
    print("\n=== Phase 3 Sanity Checks PASSED Successfully! ===")

if __name__ == "__main__":
    run_phase3_checks()
