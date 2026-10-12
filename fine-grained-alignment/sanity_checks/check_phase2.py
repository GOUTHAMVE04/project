"""
Sanity check script for Phase 2: Frozen Components.
Verifies FrozenCLIPVisionEncoder, FrozenWordEmbeddings, and mask_to_patch_grid.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from model.vision_encoder import FrozenCLIPVisionEncoder
from model.word_embeddings import FrozenWordEmbeddings
from data.patch_dataset import mask_to_patch_grid

def run_phase2_checks():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Running Phase 2 Sanity Checks on Device: {device} ===\n")

    # 1. Vision Encoder Verification
    print("--- 1. Testing FrozenCLIPVisionEncoder (ViT-B/32) ---")
    vision_enc = FrozenCLIPVisionEncoder(device=device)
    dummy_images = torch.randn(4, 3, 224, 224, device=device)

    patch_tokens = vision_enc(dummy_images, reshape_grid=False)
    patch_grid = vision_enc(dummy_images, reshape_grid=True)

    print(f"Input image tensor shape: {dummy_images.shape}")
    print(f"Extracted patch tokens shape: {patch_tokens.shape} (Expected: [4, 49, 768])")
    print(f"Reshaped patch grid shape: {patch_grid.shape} (Expected: [4, 7, 7, 768])")
    print(f"Vision Encoder total params: {vision_enc.total_params:,}")
    print(f"Vision Encoder trainable params: {vision_enc.trainable_params} (Confirmed FROZEN)")
    assert patch_tokens.shape == (4, 49, 768), "Patch token shape mismatch!"
    assert patch_grid.shape == (4, 7, 7, 768), "Patch grid shape mismatch!"
    assert vision_enc.trainable_params == 0, "Vision encoder parameters must not be trainable!"

    # 2. Word Embeddings Verification
    print("\n--- 2. Testing FrozenWordEmbeddings (GPT-2 small) ---")
    word_enc = FrozenWordEmbeddings(device=device)
    sample_labels = ["cat", "dog", "tennis racket", "dining table", "skateboard"]

    single_embed = word_enc.encode_single_label("tennis racket")
    batch_embeds = word_enc.encode_labels(sample_labels)

    print(f"Sample labels: {sample_labels}")
    print(f"Single label ('tennis racket') pooled embedding shape: {single_embed.shape} (Expected: [768])")
    print(f"Batch labels pooled embedding shape: {batch_embeds.shape} (Expected: [{len(sample_labels)}, 768])")
    print(f"Word Embeddings total params: {word_enc.total_params:,}")
    print(f"Word Embeddings trainable params: {word_enc.trainable_params} (Confirmed FROZEN)")
    assert single_embed.shape == (768,), "Single label embedding shape mismatch!"
    assert batch_embeds.shape == (len(sample_labels), 768), "Batch label embedding shape mismatch!"
    assert word_enc.trainable_params == 0, "Word embedding parameters must not be trainable!"

    # 3. Mask-to-Patch-Grid Downsampling Verification
    print("\n--- 3. Testing Mask Downsampling Utility (mask_to_patch_grid) ---")
    h, w = 480, 640
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    # Simulate an object occupying the central third of the image
    dummy_mask[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)] = 1

    grid_mask = mask_to_patch_grid(dummy_mask, grid_size=7, threshold=0.1)
    print(f"Original pixel mask shape: {dummy_mask.shape}")
    print(f"Downsampled patch grid shape: {grid_mask.shape} (Expected: [7, 7])")
    print(f"Active patches count in 7x7 grid: {grid_mask.sum().item():.0f} / 49")
    print("Downsampled 7x7 Binary Activation Pattern:")
    for row in grid_mask.numpy():
        print("  " + " ".join(["■" if v > 0 else "·" for v in row]))
    assert grid_mask.shape == (7, 7), "Patch grid mask shape mismatch!"

    print("\n=== Phase 2 Sanity Checks PASSED Successfully! ===")

if __name__ == "__main__":
    run_phase2_checks()
