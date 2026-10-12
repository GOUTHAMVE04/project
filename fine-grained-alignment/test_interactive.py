"""
Interactive Demo & Testing Script for Fine-Grained Patch Alignment.
Allows you to test any image (from COCO or your own file) with custom text queries.
Produces a side-by-side visualization:
  1. Original image
  2. Baseline (untrained) patch-similarity heatmap
  3. Trained projector patch-similarity heatmap
  4. Top decomposed words via Matching Pursuit
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from model.vision_encoder import FrozenCLIPVisionEncoder
from model.word_embeddings import FrozenWordEmbeddings
from model.projector import PatchProjector
from data.patch_dataset import get_clip_transform
from eval.matching_pursuit import build_vocabulary_dictionary, matching_pursuit

def test_image(
    image_path: str,
    query_text: str = None,
    checkpoint_path: str = "checkpoints/projector_best.pt",
    output_png: str = "sanity_checks/demo_test_result.png"
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Running Interactive Test on Device: {device} ===")
    print(f"Image: {image_path}")

    # 1. Load image
    raw_img = Image.open(image_path).convert("RGB")
    transform = get_clip_transform(image_size=224)
    img_tensor = transform(raw_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # 2. Load models
    print("Loading models...")
    vision_encoder = FrozenCLIPVisionEncoder(device=device)
    word_embeddings = FrozenWordEmbeddings(device=device)

    # Trained projector
    trained_proj = PatchProjector().to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        trained_proj.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained checkpoint from {checkpoint_path} (Epoch {ckpt.get('epoch', 'N/A')})")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found; using untrained projector.")

    # Untrained baseline projector (fixed seed for deterministic comparison)
    torch.manual_seed(42)
    baseline_proj = PatchProjector().to(device)
    baseline_proj.eval()
    trained_proj.eval()

    # 3. Extract features
    with torch.no_grad():
        raw_patches = vision_encoder(img_tensor)  # (1, 49, 768)
        base_projected = baseline_proj(raw_patches)[0]  # (49, 768)
        trained_projected = trained_proj(raw_patches)[0]  # (49, 768)

    # Normalize patch vectors
    base_norm = F.normalize(base_projected, dim=-1)
    trained_norm = F.normalize(trained_projected, dim=-1)

    # 4. If query provided, compute spatial alignment heatmaps
    if query_text:
        print(f"Query concept: '{query_text}'")
        word_emb = word_embeddings.encode_single_label(query_text).to(device)
        word_norm = F.normalize(word_emb.unsqueeze(0), dim=-1).squeeze(0)

        # Patch similarity maps: (49,) -> (7, 7)
        base_sims = torch.mv(base_norm, word_norm).view(7, 7).cpu().numpy()
        trained_sims = torch.mv(trained_norm, word_norm).view(7, 7).cpu().numpy()

        print(f"Baseline max similarity to '{query_text}': {base_sims.max():.4f}")
        print(f"Trained  max similarity to '{query_text}': {trained_sims.max():.4f}")

    # 5. Matching Pursuit decomposition on center patch
    dictionary, vocab = build_vocabulary_dictionary(word_embeddings, device=device)
    center_patch = trained_norm[24]  # Patch 24 is center in 7x7 grid
    base_center_patch = base_norm[24]

    trained_mp = matching_pursuit(center_patch, dictionary, vocab, num_steps=3)
    base_mp = matching_pursuit(base_center_patch, dictionary, vocab, num_steps=3)

    print("\n--- Center Patch Vocabulary Decomposition (Matching Pursuit) ---")
    print("Baseline (Untrained):")
    for s in base_mp:
        print(f"  Step {s['step']}: {s['word']} (coeff={s['coefficient']:+.2f}, residual={s['residual_ratio']:.2f})")

    print("\nTrained Projector:")
    for s in trained_mp:
        print(f"  Step {s['step']}: {s['word']} (coeff={s['coefficient']:+.2f}, residual={s['residual_ratio']:.2f})")

    # 6. Generate Side-by-Side Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Subplot 1: Original Image
    axes[0].imshow(raw_img.resize((224, 224)))
    axes[0].set_title("Input Image (224x224)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    if query_text:
        # Subplot 2: Baseline Heatmap
        im1 = axes[1].imshow(base_sims, cmap="coolwarm", vmin=-0.2, vmax=0.9)
        axes[1].set_title(f"Baseline Sim Map\n'{query_text}' (Max: {base_sims.max():.2f})", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Subplot 3: Trained Heatmap
        im2 = axes[2].imshow(trained_sims, cmap="coolwarm", vmin=-0.2, vmax=0.9)
        axes[2].set_title(f"Trained Alignment Map\n'{query_text}' (Max: {trained_sims.max():.2f})", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, "No query text provided.\nRun with --query <word>", ha="center", va="center")
        axes[1].axis("off")
        axes[2].axis("off")

    plt.suptitle(f"Fine-Grained Alignment Test: '{query_text}'", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved visual comparison result to: {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-grained patch alignment on an image.")
    parser.add_argument("--image", type=str, default="data/coco/val2017/000000224200.jpg", help="Path to input image")
    parser.add_argument("--query", type=str, default="dog", help="Text concept query (e.g., dog, cat, car)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/projector_best.pt", help="Path to trained projector checkpoint")
    parser.add_argument("--output", type=str, default="sanity_checks/demo_test_result.png", help="Path to save output heatmap")

    args = parser.parse_args()
    test_image(args.image, args.query, args.checkpoint, args.output)
