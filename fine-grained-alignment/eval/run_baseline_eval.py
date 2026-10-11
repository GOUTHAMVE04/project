"""
Runs all three verification scripts using the untrained / randomly-initialized projector
and saves the official baseline numbers to results/baseline_metrics.json.
"""

import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from data.patch_dataset import COCOPatchDataset, collate_patch_batch
from model.vision_encoder import FrozenCLIPVisionEncoder
from model.word_embeddings import FrozenWordEmbeddings
from model.projector import PatchProjector
from eval.align_vw import evaluate_align_vw
from eval.entropy import evaluate_von_neumann_entropy
from eval.matching_pursuit import evaluate_matching_pursuit

def run_baseline_eval():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Running Phase 5: Official Baseline Evaluation on Device: {device} ===\n")

    # Fixed seed for reproducibility of random projector weights
    torch.manual_seed(42)

    val_dataset = COCOPatchDataset(
        manifest_path="data/coco_subset_manifest.json",
        ann_file="data/coco/annotations/instances_val2017.json",
        img_dir="data/coco/val2017",
        split="val"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_patch_batch,
        num_workers=0
    )
    print(f"Loaded validation set: {len(val_dataset)} images.")

    # Initialize frozen models
    vision_encoder = FrozenCLIPVisionEncoder(device=device)
    word_embeddings = FrozenWordEmbeddings(device=device)

    # Initialize UNTRAINED projector
    untrained_projector = PatchProjector(in_dim=768, hidden_dim=768, out_dim=768).to(device)
    untrained_projector.eval()
    print("Initialized untrained (randomly-initialized) projector.\n")

    # 1. Align(V, W)
    print("Evaluating Metric 1: Align(V, W) [IoU Localization]...")
    align_metrics = evaluate_align_vw(vision_encoder, untrained_projector, word_embeddings, val_loader, device=device)
    print(f"  Align Mean IoU:          {align_metrics['align_mean_iou']}")
    print(f"  Align Acc @ IoU>=0.50:   {align_metrics['align_acc_iou_0.50']}")
    print(f"  On-Target Cosine Sim:    {align_metrics['mean_on_target_cosine_sim']}")
    print(f"  Background Cosine Sim:   {align_metrics['mean_background_cosine_sim']}")

    # 2. Von Neumann Entropy
    print("\nEvaluating Metric 2: Von Neumann Entropy [Compression & Spectral Diversity]...")
    entropy_metrics = evaluate_von_neumann_entropy(vision_encoder, untrained_projector, val_loader, device=device)
    print(f"  Entropy Before Projector (CLIP): {entropy_metrics['entropy_before_projector']}")
    print(f"  Entropy After Projector (Untrained): {entropy_metrics['entropy_after_projector']}")
    print(f"  Compression Ratio:               {entropy_metrics['entropy_compression_ratio']}")

    # 3. Matching Pursuit Decomposition
    print("\nEvaluating Metric 3: Matching Pursuit Decomposition [Vocabulary Interpretability]...")
    mp_metrics = evaluate_matching_pursuit(vision_encoder, untrained_projector, word_embeddings, val_loader, device=device, num_steps=3)
    print(f"  Top-1 Word Similarity:           {mp_metrics['mean_top1_word_similarity']}")
    print(f"  Top-1 Match Ground Truth Acc:    {mp_metrics['top1_matches_ground_truth_acc']}")
    print(f"  Residual Ratio (Step 3):         {mp_metrics['mean_residual_ratio_step3']}")

    baseline_results = {
        "model_status": "untrained_random_baseline",
        "align_vw": align_metrics,
        "von_neumann_entropy": entropy_metrics,
        "matching_pursuit": mp_metrics
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_metrics.json"

    with open(out_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\n✓ Saved official baseline metrics to: {out_path}")
    print("\n=== Phase 5 Baseline Evaluation Completed Successfully! ===")

if __name__ == "__main__":
    run_baseline_eval()
