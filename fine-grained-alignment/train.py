"""
Training script for Fine-Grained Patch Alignment (Laptop-Scale Prototype).
Trains the 2-layer MLP projector on Apple Silicon MPS backend using L_patch (Eq. 5)
with a linear beta schedule (0 -> 5).
Tracks and plots training and validation loss, measures runtime, and saves checkpoints.
"""

import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from data.patch_dataset import COCOPatchDataset, collate_patch_batch
from model.vision_encoder import FrozenCLIPVisionEncoder
from model.word_embeddings import FrozenWordEmbeddings
from model.projector import PatchProjector
from losses.patch_alignment_loss import PatchAlignmentLoss, BetaScheduler

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_training(epochs: int = 25, lr: float = 0.0005, batch_size: int = 16):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Starting Training on Device: {device} ===")

    torch.manual_seed(42)

    # 1. Datasets and Loaders
    train_dataset = COCOPatchDataset(
        manifest_path="data/coco_subset_manifest.json",
        ann_file="data/coco/annotations/instances_val2017.json",
        img_dir="data/coco/val2017",
        split="train"
    )
    val_dataset = COCOPatchDataset(
        manifest_path="data/coco_subset_manifest.json",
        ann_file="data/coco/annotations/instances_val2017.json",
        img_dir="data/coco/val2017",
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patch_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patch_batch,
        num_workers=0
    )

    print(f"Train set: {len(train_dataset)} images ({len(train_loader)} batches/epoch).")
    print(f"Val set:   {len(val_dataset)} images ({len(val_loader)} batches/epoch).")

    # 2. Initialize Models
    print("Loading frozen vision encoder (ViT-B/32) and word embeddings (GPT-2)...")
    vision_encoder = FrozenCLIPVisionEncoder(device=device)
    word_embeddings = FrozenWordEmbeddings(device=device)

    projector = PatchProjector(in_dim=768, hidden_dim=768, out_dim=768).to(device)
    projector.train()

    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    criterion = PatchAlignmentLoss()
    total_steps = epochs * len(train_loader)
    beta_scheduler = BetaScheduler(beta_start=0.0, beta_end=5.0, total_steps=total_steps)

    print(f"Trainable parameters: {projector.trainable_params:,}")
    print(f"Total training steps: {total_steps} ({epochs} epochs x {len(train_loader)} steps)\n")

    history = {
        "epoch": [],
        "train_loss_raw": [],
        "train_loss_weighted": [],
        "val_loss_raw": [],
        "beta": [],
        "epoch_time_sec": []
    }

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    total_training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        projector.train()

        train_raw_losses = []
        train_weighted_losses = []

        for batch in train_loader:
            images = batch["images"].to(device)
            instances_list = batch["instances"]

            # Extract frozen patch features
            with torch.no_grad():
                raw_patches = vision_encoder(images)  # (B, 49, 768)

                # Pre-fetch target word embeddings for each instance
                target_embeddings = []
                for inst_group in instances_list:
                    cat_names = [inst["category_name"] for inst in inst_group]
                    embeds = word_embeddings.encode_labels(cat_names).to(device)
                    target_embeddings.append([embeds[i] for i in range(len(cat_names))])

            # Forward through trainable projector
            projected_patches = projector(raw_patches)  # (B, 49, 768)

            # Compute L_patch (Eq. 5)
            raw_loss, num_insts = criterion(projected_patches, instances_list, target_embeddings)

            # Apply beta schedule
            current_beta = beta_scheduler.get_beta(global_step)
            # Use max(beta, 0.1) as gradient scaling factor to ensure nonzero updates at step 0
            effective_beta = max(current_beta, 0.1)
            loss = effective_beta * raw_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()

            train_raw_losses.append(raw_loss.item())
            train_weighted_losses.append(loss.item())
            global_step += 1

        # Validation phase
        projector.eval()
        val_raw_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                instances_list = batch["instances"]

                raw_patches = vision_encoder(images)
                target_embeddings = []
                for inst_group in instances_list:
                    cat_names = [inst["category_name"] for inst in inst_group]
                    embeds = word_embeddings.encode_labels(cat_names).to(device)
                    target_embeddings.append([embeds[i] for i in range(len(cat_names))])

                projected_patches = projector(raw_patches)
                raw_loss, _ = criterion(projected_patches, instances_list, target_embeddings)
                val_raw_losses.append(raw_loss.item())

        epoch_duration = time.time() - epoch_start
        avg_train_raw = sum(train_raw_losses) / len(train_raw_losses)
        avg_train_weighted = sum(train_weighted_losses) / len(train_weighted_losses)
        avg_val_raw = sum(val_raw_losses) / len(val_raw_losses)

        history["epoch"].append(epoch)
        history["train_loss_raw"].append(avg_train_raw)
        history["train_loss_weighted"].append(avg_train_weighted)
        history["val_loss_raw"].append(avg_val_raw)
        history["beta"].append(current_beta)
        history["epoch_time_sec"].append(epoch_duration)

        # First epoch runtime extrapolation report
        if epoch == 1:
            extrapolated_total = epoch_duration * epochs
            print(f">>> [Epoch 1 Timing Benchmark] Duration: {epoch_duration:.2f}s")
            print(f">>> Extrapolated Total Runtime for {epochs} epochs: {extrapolated_total:.1f}s ({extrapolated_total / 60:.2f} minutes).")

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train L_patch: {avg_train_raw:.4f} | "
            f"Val L_patch: {avg_val_raw:.4f} | "
            f"Beta: {current_beta:.2f} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Save best checkpoint
        if avg_val_raw < best_val_loss:
            best_val_loss = avg_val_raw
            best_ckpt_path = checkpoint_dir / "projector_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "history": history
            }, best_ckpt_path)

    total_duration = time.time() - total_training_start
    print(f"\nTraining completed in {total_duration:.1f}s ({total_duration / 60:.2f} minutes).")
    print(f"Best Validation L_patch: {best_val_loss:.4f}")

    # Save final checkpoint
    final_ckpt_path = checkpoint_dir / "projector_final.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history
    }, final_ckpt_path)
    print(f"Saved final checkpoint to {final_ckpt_path}")

    # Save training history JSON
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot and save loss curves
    plot_loss_curves(history, output_path=results_dir / "training_loss_curve.png")
    plot_loss_curves(history, output_path=Path("sanity_checks") / "training_loss_curve.png")

    return history

def plot_loss_curves(history: dict, output_path: Path):
    """Generates clean publication-quality loss curve plot."""
    epochs = history["epoch"]
    train_raw = history["train_loss_raw"]
    val_raw = history["val_loss_raw"]
    beta_vals = history["beta"]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_train = "#1f77b4"
    color_val = "#d62728"
    color_beta = "#2ca02c"

    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Patch Alignment Loss $L_{patch}$ (1 - cos_sim)", fontsize=12, fontweight="bold", color="#333333")
    line1 = ax1.plot(epochs, train_raw, "o-", color=color_train, linewidth=2, label="Train $L_{patch}$")
    line2 = ax1.plot(epochs, val_raw, "s--", color=color_val, linewidth=2, label="Validation $L_{patch}$")
    ax1.tick_params(axis="y")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Secondary y-axis for beta schedule
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Beta ($\beta$) Weight", fontsize=12, fontweight="bold", color=color_beta)
    line3 = ax2.plot(epochs, beta_vals, ":", color=color_beta, linewidth=2, label=r"Beta Schedule ($\beta$)")
    ax2.tick_params(axis="y", labelcolor=color_beta)
    ax2.set_ylim(-0.2, 5.5)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)

    plt.title("Fine-Grained Patch Alignment: Training Dynamics & Beta Schedule", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve plot to {output_path}")

if __name__ == "__main__":
    run_training()
