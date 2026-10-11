"""
Align(V, W) Patch-Localization Evaluation (Eq. 4 in NeurIPS 2025).
Evaluates how accurately projected visual patches localize target category concepts
against ground-truth instance segmentation masks via Intersection-over-Union (IoU).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

def compute_patch_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Computes IoU between binary prediction mask and ground-truth mask.
    Args:
        pred_mask: (grid, grid) or (N,) binary tensor {0, 1}
        gt_mask: (grid, grid) or (N,) binary tensor {0, 1}
    Returns:
        iou: float in [0.0, 1.0]
    """
    pred = (pred_mask > 0).float()
    gt = (gt_mask > 0).float()

    intersection = (pred * gt).sum().item()
    union = ((pred + gt) > 0).float().sum().item()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection + eps) / (union + eps)

def evaluate_align_vw(
    vision_encoder,
    projector,
    word_embeddings,
    val_loader: DataLoader,
    device: str = "mps",
    sim_threshold: float = None  # None uses adaptive threshold (mean + 0.5*std)
) -> Dict[str, float]:
    """
    Evaluates Align(V, W) across the validation set.
    For each instance, computes cosine similarity between all projected patches and the target word embedding,
    thresholds to obtain predicted patch mask, and measures IoU against ground-truth patch mask.
    """
    vision_encoder.eval()
    projector.eval()
    word_embeddings.eval()

    ious = []
    cos_sims_on_target = []
    cos_sims_background = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)  # (B, 3, 224, 224)
            instances_list = batch["instances"]

            raw_patches = vision_encoder(images)  # (B, 49, 768)
            projected_patches = projector(raw_patches)  # (B, 49, 768)
            norm_patches = F.normalize(projected_patches, dim=-1)  # (B, 49, 768)

            for b_idx, instances in enumerate(instances_list):
                img_patches = norm_patches[b_idx]  # (49, 768)

                for inst in instances:
                    cat_name = inst["category_name"]
                    gt_mask = inst["patch_mask"].to(device).view(-1)  # (49,)

                    # Get normalized word embedding
                    word_emb = word_embeddings.encode_single_label(cat_name).to(device)
                    norm_word = F.normalize(word_emb.unsqueeze(0), dim=-1).squeeze(0)  # (768,)

                    # Patch similarities: (49,)
                    patch_sims = torch.mv(img_patches, norm_word)  # (49,)

                    # Adaptive or fixed thresholding
                    if sim_threshold is None:
                        # Top percentile / adaptive threshold
                        thresh = patch_sims.mean() + 0.5 * patch_sims.std()
                    else:
                        thresh = sim_threshold

                    pred_mask = (patch_sims >= thresh).float()

                    # Guarantee at least the top-1 patch is activated
                    if pred_mask.sum() == 0:
                        pred_mask[patch_sims.argmax()] = 1.0

                    iou = compute_patch_iou(pred_mask, gt_mask)
                    ious.append(iou)

                    # Track on-target vs background cosine similarity
                    on_target_sim = patch_sims[gt_mask > 0].mean().item()
                    cos_sims_on_target.append(on_target_sim)

                    if (gt_mask == 0).sum() > 0:
                        bg_sim = patch_sims[gt_mask == 0].mean().item()
                        cos_sims_background.append(bg_sim)

    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    acc_iou_25 = float(np.mean([1.0 if x >= 0.25 else 0.0 for x in ious])) if len(ious) > 0 else 0.0
    acc_iou_50 = float(np.mean([1.0 if x >= 0.50 else 0.0 for x in ious])) if len(ious) > 0 else 0.0
    mean_on_target = float(np.mean(cos_sims_on_target)) if len(cos_sims_on_target) > 0 else 0.0
    mean_bg = float(np.mean(cos_sims_background)) if len(cos_sims_background) > 0 else 0.0

    return {
        "align_mean_iou": round(mean_iou, 4),
        "align_acc_iou_0.25": round(acc_iou_25, 4),
        "align_acc_iou_0.50": round(acc_iou_50, 4),
        "mean_on_target_cosine_sim": round(mean_on_target, 4),
        "mean_background_cosine_sim": round(mean_bg, 4),
        "num_instances_evaluated": len(ious)
    }
