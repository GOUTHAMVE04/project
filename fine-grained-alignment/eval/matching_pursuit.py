"""
Matching Pursuit Decomposition Evaluation.
Decomposes patch embeddings into sparse linear combinations of word embeddings
from a candidate vocabulary (COCO categories + common visual concepts).
Evaluates residual reconstruction error, top-1 similarity, and produces qualitative examples.
"""

from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Curated candidate vocabulary: 80 COCO categories + common visual attributes/objects
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

COMMON_ATTRIBUTES = [
    "red", "blue", "green", "yellow", "black", "white", "brown", "gray", "dark", "bright",
    "wooden", "metallic", "plastic", "fabric", "glass", "round", "square", "large", "small",
    "food", "vehicle", "animal", "furniture", "electronics", "kitchen", "outdoor", "indoor",
    "street", "grass", "water", "sky", "floor", "wall", "room", "plate", "table", "wheel"
]

CANDIDATE_VOCABULARY = sorted(list(set(COCO_CATEGORIES + COMMON_ATTRIBUTES)))

def build_vocabulary_dictionary(word_embeddings, vocab: List[str] = None, device: str = "cpu") -> Tuple[torch.Tensor, List[str]]:
    """
    Builds normalized dictionary matrix D of shape (M, 768) for candidate words.
    """
    if vocab is None:
        vocab = CANDIDATE_VOCABULARY

    embeds = word_embeddings.encode_labels(vocab).to(device)  # (M, 768)
    norm_dict = F.normalize(embeds, dim=-1)  # (M, 768)
    return norm_dict, vocab

def matching_pursuit(
    vector: torch.Tensor,
    dictionary: torch.Tensor,
    vocab: List[str],
    num_steps: int = 3
) -> List[Dict]:
    """
    Iterative Matching Pursuit decomposition of vector v using dictionary D:
    At step k:
        j* = argmax_j |D[j] . r_k|
        alpha = D[j*] . r_k
        r_{k+1} = r_k - alpha * D[j*]
    Returns list of matched steps with word, coefficient, and residual norm.
    """
    # Ensure vector is on the same device and float
    r = vector.clone().float()
    orig_norm = torch.norm(r).item()
    steps = []

    for step_idx in range(num_steps):
        # Dot product of current residual with all atoms in dictionary
        correlations = torch.mv(dictionary, r)  # (M,)
        abs_corr = torch.abs(correlations)
        best_idx = torch.argmax(abs_corr).item()

        alpha = correlations[best_idx].item()
        word = vocab[best_idx]
        best_atom = dictionary[best_idx]

        # Update residual
        r = r - alpha * best_atom
        res_norm = torch.norm(r).item()
        res_ratio = (res_norm / orig_norm) if orig_norm > 0 else 0.0

        steps.append({
            "step": step_idx + 1,
            "word": word,
            "coefficient": round(alpha, 4),
            "residual_ratio": round(res_ratio, 4)
        })

    return steps

def evaluate_matching_pursuit(
    vision_encoder,
    projector,
    word_embeddings,
    val_loader: DataLoader,
    device: str = "mps",
    num_steps: int = 3,
    sample_qualitative_count: int = 5
) -> Dict:
    """
    Evaluates Matching Pursuit decomposition across object instance patches in validation set.
    Measures top-1 cosine similarity, final residual norm ratio, and top-1 hit accuracy against GT labels.
    """
    vision_encoder.eval()
    projector.eval()
    word_embeddings.eval()

    dictionary, vocab = build_vocabulary_dictionary(word_embeddings, device=device)

    residual_ratios = []
    top1_sims = []
    top1_matches_gt = []
    qualitative_samples = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            instances_list = batch["instances"]

            raw_patches = vision_encoder(images)  # (B, 49, 768)
            projected = projector(raw_patches)  # (B, 49, 768)

            for b_idx, instances in enumerate(instances_list):
                img_patches = projected[b_idx]  # (49, 768)

                for inst in instances:
                    cat_name = inst["category_name"]
                    patch_mask = inst["patch_mask"].to(device).view(-1)  # (49,)

                    active_indices = torch.nonzero(patch_mask > 0).squeeze(-1)
                    if active_indices.numel() == 0:
                        continue

                    # Select representative patch (center or mean)
                    inst_patches = img_patches[active_indices]
                    mean_patch = inst_patches.mean(dim=0)  # (768,)

                    mp_steps = matching_pursuit(mean_patch, dictionary, vocab, num_steps=num_steps)

                    top1_word = mp_steps[0]["word"]
                    top1_coeff = mp_steps[0]["coefficient"]
                    final_ratio = mp_steps[-1]["residual_ratio"]

                    # Cosine sim of normalized patch against normalized top1 word
                    norm_patch = F.normalize(mean_patch.unsqueeze(0), dim=-1).squeeze(0)
                    dict_idx = vocab.index(top1_word)
                    cos_sim = torch.dot(norm_patch, dictionary[dict_idx]).item()

                    top1_sims.append(cos_sim)
                    residual_ratios.append(final_ratio)
                    top1_matches_gt.append(1.0 if top1_word == cat_name else 0.0)

                    if len(qualitative_samples) < sample_qualitative_count:
                        qualitative_samples.append({
                            "ground_truth_label": cat_name,
                            "top1_word": top1_word,
                            "top1_sim": round(cos_sim, 4),
                            "decomposed_atoms": [f"{s['word']} ({s['coefficient']:+.2f})" for s in mp_steps],
                            "final_residual_ratio": final_ratio
                        })

    mean_res_ratio = float(np.mean(residual_ratios)) if len(residual_ratios) > 0 else 1.0
    mean_top1_sim = float(np.mean(top1_sims)) if len(top1_sims) > 0 else 0.0
    top1_gt_acc = float(np.mean(top1_matches_gt)) if len(top1_matches_gt) > 0 else 0.0

    return {
        "mean_residual_ratio_step3": round(mean_res_ratio, 4),
        "mean_top1_word_similarity": round(mean_top1_sim, 4),
        "top1_matches_ground_truth_acc": round(top1_gt_acc, 4),
        "num_instances_evaluated": len(residual_ratios),
        "qualitative_samples": qualitative_samples
    }
