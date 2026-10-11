"""
PyTorch Dataset for Fine-Grained Patch Alignment.
Loads sampled COCO images, downsamples instance segmentation masks to patch grids,
and extracts category labels for alignment loss computation.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO

# CLIP standard normalization
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def get_clip_transform(image_size=224):
    """Standard CLIP image preprocessing transform."""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])

def mask_to_patch_grid(mask_np, grid_size=7, threshold=0.1):
    """
    Downsamples a pixel-level binary mask (H, W) to a patch grid (grid_size, grid_size).
    Calculates the area overlap fraction of the mask in each patch cell.
    If fraction >= threshold, the patch is marked 1 (active), else 0.
    If no patch exceeds threshold (e.g. tiny object), the patch with maximum overlap is activated.
    
    Returns:
        patch_mask: torch.Tensor of shape (grid_size, grid_size), dtype float32
    """
    h, w = mask_np.shape
    patch_h = h / grid_size
    patch_w = w / grid_size

    grid_mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = int(round(i * patch_h))
            y2 = int(round((i + 1) * patch_h))
            x1 = int(round(j * patch_w))
            x2 = int(round((j + 1) * patch_w))

            cell = mask_np[y1:y2, x1:x2]
            if cell.size > 0:
                grid_mask[i, j] = np.mean(cell)

    binary_grid = (grid_mask >= threshold).astype(np.float32)

    # Ensure at least one patch is active for valid objects
    if binary_grid.sum() == 0 and grid_mask.max() > 0:
        max_idx = np.unravel_index(np.argmax(grid_mask), grid_mask.shape)
        binary_grid[max_idx] = 1.0

    return torch.from_numpy(binary_grid)

class COCOPatchDataset(Dataset):
    """
    Dataset returning image tensors, patch-level binary instance masks,
    original resolution masks, and category label strings.
    """
    def __init__(self, manifest_path="data/coco_subset_manifest.json",
                 ann_file="data/coco/annotations/instances_val2017.json",
                 img_dir="data/coco/val2017",
                 split=None,
                 image_size=224,
                 grid_size=7):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.grid_size = grid_size
        self.transform = get_clip_transform(image_size)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if split is not None:
            self.images = [img for img in manifest["images"] if img["split"] == split]
        else:
            self.images = manifest["images"]

        self.coco = COCO(ann_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        entry = self.images[idx]
        file_name = entry["file_name"]
        img_path = self.img_dir / file_name

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        img_tensor = self.transform(image)  # (3, 224, 224)

        instances = []
        for inst_entry in entry["instances"]:
            ann_id = inst_entry["id"]
            ann = self.coco.loadAnns([ann_id])[0]
            mask_np = self.coco.annToMask(ann)  # (H, W) binary

            patch_mask = mask_to_patch_grid(mask_np, grid_size=self.grid_size)  # (7, 7)

            instances.append({
                "id": ann_id,
                "category_id": inst_entry["category_id"],
                "category_name": inst_entry["category_name"],
                "patch_mask": patch_mask,  # (7, 7) float tensor
                "orig_mask": torch.from_numpy(mask_np.astype(np.uint8)),  # (H, W)
                "bbox": inst_entry["bbox"],
            })

        return {
            "image_id": entry["id"],
            "file_name": file_name,
            "image": img_tensor,
            "orig_size": (orig_h, orig_w),
            "instances": instances
        }

def collate_patch_batch(batch):
    """
    Collate function to batch images and bundle instances per image.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    image_ids = [item["image_id"] for item in batch]
    file_names = [item["file_name"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    instances_list = [item["instances"] for item in batch]

    return {
        "images": images,
        "image_ids": image_ids,
        "file_names": file_names,
        "orig_sizes": orig_sizes,
        "instances": instances_list
    }
