"""
Script to prepare COCO val2017 subset for Fine-Grained Patch Alignment.
Downloads annotations, filters images with 2-8 instances, samples ~400 images,
downloads images concurrently, builds an 80/20 train/val manifest,
and produces 5 sanity-check visualizations.
"""

import os
import sys
import json
import random
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import yaml

def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_file(url, dest_path, desc=None):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return dest_path

    print(f"Downloading {url} to {dest_path}...")
    with tqdm(unit="B", unit_scale=True, miniters=1, desc=desc or dest_path.name) as t:
        def reporthook(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update(b * bsize - t.n)

        urllib.request.urlretrieve(url, filename=str(dest_path), reporthook=reporthook)
    return dest_path

def setup_coco_annotations(data_dir="data"):
    coco_dir = Path(data_dir) / "coco"
    ann_dir = coco_dir / "annotations"
    ann_file = ann_dir / "instances_val2017.json"

    if ann_file.exists():
        print(f"Annotation file found: {ann_file}")
        return ann_file

    zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = Path(data_dir) / "annotations_trainval2017.zip"

    download_file(zip_url, zip_path, desc="COCO Annotations Zip")

    print(f"Extracting instances_val2017.json from {zip_path}...")
    ann_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("instances_val2017.json"):
                with zf.open(member) as source, open(ann_file, "wb") as target:
                    target.write(source.read())
                print(f"Extracted to {ann_file}")
                break

    if zip_path.exists():
        print(f"Removing {zip_path} to conserve space...")
        zip_path.unlink()

    return ann_file

def download_single_image(file_name, img_dir):
    local_path = img_dir / file_name
    if not local_path.exists():
        url = f"http://images.cocodataset.org/val2017/{file_name}"
        urllib.request.urlretrieve(url, str(local_path))
    return file_name

def prepare_subset(config_path="configs/default.yaml", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    cfg = load_config(config_path)

    base_dir = Path(__file__).resolve().parent.parent
    os.chdir(base_dir)

    ann_file = setup_coco_annotations(data_dir="data")
    coco = COCO(str(ann_file))

    min_inst = cfg["data"].get("min_instances", 2)
    max_inst = cfg["data"].get("max_instances", 8)
    num_samples = cfg["data"].get("num_images", 400)
    train_split = cfg["data"].get("train_split", 0.8)

    img_ids = coco.getImgIds()
    valid_candidates = []

    print(f"Filtering images with {min_inst} <= instances <= {max_inst}...")
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        if min_inst <= len(ann_ids) <= max_inst:
            valid_candidates.append((img_id, ann_ids))

    print(f"Found {len(valid_candidates)} candidates meeting criteria.")
    if len(valid_candidates) < num_samples:
        print(f"Warning: Only {len(valid_candidates)} valid images found; using all.")
        sampled = valid_candidates
    else:
        sampled = random.sample(valid_candidates, num_samples)

    # Split into train and validation sets
    random.shuffle(sampled)
    num_train = int(len(sampled) * train_split)
    train_subset = sampled[:num_train]
    val_subset = sampled[num_train:]

    print(f"Sampled {len(sampled)} images: {len(train_subset)} train, {len(val_subset)} val.")

    img_dir = Path("data/coco/val2017")
    img_dir.mkdir(parents=True, exist_ok=True)

    cats = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

    # Parallel download images
    file_names_to_download = [coco.loadImgs(item[0])[0]["file_name"] for item in sampled]
    print(f"Downloading {len(file_names_to_download)} images with 16 parallel threads...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_image, fn, img_dir): fn for fn in file_names_to_download}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            future.result()

    manifest = {
        "info": {
            "description": "COCO val2017 subset for Fine-Grained Alignment Prototype",
            "seed": seed,
            "num_images": len(sampled),
            "num_train": len(train_subset),
            "num_val": len(val_subset),
            "min_instances": min_inst,
            "max_instances": max_inst,
        },
        "images": []
    }

    total_instances = 0
    all_subset = [("train", item) for item in train_subset] + [("val", item) for item in val_subset]

    print("Compiling manifest...")
    for split, (img_id, ann_ids) in tqdm(all_subset, desc="Building manifest"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        anns = coco.loadAnns(ann_ids)
        instances = []
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = cats.get(cat_id, "unknown")
            instances.append({
                "id": ann["id"],
                "category_id": cat_id,
                "category_name": cat_name,
                "bbox": ann["bbox"],  # [x, y, w, h]
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })
            total_instances += 1

        manifest["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": img_info["width"],
            "height": img_info["height"],
            "split": split,
            "instances": instances
        })

    manifest_path = Path("data/coco_subset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {manifest_path} with {len(manifest['images'])} images and {total_instances} total instances.")

    # Generate 5 sanity check visualizations
    generate_sanity_visualizations(coco, manifest, output_dir="sanity_checks", count=5)

def generate_sanity_visualizations(coco, manifest, output_dir="sanity_checks", count=5):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path("data/coco/val2017")

    print(f"Generating {count} sanity check visualizations in {out_dir}...")
    sample_images = manifest["images"][:count]

    colors = [
        (1.0, 0.2, 0.2), # Red
        (0.2, 0.8, 0.2), # Green
        (0.2, 0.4, 1.0), # Blue
        (1.0, 0.8, 0.2), # Yellow
        (0.9, 0.2, 0.9), # Magenta
        (0.2, 0.9, 0.9), # Cyan
        (1.0, 0.5, 0.0), # Orange
        (0.6, 0.2, 0.8), # Purple
    ]

    for idx, img_entry in enumerate(sample_images):
        img_id = img_entry["id"]
        file_name = img_entry["file_name"]
        img_path = img_dir / file_name

        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)
        h, w, _ = img_np.shape

        ann_ids = [inst["id"] for inst in img_entry["instances"]]
        anns = coco.loadAnns(ann_ids)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)

        overlay = np.zeros((h, w, 4), dtype=np.float32)

        for ann_idx, ann in enumerate(anns):
            mask = coco.annToMask(ann)
            color = colors[ann_idx % len(colors)]
            cat_name = img_entry["instances"][ann_idx]["category_name"]

            # Overlay mask
            overlay[mask > 0, :3] = color
            overlay[mask > 0, 3] = 0.45

            # Bounding box
            x, y, bw, bh = ann["bbox"]
            rect = patches.Rectangle((x, y), bw, bh, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)

            # Label badge
            ax.text(x, max(15, y - 5), cat_name, color="white", fontsize=10, weight="bold",
                    bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2))

        ax.imshow(overlay)
        ax.set_title(f"Sanity Check #{idx+1}: {file_name} ({len(anns)} instances - {img_entry['split']})", fontsize=11)
        ax.axis("off")

        save_path = out_dir / f"sanity_check_{idx+1}_{Path(file_name).stem}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved sanity check visualization: {save_path}")

if __name__ == "__main__":
    prepare_subset()
