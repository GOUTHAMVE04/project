import os
import re
import json
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional

from pipeline.hallucination_verifier import DescriptionHallucinationVerifier, ProposedGroundedVerifier
from perturbations.visual_perturbation import VisualPerturber
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# POPE candidate pools (Li et al., EMNLP 2023)
# ─────────────────────────────────────────────────────────────────────────────

POPE_POSITIVE_POOL = [
    "cat", "dog", "person", "car", "bicycle", "fence", "tree", "bird",
    "chair", "table", "flower", "grass", "horse", "boat", "bus",
    "laptop", "bench", "backpack", "umbrella", "bottle", "cup",
    "clock", "vase", "potted plant", "couch", "book", "knife"
]

POPE_NEGATIVE_POOL = [
    "elephant", "giraffe", "airplane", "helicopter", "submarine",
    "snowboard", "ski", "surfboard", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bear", "zebra", "kite", "baseball bat",
    "skateboard", "tennis racket", "microwave", "refrigerator", "toaster"
]


class ImagePOPEEvaluator:
    """
    Runs POPE-style binary hallucination evaluation on ANY uploaded image.

    Following Li et al. (EMNLP 2023) and Wang et al. (ACL 2024):
    - Probes the image with binary questions: 'Is there a [object] in the image?'
    - Evaluates across 3 splits:
        1. random: random mix of verified present and absent objects
        2. popular: frequent visual concepts in common scenes
        3. adversarial: visually similar hard negative distractors
    - Calculates Accuracy, Precision, Recall, F1, and Yes-Ratio using Instruction
      Contrastive Decoding (ICD) evidence grounding.
    """

    def __init__(self, verifier: ProposedGroundedVerifier):
        self.verifier = verifier

    def _probe_image_for_present_objects(self, image: Image.Image, candidates: List[str]) -> List[str]:
        present = []
        for obj in candidates:
            result = self.verifier.verify_concept_grounding(image, obj)
            if result["is_present"]:
                present.append(obj)
        return present

    def _build_pope_questions(
        self,
        image: Image.Image,
        split: str,
        n_positive: int = 6,
        n_negative: int = 6
    ) -> List[Dict[str, Any]]:
        confirmed_present = self._probe_image_for_present_objects(image, POPE_POSITIVE_POOL)
        confirmed_absent  = [o for o in POPE_NEGATIVE_POOL if o not in confirmed_present]
        pool_absent       = [o for o in POPE_POSITIVE_POOL if o not in confirmed_present]

        pos_q = confirmed_present[:n_positive]

        if split == "random":
            neg_pool = confirmed_absent + pool_absent
            neg_q = [o for o in neg_pool if o not in confirmed_present][:max(n_negative, len(pos_q) if pos_q else 6)]
        elif split == "popular":
            # Frequent concepts in common scenes that are confirmed absent in this specific image
            neg_pool = pool_absent + confirmed_absent
            neg_q = [o for o in neg_pool if o not in confirmed_present][:max(n_negative, len(pos_q) if pos_q else 6)]
        elif split == "adversarial":
            ADVERSARIAL_PAIRS = {
                "cat": "dog", "dog": "cat", "car": "bus", "bicycle": "motorcycle",
                "person": "mannequin", "tree": "potted plant", "fence": "wall",
                "bird": "kite", "chair": "bench", "couch": "chair"
            }
            hard_negs = [ADVERSARIAL_PAIRS[o] for o in confirmed_present if o in ADVERSARIAL_PAIRS and ADVERSARIAL_PAIRS[o] not in confirmed_present]
            neg_q = list(dict.fromkeys(hard_negs + pool_absent + confirmed_absent))[:max(n_negative, len(pos_q) if pos_q else 6)]
        else:
            pos_q = confirmed_present[:n_positive]
            neg_q = confirmed_absent[:n_negative]

        questions = []
        qid = 1
        for obj in pos_q:
            questions.append({
                "question_id": qid,
                "question": f"Is there a {obj} in the image?",
                "object": obj,
                "label": "yes",
                "split": split
            })
            qid += 1
        for obj in neg_q:
            questions.append({
                "question_id": qid,
                "question": f"Is there a {obj} in the image?",
                "object": obj,
                "label": "no",
                "split": split
            })
            qid += 1
        return questions

    def evaluate_split(self, image: Image.Image, split: str) -> Dict[str, Any]:
        questions = self._build_pope_questions(image, split)
        tp = fp = tn = fn = 0
        total_yes = 0
        per_question_log = []

        for q in questions:
            result = self.verifier.verify_concept_grounding(image, q["object"])
            pred = "yes" if result["is_present"] else "no"
            gt   = q["label"]

            if pred == "yes":
                total_yes += 1

            if gt == "yes" and pred == "yes":
                tp += 1; outcome = "TP"
            elif gt == "no"  and pred == "yes":
                fp += 1; outcome = "FP"
            elif gt == "no"  and pred == "no":
                tn += 1; outcome = "TN"
            else:
                fn += 1; outcome = "FN"

            per_question_log.append({
                "question": q["question"],
                "ground_truth": gt,
                "model_prediction": pred,
                "outcome": outcome,
                "standard_evidence":   result.get("standard_evidence", result.get("clean_evidence", 0)),
                "disturbance_evidence": result.get("disturbance_evidence", 0),
                "icd_calibrated_score": result.get("icd_calibrated_score", result.get("calibrated_score", 0))
            })

        total     = len(questions)
        accuracy  = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp)    if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        yes_ratio = total_yes / total if total > 0 else 0.0

        return {
            "split":         split,
            "total_samples": total,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy":       round(accuracy  * 100, 2),
            "precision":      round(precision * 100, 2),
            "recall":         round(recall    * 100, 2),
            "f1_score":       round(f1        * 100, 2),
            "yes_ratio":      round(yes_ratio * 100, 2),
            "per_question":   per_question_log
        }

    def evaluate_all_splits(self, image: Image.Image) -> Dict[str, Any]:
        """Runs all three POPE splits and returns combined report."""
        results = {}
        for split in ["random", "popular", "adversarial"]:
            results[split] = self.evaluate_split(image, split)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# JSON Report Saver
# ─────────────────────────────────────────────────────────────────────────────

def save_image_report(
    image_path: str,
    description: str,
    verification_result: Dict[str, Any],
    pope_results: Optional[Dict[str, Any]] = None
) -> str:
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    base_name  = os.path.splitext(os.path.basename(image_path))[0]
    safe_name  = re.sub(r"[^\w\-]", "_", base_name)
    out_path   = os.path.join(Config.RESULTS_DIR, f"{safe_name}_hallucination_report.json")

    report = {
        "framework": "Instruction Contrastive Decoding (ICD - Wang et al., ACL 2024)",
        "image_path":   image_path,
        "description":  description,
        "verification": verification_result,
        "pope_evaluation": pope_results
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-Side Comparison Image Builder (Matches results/images__1__perturbed_comparison.png)
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_image(
    original_img: Image.Image,
    perturbed_img: Image.Image,
    result: Dict[str, Any],
    description: str,
    pope_results: Optional[Dict[str, Any]] = None
) -> Image.Image:
    """
    Builds a diagnostic side-by-side PNG:
      Left   : Original image (V_clean)
      Middle : Perturbed image (Gaussian noise / ICD contrast counterfactual)
      Right  : Audit panel — hallucination report + POPE scores
    """
    DISPLAY_H = 340

    def resize_keep_aspect(img, height):
        w, h = img.size
        new_w = max(10, int(w * height / h))
        return img.resize((new_w, height), Image.LANCZOS)

    orig_disp = resize_keep_aspect(original_img, DISPLAY_H)
    pert_disp = resize_keep_aspect(perturbed_img, DISPLAY_H)

    PANEL_W = 400
    IMG_W   = orig_disp.width
    TOTAL_W = IMG_W + pert_disp.width + PANEL_W + 6
    TOTAL_H = DISPLAY_H + 90

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), color=(20, 20, 32))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 15)
        font_body  = ImageFont.truetype("arial.ttf", 12)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_title = font_body = font_small = ImageFont.load_default()

    # ── Header bar ────────────────────────────────────────────────────────────
    header_color = (200, 40, 50) if result["is_hallucinated"] else (34, 139, 80)
    draw.rectangle([0, 0, TOTAL_W, 50], fill=header_color)
    draw.text((10, 5),  "Instruction Contrastive Decoding (ICD) Hallucination Verifier",
              fill="white", font=font_title)
    chair_i = result["hallucination_rate_chair_i"]
    verdict = result["verdict"]
    draw.text((10, 24), f"  VERDICT: {verdict}   |   CHAIR_i: {chair_i}%   |   Wang et al. (ACL 2024)",
              fill="white", font=font_body)

    # ── Column labels ─────────────────────────────────────────────────────────
    col_pert  = IMG_W + 2
    col_panel = IMG_W + pert_disp.width + 4
    y_label   = 54

    draw.text((10,            y_label), "Original Image (V_clean)",
              fill=(200, 230, 255), font=font_small)
    draw.text((col_pert + 8,  y_label), f"Perturbed / Disturbance (sigma={Config.NOISE_STD})",
              fill=(255, 200, 130), font=font_small)
    draw.text((col_panel + 8, y_label), "Verification + POPE Report",
              fill=(200, 230, 255), font=font_small)

    # ── Images ────────────────────────────────────────────────────────────────
    canvas.paste(orig_disp, (0,           68))
    canvas.paste(pert_disp, (col_pert + 2, 68))

    # ── Dividers ──────────────────────────────────────────────────────────────
    draw.line([(col_pert,       0), (col_pert,       TOTAL_H)], fill=(55, 55, 75), width=2)
    draw.line([(col_panel + 2,  0), (col_panel + 2,  TOTAL_H)], fill=(55, 55, 75), width=2)

    # ── Audit Panel ───────────────────────────────────────────────────────────
    px = col_panel + 10
    py = 68

    # Description
    desc_short = description if len(description) <= 44 else description[:41] + "..."
    draw.text((px, py), "Description:", fill=(170, 170, 210), font=font_small)
    py += 14
    draw.text((px, py), f'"{desc_short}"', fill=(240, 240, 255), font=font_small)
    py += 20

    # Grounded
    draw.text((px, py), "Grounded:", fill=(80, 220, 130), font=font_small)
    py += 13
    draw.text((px, py), str(result["verified_objects"] or "[ none ]"),
              fill=(170, 255, 180), font=font_small)
    py += 18

    # Hallucinated
    draw.text((px, py), "Hallucinated:", fill=(255, 90, 90), font=font_small)
    py += 13
    draw.text((px, py), str(result["hallucinated_objects"] or "[ none ]"),
              fill=(255, 170, 170), font=font_small)
    py += 20

    # Claim audit
    draw.text((px, py), "Claim Audit (ICD contrast):", fill=(200, 200, 255), font=font_small)
    py += 14
    for item in result["claims_breakdown"]:
        is_pass = item["status"] == "VERIFIED_PRESENT"
        mark_color = (80, 220, 130) if is_pass else (255, 90, 90)
        mark = "[PASS]" if is_pass else "[FAIL]"
        obj  = item["claimed_object"][:14]
        c    = item.get("standard_evidence", item.get("clean_evidence", 0))
        d    = item.get("disturbance_evidence", 0)
        draw.text((px, py), f"{mark} {obj}", fill=mark_color, font=font_small)
        py += 13
        draw.text((px + 12, py), f"Std:{c:.2f}  Dist:{d:.2f}",
                  fill=(160, 160, 200), font=font_small)
        py += 15
        if py > TOTAL_H - 75:
            break

    # ── POPE scores panel ─────────────────────────────────────────────────────
    if pope_results and py < TOTAL_H - 60:
        py += 4
        draw.line([(px, py), (px + PANEL_W - 20, py)], fill=(70, 70, 100), width=1)
        py += 6
        draw.text((px, py), "POPE Evaluation (image-specific):", fill=(180, 180, 255), font=font_small)
        py += 14
        hdr = f"  {'Split':<12} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6}"
        draw.text((px, py), hdr, fill=(140, 140, 190), font=font_small)
        py += 13
        for split, metrics in pope_results.items():
            row = (f"  {split:<12} {metrics['accuracy']:>5.1f}%"
                   f" {metrics['precision']:>5.1f}%"
                   f" {metrics['recall']:>5.1f}%"
                   f" {metrics['f1_score']:>5.1f}%")
            draw.text((px, py), row, fill=(220, 220, 255), font=font_small)
            py += 13
            if py > TOTAL_H - 22:
                break

    # ── Footer ────────────────────────────────────────────────────────────────
    draw.text((px, TOTAL_H - 16),
              "ICD (Wang et al., ACL 2024) | POPE (EMNLP 2023) | CHAIR (EMNLP 2018)",
              fill=(80, 80, 120), font=font_small)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Path Resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_image_path(raw_path: Optional[str]) -> str:
    if raw_path:
        # Strip extraneous quotes
        cleaned = raw_path.strip().strip("'\"")
        if os.path.exists(cleaned):
            return os.path.abspath(cleaned)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        rel = os.path.join(project_dir, cleaned)
        if os.path.exists(rel):
            return os.path.abspath(rel)

    # Fallback search order
    project_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(project_dir, "images (1).jpg"),
        os.path.join(project_dir, "download.jpg"),
        os.path.join(project_dir, "data", "demo_park_dog.png")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
            
    for f in os.listdir(project_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.abspath(os.path.join(project_dir, f))

    raise FileNotFoundError("Could not find any suitable image in workspace.")


# ─────────────────────────────────────────────────────────────────────────────
# Core High-Level Verification API
# ─────────────────────────────────────────────────────────────────────────────

def check_hallucination_and_pope(
    image_path: str,
    description: str,
    model_mode: str = "proposed",
    run_pope: bool = True,
    save_visual: bool = True,
    verifier_instance: Optional[DescriptionHallucinationVerifier] = None
) -> Dict[str, Any]:
    """
    Checks hallucination and POPE values for WHATEVER image and description provided.
    Implements Instruction Contrastive Decoding (Wang et al., ACL 2024).

    Returns:
      Dict containing verification results, POPE evaluation metrics across all splits,
      and saved visualization file paths.
    """
    resolved_img = resolve_image_path(image_path)
    verifier = verifier_instance or DescriptionHallucinationVerifier()
    orig_pil = Image.open(resolved_img).convert("RGB")

    print("\n" + "=" * 76)
    print("  IMAGE HALLUCINATION & POPE VERIFICATION: INSTRUCTION CONTRASTIVE DECODING")
    print("  Following: Wang et al. (ACL 2024) & Li et al. (EMNLP 2023)")
    print("=" * 76)
    print(f"Target Image:      {resolved_img}")
    print(f"Input Description: \"{description}\"")
    print(f"Model Engine:      {'Instruction Contrastive Decoding (ICD)' if model_mode == 'proposed' else 'Base Model'}")
    print("-" * 76)

    # 1. Verify description
    result = verifier.verify_description(orig_pil, description, model_mode=model_mode)

    print("\n[1. HALLUCINATION VERIFICATION RESULTS]")
    print(f"  Overall Verdict:       {result['verdict']}")
    print(f"  Is Hallucinated:       {'YES' if result['is_hallucinated'] else 'NO'}")
    print(f"  CHAIR_i Hallucination: {result['hallucination_rate_chair_i']}%")
    print(f"  Claims Checked:        {result['total_claims_checked']}")
    print(f"  Grounded Objects:      {result['verified_objects']}")
    print(f"  Hallucinated Objects:  {result['hallucinated_objects']}")
    print(f"  Annotated Text:        \"{result['annotated_description']}\"")
    
    print("\n[Claim-by-Claim ICD Audit]")
    for item in result["claims_breakdown"]:
        mark = "[PASS]" if item["status"] == "VERIFIED_PRESENT" else "[FAIL]"
        std  = item.get("standard_evidence", "N/A")
        dist = item.get("disturbance_evidence", "N/A")
        icd  = item.get("icd_calibrated_score", "N/A")
        print(f"  {mark} '{item['claimed_object']:<16}' | Std:{std} | Disturbance:{dist} | ICD Logit:{icd}")

    # 2. POPE evaluation for this specific image
    pope_results = None
    if run_pope:
        print("\n[2. POPE BENCHMARK EVALUATION ON THIS IMAGE]")
        print("  Generating image-specific binary questions (Random, Popular, Adversarial)...")
        print("  Evaluating using ICD contrastive grounding engine...\n")
        
        pope_evaluator = ImagePOPEEvaluator(verifier.proposed_model)
        pope_results   = pope_evaluator.evaluate_all_splits(orig_pil)

        print(f"  {'Split':<14} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'F1':>7} {'Yes%':>7} {'Samples':>8}")
        print("  " + "-" * 62)
        for split, metrics in pope_results.items():
            print(f"  {split:<14} {metrics['accuracy']:>6.1f}% {metrics['precision']:>6.1f}% "
                  f"{metrics['recall']:>7.1f}% {metrics['f1_score']:>6.1f}% "
                  f"{metrics['yes_ratio']:>6.1f}% {metrics['total_samples']:>8}")

    # 3. Save JSON report
    report_path = save_image_report(resolved_img, description, result, pope_results)
    print(f"\n[Report Saved] -> {os.path.abspath(report_path)}")

    # 4. Save side-by-side comparison image matching results/images__1__perturbed_comparison.png
    comp_path = None
    if save_visual:
        perturbed_pil = VisualPerturber.add_gaussian_noise_pil(orig_pil, noise_std=Config.NOISE_STD)
        comparison    = build_comparison_image(orig_pil, perturbed_pil, result, description, pope_results)

        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(resolved_img))[0]
        safe_name = re.sub(r"[^\w\-]", "_", base_name)
        comp_path = os.path.join(Config.RESULTS_DIR, f"{safe_name}_perturbed_comparison.png")
        raw_path  = os.path.join(Config.RESULTS_DIR, f"{safe_name}_perturbed_raw.png")

        comparison.save(comp_path)
        perturbed_pil.save(raw_path)
        print(f"[Visual Saved] Comparison PNG -> {os.path.abspath(comp_path)}")
        print(f"[Visual Saved] Perturbed Raw  -> {os.path.abspath(raw_path)}")

    print("=" * 76 + "\n")

    return {
        "image_path": resolved_img,
        "description": description,
        "verification": result,
        "pope_results": pope_results,
        "comparison_image_path": comp_path,
        "report_json_path": report_path
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify Hallucination and POPE value for any Image and Description using Instruction Contrastive Decoding (ICD - Wang et al., ACL 2024)."
    )
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image.")
    parser.add_argument("--description", "-d", type=str, default=None, help="Image description to verify.")
    parser.add_argument("--model", "-m", type=str, default="proposed", choices=["proposed", "base"], help="Model mode ('proposed' uses ICD).")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare Base Model vs. Proposed ICD Model side by side.")
    parser.add_argument("--no-pope", action="store_true", help="Skip POPE benchmark evaluation.")
    parser.add_argument("--no-visual", action="store_true", help="Skip generating the side-by-side comparison PNG.")
    parser.add_argument("--interactive", action="store_true", help="Prompt interactively for image path and description.")

    args = parser.parse_args()

    image_path = args.image
    description = args.description

    # Interactive prompt if requested or if no arguments passed and running in a terminal
    if args.interactive or (image_path is None and description is None and sys.stdin.isatty()):
        print("\n--- Interactive Hallucination & POPE Verifier ---")
        prompt_img = input("Enter Image path (leave blank for default): ").strip()
        if prompt_img:
            image_path = prompt_img
        prompt_desc = input("Enter Image Description to verify: ").strip()
        if prompt_desc:
            description = prompt_desc

    if not description:
        description = "A cat walking on top of a wooden fence"

    try:
        resolved_img = resolve_image_path(image_path)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    verifier = DescriptionHallucinationVerifier()

    if args.compare:
        print("\n>>> COMPARISON MODE: BASE MODEL vs. PROPOSED ICD MODEL <<<")
        orig_pil = Image.open(resolved_img).convert("RGB")
        res_base = verifier.verify_description(orig_pil, description, model_mode="base")
        res_prop = verifier.verify_description(orig_pil, description, model_mode="proposed")

        print(f"{'Metric':<25} | {'Base Model':<22} | {'Proposed ICD (ACL 2024)':<25}")
        print("-" * 78)
        print(f"{'Overall Verdict':<25} | {res_base['verdict']:<22} | {res_prop['verdict']:<25}")
        print(f"{'CHAIR_i Hallucination':<25} | {str(res_base['hallucination_rate_chair_i'])+'%':<22} | {str(res_prop['hallucination_rate_chair_i'])+'%':<25}")
        print(f"{'Grounded Objects':<25} | {str(len(res_base['verified_objects']))+' detected':<22} | {str(len(res_prop['verified_objects']))+' verified':<25}")
        print(f"{'Hallucinated':<25} | {str(res_base['hallucinated_objects']):<22} | {str(res_prop['hallucinated_objects']):<25}")
        print("=" * 78)

    check_hallucination_and_pope(
        image_path=resolved_img,
        description=description,
        model_mode=args.model,
        run_pope=not args.no_pope,
        save_visual=not args.no_visual,
        verifier_instance=verifier
    )


if __name__ == "__main__":
    main()
