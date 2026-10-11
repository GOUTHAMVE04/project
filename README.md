# Adaptive, Fine-Grained, Evidence-Grounded Vision-Language Model (VLM) Framework

> **Research Question 3 (RQ3): Visual Evidence Verification & Anti-Hallucination**  
> An evidence-grounded pipeline that detects whether image descriptions contain **visual hallucinations** using Visual Contrastive Decoding (VCD), Woodpecker concept probing, and CHAIR metrics.

---

## Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [Research Papers Used](#research-papers-used)
3. [Project Structure](#project-structure)
4. [Requirements & Setup](#requirements--setup)
5. [How to Run — All Execution Modes](#how-to-run--all-execution-modes)
   - [Quick Demo (Mock Mode)](#1-quick-demo-mock-mode)
   - [Image Description Hallucination Verifier](#2-image-description-hallucination-verifier)
   - [POPE Benchmark Evaluation](#3-pope-benchmark-evaluation)
6. [Perturbed Image Visualization](#perturbed-image-visualization)
7. [How Each Component Works](#how-each-component-works)
8. [Configuration Options](#configuration-options)
9. [Understanding the Output](#understanding-the-output)
10. [Switching to Real Model Weights](#switching-to-real-model-weights)

---

## What This Project Does

Modern Vision-Language Models (VLMs) often generate **hallucinated content** — they describe objects that are not actually present in an image because they rely on language memory (priors) rather than truly looking at the image.

**This framework detects hallucinations in image descriptions by:**

1. **Extracting all physical object claims** from the text description.
2. **Running two forward passes** through the vision encoder:
   - One on the **clean original image** → Real Evidence Score.
   - One on a **Gaussian-noise-distorted image** → Hallucinated Prior Score.
3. **Subtracting** the noise-based score from the clean score (Visual Contrastive Decoding).
4. **Flagging hallucinated objects** and reporting a CHAIR hallucination rate.

**Example:**
```
Image: A cat sitting on a wooden fence in a garden
Description: "A cat walking through a lake"

→ cat:       VERIFIED PRESENT   (Clean: 0.85 | Perturbed: 0.49)
→ lake:      HALLUCINATED ABSENT (Clean: 0.00 | Perturbed: 0.00)

Overall Verdict: HALLUCINATED | CHAIR_i = 50.0%
```

---

## Research Papers Used

| Paper | Contribution in This Project |
|:------|:-----------------------------|
| **VCD** — *Leng et al., CVPR 2024*<br>"Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding" | Core contrastive decoding formula: `(1+α)·Logits_clean − α·Logits_perturbed` and Gaussian visual perturbation |
| **Woodpecker** — *Yin et al., CVPR 2024*<br>"Woodpecker: Hallucination Correction for Multimodal LLMs" | Multi-stage claim extraction and binary visual probing pipeline |
| **CHAIR** — *Rohrbach et al., EMNLP 2018*<br>"Object Hallucination in Image Captioning" | `CHAIR_i` (Instance Hallucination Rate) metric: `hallucinated_objects / total_objects × 100%` |
| **POPE** — *Li et al., EMNLP 2023*<br>"Evaluating Object Hallucination in Large Vision-Language Models" | POPE benchmark loader and binary Yes/No evaluation splits (random, popular, adversarial) |
| **CLIP** — *Radford et al., ICML 2021*<br>"Learning Transferable Visual Models From Natural Language Supervision" | Frozen vision encoder backbone for visual evidence extraction |
| **LLaVA / LLaVA-1.5** — *Liu et al., NeurIPS 2023 / CVPR 2024* | Base VLM architecture wrapped by `VLMModelWrapper` |
| **DoLa** — *Chuang et al., ICLR 2024* | Adaptive cutoff threshold β for candidate truncation in contrastive decoding |
| **MHal-Control** — *Favero et al., CVPR 2024* | Competitive reference anchors for zero-shot visual disambiguation |

---

## Project Structure

```
project/
├── README.md                         # This file
├── requirements.txt                  # Python package dependencies
├── config.py                         # All hyperparameters (α, β, noise_std, model ID)
├── main.py                           # Master entry point (full pipeline + POPE benchmark)
├── verify_description.py             # *** IMAGE HALLUCINATION CHECKER (main tool) ***
│
├── models/
│   ├── llava_wrapper.py              # VLM wrapper (SmolVLM / LLaVA / Moondream / Mock)
│   └── contrastive_decoder.py        # RQ3 VCD core engine: dual forward pass + calibration
│
├── perturbations/
│   └── visual_perturbation.py        # Gaussian noise generator (PIL + Tensor modes)
│
├── pipeline/
│   ├── grounded_pipeline.py          # Unified pipeline with RQ1/RQ2 hooks
│   └── hallucination_verifier.py     # Proposed Model: CLIP + VCD + CHAIR verifier
│
├── benchmarks/
│   ├── pope_loader.py                # POPE dataset loader (random/popular/adversarial)
│   └── eval_pope.py                  # Evaluation engine: Accuracy, Precision, Recall, F1
│
├── data/
│   └── pope/                         # Auto-generated POPE benchmark samples
│
└── results/
    └── pope_evaluation_results.json  # Saved evaluation results
```

---

## Requirements & Setup

### Step 1: Install Python 3.10 or 3.12
Download from [python.org](https://www.python.org/downloads/).

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The key packages installed:
| Package | Version | Purpose |
|:--------|:--------|:--------|
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `transformers` | ≥ 4.37.0 | HuggingFace models (CLIP, SmolVLM, LLaVA) |
| `Pillow` | ≥ 9.0.0 | Image loading and manipulation |
| `numpy` | ≥ 1.22.0 | Numerical operations and noise generation |
| `accelerate` | ≥ 0.26.0 | Multi-GPU and CPU memory optimization |

### Step 3: (First-Time Only) Models will Auto-Download
The first time you run the hallucination verifier, it will automatically download:
- **`openai/clip-vit-base-patch32`** (~600 MB) — Vision encoder for evidence verification.
- No login or API key is required.

---

## How to Run — All Execution Modes

### 1. Quick Demo (Mock Mode)

Runs the full framework instantly with simulated model outputs — no large weight downloads needed.

```bash
python main.py
```

**What it does:**
- Initializes the Grounded VLM Pipeline.
- Registers RQ1 (Token Selection) and RQ2 (Saliency Alignment) hooks.
- Runs a sample inference: `"Is there a dog in the image?"`
- Evaluates the POPE benchmark on the `random` split.
- Saves results to `results/pope_evaluation_results.json`.

**Expected Output:**
```
================================================================
   Grounded VLM Framework (RQ3 Core Engine & RQ1/2/4 Scalability)  
================================================================
[Step 1] Initializing Grounded VLM Inference Pipeline...
[VLMModelWrapper] Running in INSTANT DEMO MODE (USE_MOCK_MODEL = True)

[Inference Result]:
  Question:      Is there a dog in the image?
  Response:      Yes, object verified via visual evidence contrast.
  RQ3 Grounded:  True
  Latency (sec): 0.0

POPE Evaluation Summary:
  Baseline LLaVA-1.5: Accuracy = 50.0%, F1 = 66.67%, Yes-Ratio = 100.0%
  RQ3 Grounded VCD:   Accuracy = 50.0%, F1 = 66.67%, Yes-Ratio = 100.0%
```

---

### 2. Image Description Hallucination Verifier

The main research tool. Upload any image and provide any text description — it tells you which objects are genuinely in the image and which are hallucinated.

#### Basic Usage

```bash
python verify_description.py --image "path/to/your/image.jpg" --description "Your description here"
```

#### Examples

**Example A — Faithful Description (should show GROUNDED):**
```bash
python verify_description.py --image "images (1).jpg" --description "A cat walking on top of a wooden fence"
```

**Example B — Hallucinated Description (should show HALLUCINATED):**
```bash
python verify_description.py --image "images (1).jpg" --description "A dog swimming in a lake with an elephant"
```

**Example C — Partial Hallucination:**
```bash
python verify_description.py --image "images (1).jpg" --description "A cat walking on a fence next to a red bicycle"
```

**Example D — Compare Base Model vs Proposed Model side-by-side:**
```bash
python verify_description.py --image "images (1).jpg" --description "A dog next to a bicycle" --compare
```

**Example E — Use Base Model only (uncalibrated, shows hallucination problem):**
```bash
python verify_description.py --image "images (1).jpg" --description "A dog next to a bicycle" --model base
```

#### All Available Arguments

| Argument | Short | Default | Description |
|:---------|:------|:--------|:------------|
| `--image` | `-i` | Auto-detect | Path to image file (JPG/PNG). Supports paths with spaces. |
| `--description` | `-d` | Demo text | The description text to verify for hallucinations. |
| `--model` | `-m` | `proposed` | `proposed` (RQ3 VCD Grounded) or `base` (Uncalibrated Baseline). |
| `--compare` | `-c` | Off | When set, runs both models and prints a side-by-side comparison table. |
| `--no-visual` | — | Off | Skip saving the perturbed comparison image (console output only). |

---

### 3. POPE Benchmark Evaluation

Runs the full hallucination benchmark from Li et al. (EMNLP 2023) across multiple difficulty splits.

```bash
python main.py
```

Results are saved to `results/pope_evaluation_results.json`.

To view saved results:
```bash
type results\pope_evaluation_results.json
```

---

## Perturbed Image Visualization

Every time you run `verify_description.py`, two image files are **automatically saved** to the `results/` folder:

| File | Description |
|:-----|:------------|
| `results/<image_name>_perturbed_comparison.png` | **Side-by-side panel**: Original \| Perturbed \| Claim Audit. Color-coded verdict header (🔴 red = hallucinated, 🟢 green = grounded). |
| `results/<image_name>_perturbed_raw.png` | The raw Gaussian-noise-perturbed image only (V_pert used internally by VCD). |

### What the Comparison Image Shows

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RED/GREEN HEADER: VERDICT: HALLUCINATED | CHAIR_i Hallucination: 50.0%    │
├────────────────────┬───────────────────────┬────────────────────────────────┤
│  Original Image    │   Perturbed Image     │   Claim Audit Panel            │
│  (V_clean)         │   (V_pert, sigma=0.5) │                                │
│                    │                       │  Description: "..."            │
│  What is actually  │  Gaussian noise added │  Grounded Objects: [cat]       │
│  in the image      │  to erase visual      │  Hallucinated: [lake]          │
│                    │  details; VCD scores  │                                │
│                    │  measured here too    │  [PASS] cat  Clean:0.85        │
│                    │                       │  [FAIL] lake Clean:0.00        │
└────────────────────┴───────────────────────┴────────────────────────────────┘
```

**The perturbed image is critical to understand VCD:** Notice that the cat outline is faintly visible even through heavy noise — this residual signal (high Clean score, lower Perturbed score) is what VCD uses to confirm the object is genuinely present. Objects that never existed (like `lake`) score near-zero on both clean and perturbed passes.

### Skip Image Generation

If you only want console output and no image files:

```bash
python verify_description.py --image "images (1).jpg" --description "Your text" --no-visual
```

---

## How Each Component Works

### A. Visual Perturbation (Gaussian Noise)
**File:** `perturbations/visual_perturbation.py`

```
Clean Image  →  Add noise N(0, 0.5²)  →  Perturbed Image
```
Objects that are genuinely present lose signal when noise is added.
Objects that are hallucinated (language guesses) stay the same.

### B. Contrastive Calibration
**File:** `models/contrastive_decoder.py`

```
Calibrated Score = (1 + α) × Clean_Score − α × Perturbed_Score
                 = (1 + 1.0) × Clean_Score − 1.0 × Perturbed_Score
```
α = 1.0 (set in `config.py`). Higher α = stronger anti-hallucination penalty.

### C. Competitive Reference Anchors
**File:** `pipeline/hallucination_verifier.py`

Rather than checking `"dog"` vs `"empty background"`, the system compares against 6 competitive visual categories (cat, fence, car, bicycle, tree, person...) so the result is meaningful.

### D. CHAIR_i Metric
```
CHAIR_i = (Hallucinated Objects / Total Objects Mentioned) × 100%
```
- `0%` = Perfectly faithful description.
- `100%` = Every object mentioned is hallucinated.

---

## Configuration Options

All settings are in `config.py`. Key parameters:

```python
# Switch between instant demo mode and real model mode
USE_MOCK_MODEL = True   # True = instant, False = downloads real VLM weights

# Model to use when USE_MOCK_MODEL = False
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"  # ~500MB, fast
# Alternatives:
# "HuggingFaceTB/SmolVLM-Instruct"    # ~1.5GB, better quality
# "vikhyatk/moondream2"               # ~1.8GB
# "llava-hf/llava-1.5-7b-hf"          # ~14GB, best quality

# RQ3 Contrastive Decoding Hyperparameters
VCD_ALPHA = 1.0    # Contrastive amplification coefficient (higher = stronger penalty)
VCD_BETA  = 0.1    # Adaptive cutoff threshold (only top tokens are calibrated)
NOISE_STD = 0.5    # Gaussian noise strength for visual perturbation
```

---

## Understanding the Output

```
[PROPOSED MODEL VERIFICATION RESULTS]
  Model Engine:          Proposed Model (RQ3 Evidence-Grounded VCD)
  Overall Verdict:       HALLUCINATED
  Is Hallucinated:       YES (Hallucination Detected)
  CHAIR_i Hallucination: 50.0%                  ← 50% of mentioned objects are fake
  Total Claims Checked:  2

  [Grounded in Image]:     ['cat']              ← These objects are real
  [Hallucinated / Absent]: ['lake']             ← These do NOT exist in the image

[ANNOTATED DESCRIPTION]
  "A cat walking through a [HALLUCINATED: lake]" ← Fake objects are marked inline

[DETAILED CLAIM AUDIT]
  [PASS] 'cat'  → Status: VERIFIED_PRESENT  | Clean: 0.8526 | Perturbed: 0.4867 | Calibrated: 26.89
  [FAIL] 'lake' → Status: HALLUCINATED_ABSENT | Clean: 0.00 | Perturbed: 0.0029 | Calibrated: 8.96
```

| Column | Meaning |
|:-------|:--------|
| `[PASS]` / `[FAIL]` | Whether the object was verified as present in the image |
| `Clean:` | Confidence score with the original, unmodified image (higher = stronger visual evidence) |
| `Perturbed:` | Confidence score with Gaussian-noisy image (reveals language prior guessing) |
| `Calibrated:` | VCD formula output: `(1+α)×Clean − α×Perturbed` (higher = stronger grounding) |

---

## Switching to Real Model Weights

To run with a real VLM instead of the mock simulation:

1. Open `config.py`
2. Change:
   ```python
   USE_MOCK_MODEL = True   →   USE_MOCK_MODEL = False
   ```
3. Optionally set a lighter model:
   ```python
   MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"  # ~500MB, fast on CPU
   ```
4. Run:
   ```bash
   python main.py
   ```
   or
   ```bash
   python verify_description.py --image "your_image.jpg" --description "Your description"
   ```

> **Note:** The first run will download model weights (~500MB to ~1.5GB). After that, they are cached locally and load in a few seconds.

---

## Quick Reference Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full demo + POPE benchmark (instant, no download)
python main.py

# 3. Check if a description is hallucinated (uses your own image)
python verify_description.py --image "images (1).jpg" --description "A cat on a fence"

# 4. Compare base model vs proposed model
python verify_description.py --image "images (1).jpg" --description "A dog on a fence" --compare

# 5. Force base (uncalibrated) model only
python verify_description.py --image "images (1).jpg" --description "A dog on a fence" --model base
```

---

*Project: Adaptive, Fine-Grained, Evidence-Grounded Vision-Language Models — RQ3 Implementation*
