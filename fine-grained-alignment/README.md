# Fine-Grained Patch Alignment (Laptop-Scale Prototype)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-MPS%20Accelerated-green.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-success.svg)]()

A lightweight, self-contained implementation of **Fine-Grained Vision-Language Patch Alignment**, developed as **Module 2** of our university seminar and research project on **Multimodal Reinforcement Learning (MMRL) and Hallucination Reduction in Vision-Language Models (VLMs)**.

---

## 1. Project Context & Architecture

Multimodal Large Language Models (MLLMs) frequently suffer from **object hallucinations** (generating objects not present in the visual scene or misattributing spatial/semantic attributes). In our research project, this challenge is tackled across three complementary modules:

1. **Token Pruning** (`token-pruning/`): Dynamic visual token budget reduction to eliminate redundant visual background and accelerate inference.
2. **Fine-Grained Alignment** (`fine-grained-alignment/` — **This Module**): Direct patch-level grounding between visual token features and linguistic word embeddings, correcting the weak correspondence caused by coarse caption-level training.
3. **Adaptive Grounding** (`adaptive-grounding/`): Dynamic spatial grounding linking queries to relevant visual regions during inference.

This module is designed to be self-contained and cleanly modular, allowing its aligned projector weights and token representations to interface downstream with pruning and grounding.

---

## 2. Foundations & Citations

This prototype implements a scoped-down version of Patch-Aligned Training from:

> **Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models**  
> *Jiachen Jiang, Jinxin Zhou, Bo Peng, Xia Ning, Zhihui Zhu*  
> *NeurIPS 2025* (arXiv: [2505.17316](https://arxiv.org/abs/2505.17316))  
> 
> **Core Finding:** Standard MLLMs trained solely on caption generation loss ($\mathcal{L}_{\text{caption}}$) suffer from weak patch-word correspondence, leading to severe visual hallucinations.  
> **Proposed Solution:** Patch-Aligned Training introduces an auxiliary patch-alignment loss ($\mathcal{L}_{\text{patch}}$) pulling visual patch embeddings directly towards target word embeddings with a dynamic $\beta$ schedule.

The architectural connector follows the design conventions established by:

> **Visual Instruction Tuning (LLaVA)**  
> *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee*  
> *NeurIPS 2023* (GitHub: [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA))  
> 
> Utilizing LLaVA's 2-layer MLP connector design (`Linear` $\to$ `GELU` $\to$ `Linear`), keeping the vision encoder and language model embedding layer frozen.

---

## 3. Methodology & Architecture

```
                 [ Input Image (224x224) ]
                             │
                             ▼
              [ Frozen CLIP ViT-B/32 Encoder ]
                             │
                             ▼ (49 Spatial Patch Tokens, 768-d)
              ┌───────────────────────────────┐
              │  Trainable 2-Layer MLP        │
              │  Projector (1.18M Params)     │
              │  Linear(768) -> GELU -> Linear│
              └──────────────┬────────────────┘
                             │
                             ▼ (Projected Patch Tokens, 768-d)
                  [ Masked Patch Pooling ]
                             │
                             ▼ (v_k per instance)
                             ▲
               Cosine Similarity: cos(v_k, w_k)
                             ▼
                             │
                  [ Word Embedding Pooling ]
                             ▲
                             │ (w_k per instance, 768-d)
              [ Frozen GPT-2 Embedding Table ]
                             ▲
                             │
                 [ Target Category Labels ]
```

### Key Components:
- **Frozen Vision Encoder (`model/vision_encoder.py`):** OpenCLIP ViT-B/32 (`openai` pretrained) extracting $7 \times 7 = 49$ spatial patch tokens of dimension 768 (excluding the CLS token). Parameters: 87.8M (0 trainable).
- **Frozen Word Embeddings (`model/word_embeddings.py`):** GPT-2 small token embedding table (`wte`, $50,257 \times 768$). Multi-subtoken category names (e.g., `"tennis racket"` $\to$ `['ten', 'nis', ' racket']`) are tokenized and mean-pooled across subtokens. Parameters: 38.6M (0 trainable).
- **Trainable 2-Layer MLP Projector (`model/projector.py`):** LLaVA-style projector connector:
  $$\text{Linear}(768, 768) \longrightarrow \text{GELU}() \longrightarrow \text{Linear}(768, 768)$$
  **This is the sole trainable module** ($1,181,184$ trainable parameters).
- **Patch Alignment Loss (`losses/patch_alignment_loss.py`):** Implements Eq. 5 from the paper:
  $$v_k = \frac{\sum_{p=1}^N M_{k, p} V_p}{\sum_{p=1}^N M_{k, p}}, \quad \mathcal{L}_{\text{patch}} = \frac{1}{K} \sum_{k=1}^K \left(1 - \frac{v_k \cdot w_k}{\|v_k\|_2 \|w_k\|_2}\right)$$
- **Dynamic $\beta$ Schedule:** Linear ramp $\beta(t) = 0.0 \to 5.0$ over training steps, reflecting the paper's best-performing ablation setting.

---

## 4. Scope & Deliberate Simplifications

To create an effective, reproducible prototype runnable on Apple Silicon (M1 Mac) in **under 10 minutes**, several deliberate scoping choices were made:

### What Is Built:
1. Frozen ViT-B/32 extracting patch-level (not pooled) spatial tokens.
2. Frozen GPT-2 small word-embedding table with subtoken mean-pooling.
3. 2-layer MLP projector as the single trainable component.
4. Patch-alignment loss ($\mathcal{L}_{\text{patch}}$) with masked pooling and linear $\beta$ schedule.
5. COCO val2017 subset: 400 images with 2–8 instances each (1,569 total instances), split 80/20 train/val.
6. Three evaluation metrics: $\text{Align}(V, W)$ (IoU localization), Von Neumann entropy (spectral compression), and Matching Pursuit decomposition (vocabulary interpretability).

### What Is Deliberately Skipped (and Justification):
1. **Auto-Annotation Pipeline (RAM $\to$ Grounding DINO $\to$ SAM):** Skipped because COCO val2017 provides high-quality ground-truth instance segmentation masks and category labels directly, removing external annotation noise and computational overhead.
2. **Generative Caption Loss ($\mathcal{L}_{\text{caption}}$) & Full LLM Decoder:** Skipped because training the projector on $\mathcal{L}_{\text{patch}}$ alone cleanly isolates the patch-alignment contribution without needing autoregressive token generation.
3. **558K-Scale Pretraining & 7B LLM Backbone:** Scaled down to ViT-B/32 + GPT-2 embeddings + 400 COCO images to fit local hardware constraints while demonstrating clear statistical and qualitative convergence.
4. **Downstream Heavy Benchmarks (POPE, GQA, MME):** Evaluated directly on fine-grained metrics ($\text{Align}(V, W)$, Von Neumann entropy, Matching Pursuit) targeting patch-level semantic fidelity.

---

## 5. Experimental Results & Verification

### 5.1 Quantitative Results Table (Held-Out Validation Set)

All metrics evaluated on 80 held-out validation images (310 object instances) comparing the untrained/randomly-initialized projector baseline against the trained projector:

| Evaluation Metric | Baseline (Untrained) | Trained Projector | Delta ($\Delta$) | Impact & Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **$\text{Align}(V, W)$ Mean IoU** | 0.1069 | **0.2051** | **+0.0982** | **+91.9% relative increase** in patch localization accuracy |
| **$\text{Align}(V, W)$ Acc @ $\text{IoU} \ge 0.50$** | 1.29% | **11.29%** | **+10.00%** | **+875.2% relative gain** in high-overlap patch grounding |
| **On-Target Cosine Similarity** | -0.0170 | **+0.6307** | **+0.6477** | Visual patches strongly align with object words |
| **Background Cosine Similarity** | -0.0203 | **+0.4232** | +0.4435 | Distinct foreground-background separation (+0.2075 gap) |
| **Von Neumann Entropy $S(\rho)$** | 2.0169 | **1.5675** | **-0.4494** | Semantic structuring onto lower-dimensional manifold |
| **Entropy Compression Ratio** | 0.9794 | **0.7612** | **-0.2182** | Effective compression across projector transformation |
| **Matching Pursuit GT Match Acc** | 0.00% | **60.97%** | **+60.97%** | Majority of patches match GT label top-1 |
| **Matching Pursuit Top-1 Word Sim** | -0.0716 | **+0.8486** | **+0.9202** | High correlation against candidate vocabulary |
| **Matching Pursuit Residual Ratio** | 0.9889 | **0.3718** | **-0.6171** | **62.8% of patch energy captured** by visual words |

### 5.2 Qualitative Matching Pursuit Decompositions

Matching Pursuit greedily decomposes patch embeddings into atoms from a candidate dictionary (80 COCO categories + visual attributes). Notice the dramatic shift from garbled noise to coherent semantic labels:

| Sample | Ground Truth Object | Baseline (Untrained Projector) | Trained Projector | Result |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **`truck`** | `wine glass (-0.77)`, `refrigerator (+0.87)` *(cos = -0.0655)* | **`truck (+10.49)`**, `person (+2.51)` *(cos = **0.9152**)* | **Exact Grounding** |
| **2** | **`person`** | `horse (-1.42)`, `umbrella (+1.63)` *(cos = -0.0879)* | **`person (+8.71)`**, `cow (+3.26)` *(cos = **0.7835**)* | **Exact Grounding** |
| **3** | **`person`** | `grass (-0.89)`, `elephant (+0.95)` *(cos = -0.0733)* | **`person (+21.73)`**, `teddy bear (-1.40)` *(cos = **0.9895**)* | **Exact Grounding** |
| **4** | **`tennis racket`** | `white (-1.50)`, `mouse (+1.08)` *(cos = -0.1006)* | **`tennis racket (+22.47)`**, `frisbee (+2.80)` *(cos = **0.9427**)* | **Exact Grounding** |
| **5** | **`person`** | `white (-0.90)`, `elephant (+1.14)` *(cos = -0.0806)* | **`person (+12.54)`**, `baseball glove (+2.31)` *(cos = **0.9629**)* | **Exact Grounding** |

### 5.3 Training Dynamics & Hardware Benchmarks

- **Target Device:** MacBook Air M1 (Apple Silicon MPS backend).
- **Epoch 1 Duration:** 15.04s.
- **Total Training Duration (25 Epochs):** **322.2s (5.37 minutes)**.
- **Convergence:** Training loss dropped monotonically from **0.5326** to **0.0222**; Validation loss reached **0.2624**.
- **Curve Plot:** Saved to `results/training_loss_curve.png`.

---

## 6. Repository Structure

```
fine-grained-alignment/
├── data/
│   ├── prepare_coco_subset.py      # Download annotations, filter 2-8 instances, sample 400 images
│   └── patch_dataset.py            # PyTorch Dataset with mask downsampling to 7x7 patch grids
├── model/
│   ├── vision_encoder.py           # Frozen OpenCLIP ViT-B/32 patch token extractor (49x768)
│   ├── projector.py                # Trainable LLaVA-style 2-layer MLP projector (1.18M params)
│   └── word_embeddings.py          # Frozen GPT-2 word embedding table with subtoken pooling
├── losses/
│   └── patch_alignment_loss.py     # L_patch (Eq. 5) masked pooling loss & linear beta schedule
├── train.py                        # Training loop on MPS backend with timing benchmarks & logging
├── eval/
│   ├── align_vw.py                 # Align(V, W) IoU patch-localization evaluation
│   ├── entropy.py                  # Von Neumann entropy calculation before vs after projector
│   ├── matching_pursuit.py         # Matching pursuit sparse vocabulary decomposition
│   ├── run_baseline_eval.py        # Baseline evaluation runner on untrained projector
│   └── run_final_eval.py           # Final evaluation runner & comparative summary generator
├── configs/
│   └── default.yaml                # Hyperparameters, paths, and architecture configuration
├── checkpoints/                    # Saved model checkpoints (git-ignored)
├── results/                        # Baseline metrics, trained metrics, comparisons, and loss curves
├── sanity_checks/                  # 5 COCO overlay PNGs and unit verification scripts
└── README.md                       # Comprehensive documentation
```

---

## 7. Step-by-Step Reproduction Guide

To reproduce all results from scratch:

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/VDeepak05/MMRL_Hallucination_Reduction.git
cd MMRL_Hallucination_Reduction/fine-grained-alignment

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision transformers open_clip_torch pycocotools pillow numpy matplotlib scikit-learn pyyaml tqdm
```

### 2. Prepare COCO Dataset Subset
```bash
python data/prepare_coco_subset.py
# Downloads instances_val2017.json, filters 2-8 instances, samples 400 images,
# creates data/coco_subset_manifest.json and 5 sanity check images in sanity_checks/
```

### 3. Run Phase Sanity Verifications
```bash
# Verify frozen components and mask downsampling
python sanity_checks/check_phase2.py

# Verify trainable projector and parameter freezing audit
python sanity_checks/check_phase3.py

# Verify L_patch loss, beta schedule, and identity sanity check (loss ≈ 0)
python sanity_checks/check_phase4.py
```

### 4. Run Untrained Baseline Evaluation
```bash
python eval/run_baseline_eval.py
# Computes Align(V,W), Von Neumann entropy, and Matching Pursuit on untrained projector
# Output saved to results/baseline_metrics.json
```

### 5. Train the Projector
```bash
python train.py
# Runs 25 epochs on device="mps" (~5 minutes runtime)
# Saves best checkpoint to checkpoints/projector_best.pt
# Generates results/training_loss_curve.png
```

### 6. Run Final Comparative Evaluation
```bash
python eval/run_final_eval.py
# Evaluates trained projector and generates:
# - results/trained_metrics.json
# - results/comparison_summary.json
# - results/comparison_table.md
```
