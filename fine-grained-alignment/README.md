# Fine-Grained Patch Alignment (Laptop-Scale Prototype)

A lightweight, self-contained implementation of fine-grained vision-language patch alignment, developed as part of a larger research project on **Multimodal Reinforcement Learning (MMRL) and Hallucination Reduction in Vision-Language Models (VLMs)**.

## Project Context

This module represents **Module 2: Fine-Grained Alignment** in a 3-part modular architecture:
1. **Token Pruning** — Visual token budget reduction for efficient inference.
2. **Fine-Grained Alignment** (This Module) — Enforcing explicit patch-level token to word-embedding correspondence to prevent semantic drift and hallucination.
3. **Adaptive Grounding** — Dynamic grounding of queries to image regions.

---

## Foundations & Citations

This prototype implements a scoped-down version of Patch-Aligned Training from:

> **Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models**  
> *Jiachen Jiang, Jinxin Zhou, Bo Peng, Xia Ning, Zhihui Zhu*  
> *NeurIPS 2025* (arXiv: [2505.17316](https://arxiv.org/abs/2505.17316))  
> 
> *Diagnosis:* Standard MLLMs trained solely on caption generation loss ($\mathcal{L}_{\text{caption}}$) suffer from weak patch-word correspondence, leading to object hallucinations.  
> *Remedy:* Patch-Aligned Training introduces an auxiliary patch-alignment loss ($\mathcal{L}_{\text{patch}}$) pulling visual patch embeddings directly towards target word embeddings with a dynamic $\beta$ schedule.

The architectural connector adheres to the projector conventions established by:

> **Large Language and Vision Assistant (LLaVA)**  
> *Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee*  
> *NeurIPS 2023* (GitHub: [LLaVA](https://github.com/haotian-liu/LLaVA))  
> 
> Following LLaVA's 2-layer MLP connector design (`Linear` $\to$ `GELU` $\to$ `Linear`), keeping the vision encoder and LLM embedding layer frozen.

---

## Scope & Intentional Simplifications

To allow rapid iteration and execution on Apple Silicon (M1 Mac, unified memory, MPS backend) in well under an hour, several deliberate scoping choices were made:

### What Is Built:
- **Frozen Vision Encoder:** OpenCLIP ViT-B/32 extracting patch-level embeddings ($7 \times 7 = 49$ spatial tokens, dimension 768) rather than only the global pooled token.
- **Frozen Word Embeddings:** GPT-2 small embedding matrix (dimension 768). Label phrases are tokenized and mean-pooled across subtokens.
- **Trainable 2-Layer MLP Projector:** The sole trainable component ($768 \to 768 \to 768$ with GELU activation), aligning visual patch tokens into the language embedding space.
- **Patch-Alignment Loss ($\mathcal{L}_{\text{patch}}$):** Masks visual patches via downsampled ground-truth instance masks, pools patch tokens per instance, and minimizes $1 - \text{cosine\_similarity}(v_i, w_i)$.
- **Dynamic $\beta$ Schedule:** Linear ramp from $\beta = 0$ to $\beta = 5$ across training iterations.
- **Dataset:** ~400 images sampled from COCO val2017 with 2–8 instances per image (~1,500–2,000 instances total), split 80/20 train/validation.
- **Three Verification & Evaluation Metrics:**
  1. $\text{Align}(V, W)$ (Eq. 4 in the paper) — Patch localization accuracy via IoU.
  2. **Von Neumann Entropy** (Eq. 1 in the paper) — Visual embedding compression / spectral spread across projector transformation.
  3. **Matching Pursuit Decomposition** — Iterative sparse decomposition of patch tokens against a candidate vocabulary for qualitative interpretability.

### What Is Deliberately Skipped (and Rationale):
- **RAM + Grounding DINO + SAM Auto-annotation Pipeline:** Skipped because COCO val2017 provides ground-truth segmentation masks and category labels directly, removing external annotation noise and computational overhead.
- **Full Caption Loss ($\mathcal{L}_{\text{caption}}$) & Generative LLM:** Skipped because training the projector on $\mathcal{L}_{\text{patch}}$ alone cleanly isolates the patch-alignment contribution without needing full autoregressive generation.
- **558K-scale Pretraining & 7B LLM Backbone:** Scaled down to ViT-B/32 + GPT-2 embeddings + 400 COCO images to fit Apple Silicon hardware budget (< 1 hour runtime).
- **Downstream Benchmarks (GQA, POPE, RefCOCO):** Evaluated directly on fine-grained metrics ($\text{Align}(V, W)$, Von Neumann entropy, Matching Pursuit) targeting patch-level semantic fidelity.

---

## Repository Structure

```
fine-grained-alignment/
├── data/
│   ├── prepare_coco_subset.py    # Downloads/subsets COCO val2017 instances & creates manifest
│   └── patch_dataset.py          # PyTorch Dataset for images, masks, and category labels
├── model/
│   ├── vision_encoder.py         # Frozen CLIP ViT-B/32 patch extractor
│   ├── projector.py              # Trainable 2-layer MLP projector
│   └── word_embeddings.py        # Frozen language embedding table & token pooling
├── losses/
│   └── patch_alignment_loss.py   # L_patch loss with masked pooling and beta schedule
├── train.py                      # Training loop with MPS support, timing & logging
├── eval/
│   ├── align_vw.py               # Align(V, W) IoU patch-localization evaluation
│   ├── entropy.py                # Von Neumann entropy calculation
│   └── matching_pursuit.py       # Matching pursuit sparse vocabulary decomposition
├── configs/
│   └── default.yaml              # Hyperparameters and paths
├── checkpoints/                  # Model weights (git-ignored)
├── results/                      # Baseline and trained evaluation metrics
├── sanity_checks/                # Visualizations and sanity verification outputs
└── README.md                     # Documentation
```

---

## Hardware & Environment

- **Target Architecture:** Apple Silicon M1 (MPS backend: `device = "mps"`).
- **Environment Setup:**
  ```bash
  source ../.venv/bin/activate
  # Verify MPS
  python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
  ```
