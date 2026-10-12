# Implementation Plan: Adaptive, Fine-Grained, Evidence-Grounded Vision-Language Model

## Executive Summary
This document outlines the architecture, implementation blueprint, and code provenance strategy for **Research Question 3 (RQ3: Visual Evidence Verification & Grounded Generation)**, designed as a modular framework that scales seamlessly into **RQ1 (Adaptive Visual Acquisition)**, **RQ2 (Fine-Grained Alignment)**, and **RQ4 (Joint Optimization)**.

---

## 1. Code Provenance Strategy (GitHub vs Custom Generation)

### Strategy: Lightweight Hybrid Modular Implementation
- **Backbone & Library Standards (Hugging Face / PyTorch)**: We utilize official, maintained open-source libraries (`transformers`, `torch`, `torchvision`, `accelerate`) supporting **Lightweight VLMs** (such as `HuggingFaceTB/SmolVLM-Instruct` ~1.5 GB, `Qwen/Qwen2-VL-2B-Instruct` ~2.5 GB, or `vikhyatk/moondream2` ~1.8 GB).
- **Why Lightweight Models?**
  1. **Efficiency**: 1.5 GB weights download in seconds and run comfortably on consumer GPUs / laptop hardware without VRAM crashes or multi-gigabyte bloat.
  2. **Identical Functional Capability**: Provides full visual grounding, object detection, and visual token output for contrastive evidence verification (RQ3).
  3. **Universal Adapter**: `VLMModelWrapper` uses Hugging Face `AutoModelForVision2Seq`, allowing effortless switching between `SmolVLM`, `Qwen2-VL`, `Moondream2`, or legacy `LLaVA-7B` via `config.py`.


---

## 2. Project Architecture & File Structure

```
project/
├── IMPLEMENTATION_PLAN.md          # Complete implementation blueprint (this file)
├── PROGRESS_LOG.md                 # State tracking & checkpoint log for recovery
├── requirements.txt                # Python dependencies
├── config.py                       # Global configuration & hyperparameters
├── models/
│   ├── __init__.py
│   ├── llava_wrapper.py            # HuggingFace LLaVA model loader & forward hooks
│   └── contrastive_decoder.py      # Core RQ3 Contrastive Decoding & Logit Calibration
├── perturbations/
│   ├── __init__.py
│   └── visual_perturbation.py      # Gaussian noise & targeted token masking
├── pipeline/
│   ├── __init__.py
│   └── grounded_pipeline.py        # Unified pipeline (Hooks for RQ1, RQ2, RQ3, RQ4)
└── benchmarks/
    ├── __init__.py
    ├── pope_loader.py              # POPE benchmark dataset loader
    └── eval_pope.py                # Quantitative evaluation script (Acc, F1, Yes-Ratio)
```

---

## 3. Mathematical & Algorithmic Formulation of RQ3

RQ3 addresses visual hallucination by contrasting logits from original visual evidence $V$ against distorted visual evidence $V_{\text{pert}}$:

$$\text{Logits}_{\text{calibrated}}(y_t) = (1 + \alpha) \cdot \text{Logits}(y_t \mid V, Q, y_{<t}) - \alpha \cdot \text{Logits}(y_t \mid V_{\text{pert}}, Q, y_{<t})$$

### Adaptive Candidate Truncation (Adaptive Cutoff $\beta$)
To prevent penalizing fluent language syntax (e.g., articles, prepositions), contrastive decoding is only applied to candidate tokens whose original probability exceeds an adaptive threshold $\beta$:

$$\mathcal{V}_{\text{head}} = \{ y \in \mathcal{V} \mid P(y \mid V, Q, y_{<t}) \ge \beta \cdot \max_{w} P(w \mid V, Q, y_{<t}) \}$$

Tokens outside $\mathcal{V}_{\text{head}}$ retain their original logit values.

---

## 4. Integration Design across RQs

```
[Input Image V & Question Q]
         │
         ▼
 ┌───────────────┐
 │  RQ1 Module   │ ──(Token Selection: Prunes visual tokens)
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  RQ2 Module   │ ──(Fine-Grained Alignment: Computes token saliency map)
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  RQ3 Module   │ ──(Grounded Decoding: Perturbs visual tokens + Calibrates logits)
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  RQ4 Module   │ ──(Joint Benchmark: Evaluates POPE Accuracy/Efficiency Tradeoff)
 └───────────────┘
```

---

## 5. Execution Checkpoints & Milestones

| Milestone | Component | Description | Status |
| :--- | :--- | :--- | :--- |
| **M1** | Documentation & Plan | Create `IMPLEMENTATION_PLAN.md` and `PROGRESS_LOG.md` | In Progress |
| **M2** | Environment Setup | Create `requirements.txt` and `config.py` | Pending |
| **M3** | Model & Perturbation | Implement `llava_wrapper.py` & `visual_perturbation.py` | Pending |
| **M4** | RQ3 Core Engine | Implement `contrastive_decoder.py` (VCD Logit Processor) | Pending |
| **M5** | Unified Pipeline | Implement `grounded_pipeline.py` with RQ1/RQ2 hooks | Pending |
| **M6** | Benchmark & Eval | Implement POPE benchmark loader & evaluation runner | Pending |
| **M7** | Final Verification | Dry run evaluation & generate output report | Pending |

---

## 6. Recovery & Token Safety Protocol
After completing each milestone, `PROGRESS_LOG.md` will be updated with:
1. Completed component details.
2. File paths created.
3. Verification results.
4. Next exact step to execute if context/session resets.
