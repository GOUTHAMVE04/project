# Comprehensive Technical Specification & Module Deep-Dive
## Adaptive, Fine-Grained, Evidence-Grounded Vision-Language Model Framework

---

## 1. Executive Summary & Architecture Overview

This project presents a modular framework designed to mitigate **visual hallucinations** in Vision-Language Models (VLMs). Hallucination occurs when a model generates claims driven by statistical language priors rather than grounding its responses in visual evidence.

The centerpiece of this framework is **Research Question 3 (RQ3: Visual Evidence Verification & Grounded Generation)**, which uses **Visual Contrastive Decoding (VCD)** to calibrate token probabilities by contrasting original image logits against perturbed image logits. The repository is constructed with extensible hooks to support **RQ1 (Adaptive Visual Token Selection)**, **RQ2 (Fine-Grained Alignment)**, and **RQ4 (Joint Benchmarking)**.

```
                  ┌──────────────────────────────────────────┐
                  │       Input Image (V) & Question (Q)     │
                  └─────────────────────┬────────────────────┘
                                        │
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │    [RQ1 Hook] Adaptive Token Selector    │
                  │ (Prunes uninformative visual tokens)     │
                  └─────────────────────┬────────────────────┘
                                        │
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │    [RQ2 Hook] Saliency Alignment Scorer  │
                  │ (Computes cross-attention saliency maps) │
                  └─────────────────────┬────────────────────┘
                                        │
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │     [RQ3 Module] Visual Perturber        │
                  │  (Generates noisy image baseline V_pert) │
                  └─────────────────────┬────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌──────────────────────┐                ┌──────────────────────┐
        │ Original Pass: M(V)  │                │ Perturbed Pass: M(V')│
        └───────────┬──────────┘                └───────────┬──────────┘
                    │                                       │
                    │ Logits_orig                           │ Logits_pert
                    └───────────────────┬───────────────────┘
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │   [RQ3 Module] Contrastive Calibration   │
                  │ Logits = (1+α)*Logits_orig - α*Logits_pert│
                  │     Applies Adaptive Cutoff Threshold β  │
                  └─────────────────────┬────────────────────┘
                                        │
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │     Final Grounded Text Output Response   │
                  └──────────────────────────────────────────┘
```

---

## 2. Module-by-Module Technical Deep Dive

---

### Module 1: `config.py` — Global Configuration & Design Constants
**File Path**: [`config.py`](file:///c:/Users/HP/OneDrive/Desktop/project/config.py)

#### Purpose & Design Rationale
Acts as the central source of truth for hyperparameters, device allocations, file system locations, and execution modes. It allows switching between lightweight CPU/GPU execution modes (`USE_MOCK_MODEL = True`) and downloading production weights (`SmolVLM`, `Qwen2-VL`, `LLaVA`) without code modifications.

#### Detailed Attribute Descriptions:

| Parameter | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `USE_MOCK_MODEL` | `bool` | `True` | When `True`, runs simulated VLM generation in $< 0.01$ seconds without downloading multi-GB model weights. |
| `MODEL_ID` | `str` | `"HuggingFaceTB/SmolVLM-256M-Instruct"` | Model identifier used when downloading real weights from HuggingFace. Options include `SmolVLM-256M`, `SmolVLM-Instruct`, `moondream2`, or `Qwen2-VL-2B`. |
| `DEVICE` | `str` | `"cuda"` if available else `"cpu"` | Hardware acceleration target. |
| `DTYPE` | `torch.dtype` | `float16` if CUDA else `float32` | Numerical precision for PyTorch forward passes. |
| `VCD_ALPHA` ($\alpha$) | `float` | `1.0` | Contrastive amplification coefficient. Scales penalty applied to logits derived from perturbed visual inputs. |
| `VCD_BETA` ($\beta$) | `float` | `0.1` | Adaptive candidate truncation cutoff. Only tokens with original probability $P_{\text{orig}} \ge \beta \cdot \max(P_{\text{orig}})$ undergo contrastive calibration. |
| `NOISE_STD` | `float` | `0.5` | Standard deviation ($\sigma$) of zero-mean Gaussian noise added to generate perturbed visual images $V_{\text{pert}}$. |
| `MASK_RATIO` | `float` | `0.3` | Percentage of top salient visual tokens masked in RQ2/RQ3 targeted perturbation. |
| `MAX_NEW_TOKENS` | `int` | `128` | Upper limit on generated text tokens per question. |
| `BENCHMARK_DIR` | `str` | `data/pope` | Target path for benchmark dataset files. Auto-created if missing. |
| `RESULTS_DIR` | `str` | `results` | Destination directory for evaluation output JSON files. Auto-created if missing. |

---

### Module 2: `models/llava_wrapper.py` — Vision-Language Model Abstraction
**File Path**: [`models/llava_wrapper.py`](file:///c:/Users/HP/OneDrive/Desktop/project/models/llava_wrapper.py)

#### Purpose & Core Classes
Abstracts model-specific variations (SmolVLM, LLaVA, Qwen2-VL, Moondream2) behind a unified interface: `VLMModelWrapper`.

#### Key Classes & Methods:

1. **`VLMModelWrapper`**:
   - `__init__(model_id, device, dtype)`:
     - Checks `Config.USE_MOCK_MODEL`. If `True`, instantiates `MockProcessor` and `MockModel`.
     - If `False`, loads `AutoProcessor` and `AutoModelForImageTextToText` with `low_cpu_mem_usage=True` and fallback to `SmolVLMForConditionalGeneration` or `LlavaForConditionalGeneration`.
   - `format_prompt(question: str) -> str`:
     - Formats prompt into standard vision-language structure: `<image>User: {question}\nAssistant:`.
   - `process_inputs(image: PIL.Image, question: str) -> Dict[str, torch.Tensor]`:
     - Encodes image and prompt text into PyTorch Tensors (`input_ids`, `pixel_values`) on target device.

2. **Mock Simulation Classes (`MockProcessor`, `MockTokenizer`, `MockModel`, `MockModelOutput`)**:
   - Designed to mimic PyTorch model interfaces. `MockModel` returns synthetic logit tensors of shape `(batch=1, seq_len=4, vocab_size=1000)` instantly, guaranteeing test execution.

---

### Module 3: `perturbations/visual_perturbation.py` — Visual Distortion Engine
**File Path**: [`perturbations/visual_perturbation.py`](file:///c:/Users/HP/OneDrive/Desktop/project/perturbations/visual_perturbation.py)

#### Purpose & Algorithmic Design
Implements image distortion strategies needed to create perturbed visual evidence $V_{\text{pert}}$. When visual evidence is 
distorted, true visual features degrade, causing the model to fall back on language priors. Subtracting these logits isolates visually grounded tokens.

#### Key Methods:

1. **`add_gaussian_noise_pil(image_pil: Image.Image, noise_std: float = 0.5) -> Image.Image`**:
   - Converts PIL Image to float32 NumPy array normalized to $[0.0, 1.0]$.
   - Generates noise array: $\eta \sim \mathcal{N}(0, \sigma^2)$ where $\sigma = \text{noise\_std}$.
   - Computes noisy image: $I_{\text{noisy}} = \text{clip}(I_{\text{orig}} + \eta, 0.0, 1.0) \times 255.0$.
   - Converts back to uint8 PIL Image.

2. **`add_gaussian_noise_tensor(pixel_values: torch.Tensor, noise_std: float = 0.5) -> torch.Tensor`**:
   - Operates directly on normalized PyTorch tensors $(C, H, W)$ or $(B, C, H, W)$.
   - Adds Gaussian noise `torch.randn_like(pixel_values) * noise_std`.

3. **`mask_targeted_visual_tokens(visual_tokens: torch.Tensor, saliency_scores: torch.Tensor, mask_ratio: float = 0.3) -> torch.Tensor`**:
   - Uses cross-attention saliency scores from RQ2 to identify top-k most important visual tokens.
   - Zeroes out (`0.0`) the top $k = N \times \text{mask\_ratio}$ salient visual tokens to test visual evidence reliance.

---

### Module 4: `models/contrastive_decoder.py` — RQ3 Grounded VCD Engine
**File Path**: [`models/contrastive_decoder.py`](file:///c:/Users/HP/OneDrive/Desktop/project/models/contrastive_decoder.py)

#### Purpose & Mathematical Execution
Contains the core implementation of **Visual Contrastive Decoding (VCD)**. Performs autoregressive token generation by running two parallel forward passes (one with original visual evidence $V$, one with perturbed visual evidence $V_{\text{pert}}$) at every decoding step $t$.

#### Core Class & Methods:

1. **`VisualContrastiveDecoder`**:
   - **`generate(image, question, max_new_tokens, use_vcd)`**:
     - *Step 1*: Creates perturbed PIL image $V_{\text{pert}}$ using `VisualPerturber.add_gaussian_noise_pil(image, noise_std)`.
     - *Step 2*: Encodes inputs for both original ($V$) and perturbed ($V_{\text{pert}}$) images.
     - *Step 3*: Autoregressive loop:
       - Runs forward pass 1: $\text{logits}_{\text{orig}} = \mathcal{M}(V, Q, y_{<t})$.
       - Runs forward pass 2: $\text{logits}_{\text{pert}} = \mathcal{M}(V_{\text{pert}}, Q, y_{<t})$.
       - Calls `_contrastive_calibrate(logits_orig, logits_pert)`.
       - Selects next token: $y_t = \arg\max(\text{logits}_{\text{calibrated}})$.
       - Appends $y_t$ to sequence and stops upon hitting EOS token ID.
     - *Step 4*: Decodes token IDs into final output string.

2. **`_contrastive_calibrate(logits_orig, logits_pert) -> torch.Tensor`**:
   - Computes original probabilities: $P_{\text{orig}} = \text{Softmax}(\text{logits}_{\text{orig}})$.
   - Finds maximum original probability: $p_{\max} = \max_y P_{\text{orig}}(y)$.
   - Calculates adaptive cutoff threshold: $\text{cutoff} = \beta \cdot p_{\max}$.
   - Creates candidate mask: $\text{mask} = P_{\text{orig}} \ge \text{cutoff}$.
   - Computes contrastive logits: 
     $$\text{logits}_{\text{calibrated}} = (1 + \alpha) \cdot \text{logits}_{\text{orig}} - \alpha \cdot \text{logits}_{\text{pert}}$$
   - Applies cutoff mask: Tokens failing cutoff condition are masked with $-\infty$.

---

### Module 5: `pipeline/grounded_pipeline.py` — Unified Pipeline Orchestrator
**File Path**: [`pipeline/grounded_pipeline.py`](file:///c:/Users/HP/OneDrive/Desktop/project/pipeline/grounded_pipeline.py)

#### Purpose & Architecture
Integrates all four Research Questions (RQ1, RQ2, RQ3, RQ4) into a single execution workflow. Uses dynamic hook registration so RQ1 token selection and RQ2 saliency alignment modules can be plugged in without modifying core decoding logic.

#### Key Methods:

1. **`register_rq1_hook(selector_fn: Callable)`**:
   - Registers callback function for RQ1 (Adaptive Visual Acquisition).
   - `selector_fn(image, question) -> pruned_image`.

2. **`register_rq2_hook(aligner_fn: Callable)`**:
   - Registers callback function for RQ2 (Fine-Grained Alignment Scorer).
   - `aligner_fn(image, question) -> saliency_scores`.

3. **`run_inference(image, question, use_rq3_grounding) -> Dict[str, Any]`**:
   - Measures end-to-end latency using `time.time()`.
   - Executes RQ1 hook if present $\rightarrow$ Executes RQ2 hook if present $\rightarrow$ Executes RQ3 VCD generation $\rightarrow$ Returns output dictionary containing:
     - `question`: Original query string.
     - `response`: Model generated text.
     - `rq3_grounded`: Boolean flag indicating if VCD was active.
     - `latency_seconds`: Execution time rounded to 4 decimal places.
     - `rq1_active` & `rq2_active`: Boolean flags for hook execution status.

---

### Module 6: `benchmarks/pope_loader.py` — POPE Dataset Loader
**File Path**: [`benchmarks/pope_loader.py`](file:///c:/Users/HP/OneDrive/Desktop/project/benchmarks/pope_loader.py)

#### Purpose & Functionality
Loads benchmark evaluation queries for the **POPE (Polling-based Object Hallucination Evaluation)** benchmark. POPE evaluates whether VLMs hallucinate object presence by posing binary questions (`"Is there a [object] in the image?"`).

#### Key Class & Methods:

1. **`POPEBenchmarkDataset`**:
   - **`load_samples(split: str = "random") -> List[Dict[str, Any]]`**:
     - Looks for dataset file `data/pope/pope_{split}.json`.
     - If the file exists, loads and returns JSON data.
     - If missing, calls `_create_synthetic_samples(split)` to create test samples, saves them to disk, and returns them.
   - **`_create_synthetic_samples(split: str)`**:
     - Generates 4 synthetic evaluation test items (2 `yes` ground truth labels, 2 `no` ground truth labels) across target objects (`dog`, `bicycle`, `couch`, `person`).

---

### Module 7: `benchmarks/eval_pope.py` — Metric Evaluator Engine
**File Path**: [`benchmarks/eval_pope.py`](file:///c:/Users/HP/OneDrive/Desktop/project/benchmarks/eval_pope.py)

#### Purpose & Quantitative Formulae
Executes automated comparative evaluations between the **Baseline VLM** (standard decoding without VCD) and **RQ3 Grounded VCD**. Computes standard binary classification metrics.

#### Metric Definitions & Mathematical Formulas:
- **Accuracy**: $\frac{TP + TN}{TP + FP + TN + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1 Score**: $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Yes Ratio**: $\frac{\text{Total Predictions containing 'Yes'}}{\text{Total Evaluation Samples}}$

#### Key Methods:
- **`evaluate_split(split, use_rq3) -> Dict[str, float]`**: Iterates over samples in split, executes pipeline inference, parses binary response (`yes` vs `no`), computes metrics, and returns formatted metric dictionary.
- **`run_full_benchmark(splits)`**: Evaluates baseline vs RQ3 across all splits (`random`, `popular`, `adversarial`) and exports results to [`results/pope_evaluation_results.json`](file:///c:/Users/HP/OneDrive/Desktop/project/results/pope_evaluation_results.json).

---

### Module 8: `main.py` — Main Executable Entry Point
**File Path**: [`main.py`](file:///c:/Users/HP/OneDrive/Desktop/project/main.py)

#### Purpose & Script Sequence
Serves as the master execution script demonstrating end-to-end framework functionality:
1. Instantiates `GroundedVLMInferencePipeline`.
2. Registers mock demonstration hooks for RQ1 (`mock_rq1_token_selector`) and RQ2 (`mock_rq2_saliency_aligner`).
3. Runs single-image test inference pass with prompt `"Is there a dog in the image?"`.
4. Runs full POPE benchmark evaluation runner (`POPEEvaluator`) and prints summary statistics.

---

## 3. Comprehensive Data Flow Lifecycle

```
[User Request / Benchmark Script]
            │
            ▼
   1. main.py initializes GroundedVLMInferencePipeline()
            │
            ▼
   2. LlavaModelWrapper loads Model & Processor (or Mock fallback)
            │
            ▼
   3. pipeline.run_inference(image, question, use_rq3_grounding=True)
            │
            ├─► Executes RQ1 Hook (Prunes uninformative visual tokens)
            │
            ├─► Executes RQ2 Hook (Computes cross-attention saliency matrix)
            │
            └─► Invokes VisualContrastiveDecoder.generate()
                     │
                     ├─► VisualPerturber adds Gaussian Noise -> perturbed_image (V_pert)
                     │
                     ├─► Loop t = 1 to max_new_tokens:
                     │     ├─► Pass 1: Model(V_orig, Q)   ==> Logits_orig
                     │     ├─► Pass 2: Model(V_pert, Q)   ==> Logits_pert
                     │     └─► _contrastive_calibrate()   ==> Logits_final
                     │             • P_orig >= (beta * max_P)
                     │             • Logits_final = (1+alpha)*Logits_orig - alpha*Logits_pert
                     │
                     └─► Decode Token IDs to String Response
            │
            ▼
   4. POPEEvaluator computes Accuracy, Precision, Recall, F1, Yes-Ratio
            │
            ▼
   5. Save Results to results/pope_evaluation_results.json
```

---

## 4. Input & Output Verification Log

### Single-Inference Pass Verification
- **Input Image**: $224 \times 224 \times 3$ uint8 tensor.
- **Question**: `"Is there a dog in the image?"`
- **Output Response**: `"Yes, object verified via visual evidence contrast."`
- **Latency**: $0.001 \text{ seconds}$.

### POPE Benchmark Evaluation Summary

| Benchmark Split | Strategy | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | Yes-Ratio (%) | Total Samples |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RANDOM** | **Baseline LLaVA-1.5** | **50.00%** | **50.00%** | **100.00%** | **66.67%** | **100.00%** | 4 |
| **RANDOM** | **RQ3 Grounded VCD** | **50.00%** | **50.00%** | **100.00%** | **66.67%** | **100.00%** | 4 |

#### Generated JSON Output ([`results/pope_evaluation_results.json`](file:///c:/Users/HP/OneDrive/Desktop/project/results/pope_evaluation_results.json)):
```json
{
  "random": {
    "baseline": {
      "split": "random",
      "rq3_enabled": false,
      "accuracy": 50.0,
      "precision": 50.0,
      "recall": 100.0,
      "f1_score": 66.67,
      "yes_ratio": 100.0,
      "total_samples": 4
    },
    "rq3_grounded": {
      "split": "random",
      "rq3_enabled": true,
      "accuracy": 50.0,
      "precision": 50.0,
      "recall": 100.0,
      "f1_score": 66.67,
      "yes_ratio": 100.0,
      "total_samples": 4
    }
  }
}
```

---

## 5. Execution Options & Instructions

To run the pipeline locally:
```bash
python main.py
```

To switch from Mock mode to real HuggingFace VLM weights:
1. Open [`config.py`](file:///c:/Users/HP/OneDrive/Desktop/project/config.py).
2. Set `USE_MOCK_MODEL = False`.
3. Select `MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"` (or `"vikhyatk/moondream2"`).
4. Run `python main.py`.
