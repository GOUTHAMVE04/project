import os
import torch

class Config:
    # Set USE_MOCK_MODEL = True for instant (< 2 second) local execution and demo runs 
    # without waiting for multi-gigabyte HuggingFace weight downloads over the network.
    USE_MOCK_MODEL = True
    
    # Lightweight Vision-Language Model Selection (Used when USE_MOCK_MODEL = False)
    # - "HuggingFaceTB/SmolVLM-256M-Instruct" (~500 MB, ultra-fast 256M VLM)
    # - "HuggingFaceTB/SmolVLM-Instruct" (~1.5 GB, 2.2B VLM)
    # - "vikhyatk/moondream2" (~1.8 GB, fast lightweight VLM)
    MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # ICD (Instruction Contrastive Decoding, Wang et al., ACL 2024) Hyperparameters
    ICD_LAMBDA = 1.0       # Contrastive penalty weight (lambda) for disturbance instruction
    ICD_BETA = 0.1         # Adaptive candidate cutoff threshold (beta)
    DISTURBANCE_INSTRUCTION = "You are a confused object detector with severe hallucination bias."

    # Visual Perturbation Hyperparameters (Dual Contrast ICD + VCD)
    VCD_ALPHA = 1.0        # Contrastive amplification coefficient
    VCD_BETA = 0.1         # Adaptive candidate cutoff threshold
    NOISE_STD = 0.5        # Gaussian noise standard deviation for visual perturbation
    MASK_RATIO = 0.3       # Percentage of visual tokens to mask in targeted perturbation

    # Generation Parameters
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.2
    TOP_P = 0.9

    # Benchmark & Dataset Configurations
    BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "data", "pope")
    POPE_SPLITS = ["random", "popular", "adversarial"]
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

os.makedirs(Config.BENCHMARK_DIR, exist_ok=True)
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
