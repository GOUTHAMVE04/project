import sys
import os
from PIL import Image
import numpy as np

from config import Config
from pipeline.grounded_pipeline import GroundedVLMInferencePipeline
from benchmarks.eval_pope import POPEEvaluator

def mock_rq1_token_selector(image: Image.Image, question: str) -> Image.Image:
    """
    Demo Hook for RQ1: Adaptive Visual Token Selection.
    Simulates token selection / ROI cropping before VLM decoder pass.
    """
    print("  -> [RQ1 Hook Executed]: Dynamically selected 60% salient visual tokens.")
    return image

def mock_rq2_saliency_aligner(image: Image.Image, question: str):
    """
    Demo Hook for RQ2: Fine-Grained Alignment Scorer.
    Simulates cross-attention visual token saliency alignment scoring.
    """
    print("  -> [RQ2 Hook Executed]: Computed cross-attention saliency map for token grounding.")
    return np.ones(576)

def main():
    print("================================================================")
    print("   Grounded VLM Framework (RQ3 Core Engine & RQ1/2/4 Scalability)  ")
    print("================================================================")

    # 1. Initialize Unified Pipeline (Lazy-loads model when needed)
    print("\n[Step 1] Initializing Grounded VLM Inference Pipeline...")
    pipeline = GroundedVLMInferencePipeline()

    # 2. Register RQ1 and RQ2 Hooks to prove framework modularity
    print("\n[Step 2] Registering RQ1 (Token Selection) and RQ2 (Saliency Alignment) Hooks...")
    pipeline.register_rq1_hook(mock_rq1_token_selector)
    pipeline.register_rq2_hook(mock_rq2_saliency_aligner)

    # 3. Single Image Inference Test
    print("\n[Step 3] Running Test Inference Pass...")
    dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    question = "Is there a dog in the image?"
    
    result = pipeline.run_inference(dummy_img, question, use_rq3_grounding=True)
    print(f"\n[Inference Result]:")
    print(f"  Question:      {result['question']}")
    print(f"  Response:      {result['response']}")
    print(f"  RQ3 Grounded:  {result['rq3_grounded']}")
    print(f"  Latency (sec): {result['latency_seconds']}")

    # 4. Benchmark Evaluation
    print("\n[Step 4] Running POPE Benchmark Hallucination Evaluation Engine...")
    evaluator = POPEEvaluator(pipeline)
    results = evaluator.run_full_benchmark(splits=["random"])

    print("\n================================================================")
    print("                      POPE Evaluation Summary                  ")
    print("================================================================")
    for split, res in results.items():
        print(f"Split: {split.upper()}")
        print(f"  Baseline LLaVA-1.5: Accuracy = {res['baseline']['accuracy']}%, F1 = {res['baseline']['f1_score']}%, Yes-Ratio = {res['baseline']['yes_ratio']}%")
        print(f"  RQ3 Grounded VCD:   Accuracy = {res['rq3_grounded']['accuracy']}%, F1 = {res['rq3_grounded']['f1_score']}%, Yes-Ratio = {res['rq3_grounded']['yes_ratio']}%")

if __name__ == "__main__":
    main()
