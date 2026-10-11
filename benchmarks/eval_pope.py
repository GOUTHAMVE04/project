import os
import json
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Any
from config import Config
from pipeline.grounded_pipeline import GroundedVLMInferencePipeline
from benchmarks.pope_loader import POPEBenchmarkDataset

class POPEEvaluator:
    """
    Evaluates VLM hallucination metrics on the POPE benchmark dataset (Li et al., EMNLP 2023).
    Compares Standard Baseline vs. Instruction Contrastive Decoding (ICD, Wang et al., ACL 2024).
    """
    def __init__(self, pipeline: GroundedVLMInferencePipeline, data_dir: str = Config.BENCHMARK_DIR):
        self.pipeline = pipeline
        self.dataset_loader = POPEBenchmarkDataset(data_dir)

    def evaluate_split(self, split: str = "random", use_icd: bool = True) -> Dict[str, float]:
        """
        Runs evaluation over a POPE split and computes Accuracy, Precision, Recall, F1, and Yes Ratio.
        """
        samples = self.dataset_loader.load_samples(split)
        
        tp, fp, tn, fn = 0, 0, 0, 0
        total_yes = 0

        # Create dummy image for synthetic evaluation
        dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

        for sample in samples:
            question = sample["question"]
            ground_truth = sample["label"].lower().strip()

            # Execute pipeline
            result = self.pipeline.run_inference(dummy_img, question, use_rq3_grounding=use_icd, use_icd=use_icd)
            pred_raw = result["response"].lower().strip()
            
            # Binary parsing of response
            pred = "yes" if "yes" in pred_raw else "no"

            if pred == "yes":
                total_yes += 1

            if ground_truth == "yes" and pred == "yes":
                tp += 1
            elif ground_truth == "no" and pred == "yes":
                fp += 1
            elif ground_truth == "no" and pred == "no":
                tn += 1
            elif ground_truth == "yes" and pred == "no":
                fn += 1

        total = len(samples)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        yes_ratio = total_yes / total if total > 0 else 0.0

        metrics = {
            "split": split,
            "icd_enabled": use_icd,
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "yes_ratio": round(yes_ratio * 100, 2),
            "total_samples": total
        }

        return metrics

    def run_full_benchmark(self, splits: List[str] = Config.POPE_SPLITS) -> Dict[str, Any]:
        """
        Runs full comparison across baseline and ICD (Instruction Contrastive Decoding).
        """
        full_results = {}
        for split in splits:
            print(f"\n--- Evaluating POPE Split: {split} ---")
            baseline_metrics = self.evaluate_split(split, use_icd=False)
            icd_metrics = self.evaluate_split(split, use_icd=True)
            
            full_results[split] = {
                "baseline": baseline_metrics,
                "icd_grounded": icd_metrics,
                "rq3_grounded": icd_metrics  # backward-compatible key
            }

        output_path = os.path.join(Config.RESULTS_DIR, "pope_evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)

        print(f"\n[POPEEvaluator] Results saved to {output_path}")
        return full_results
