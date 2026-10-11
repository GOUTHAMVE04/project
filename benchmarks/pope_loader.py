import json
import os
from PIL import Image
from typing import List, Dict, Any

class POPEBenchmarkDataset:
    """
    Loader and synthetic sample generator for POPE (Polling-based Object Hallucination Evaluation) benchmark.
    Tests binary 'Is there a [object] in the image?' queries across random, popular, and adversarial splits.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_samples(self, split: str = "random") -> List[Dict[str, Any]]:
        """
        Loads POPE evaluation items for a given split. Generates synthetic samples if data file not present.
        """
        file_path = os.path.join(self.data_dir, f"pope_{split}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            print(f"[POPELoader] {file_path} not found. Creating synthetic demo evaluation samples for '{split}' split...")
            synthetic_samples = self._create_synthetic_samples(split)
            with open(file_path, "w") as f:
                json.dump(synthetic_samples, f, indent=2)
            return synthetic_samples

    def _create_synthetic_samples(self, split: str) -> List[Dict[str, Any]]:
        """
        Creates synthetic POPE test cases to enable dry runs and verification immediately.
        """
        return [
            {
                "question_id": 1,
                "question": "Is there a dog in the image?",
                "label": "yes",
                "split": split,
                "image_path": "synthetic"
            },
            {
                "question_id": 2,
                "question": "Is there a bicycle in the image?",
                "label": "no",
                "split": split,
                "image_path": "synthetic"
            },
            {
                "question_id": 3,
                "question": "Is there a couch in the image?",
                "label": "no",
                "split": split,
                "image_path": "synthetic"
            },
            {
                "question_id": 4,
                "question": "Is there a person in the image?",
                "label": "yes",
                "split": split,
                "image_path": "synthetic"
            }
        ]
