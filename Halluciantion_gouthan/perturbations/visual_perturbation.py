import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class VisualPerturber:
    """
    Implements visual noise and token-level targeted perturbation strategies for RQ3 evidence verification.
    """

    @staticmethod
    def add_gaussian_noise_pil(image_pil: Image.Image, noise_std: float = 0.5) -> Image.Image:
        """
        Adds Gaussian noise to a PIL image to create a visually perturbed baseline.
        """
        img_array = np.array(image_pil, dtype=np.float32) / 255.0
        noise = np.random.normal(loc=0.0, scale=noise_std, size=img_array.shape)
        noisy_img = np.clip(img_array + noise, 0.0, 1.0) * 255.0
        return Image.fromarray(noisy_img.astype(np.uint8))

    @staticmethod
    def add_gaussian_noise_tensor(pixel_values: torch.Tensor, noise_std: float = 0.5) -> torch.Tensor:
        """
        Adds Gaussian noise directly to normalized image tensor (C, H, W) or (B, C, H, W).
        """
        noise = torch.randn_like(pixel_values) * noise_std
        return pixel_values + noise

    @staticmethod
    def mask_targeted_visual_tokens(visual_tokens: torch.Tensor, saliency_scores: torch.Tensor, mask_ratio: float = 0.3) -> torch.Tensor:
        """
        Targeted perturbation driven by RQ2 Saliency / Alignment scores.
        Masks top-k salient visual tokens to test visual evidence dependency.
        """
        B, N, D = visual_tokens.shape
        num_mask = int(N * mask_ratio)
        
        # Get indices of top salient tokens
        _, top_indices = torch.topk(saliency_scores, k=num_mask, dim=-1)
        
        perturbed_tokens = visual_tokens.clone()
        for b in range(B):
            perturbed_tokens[b, top_indices[b], :] = 0.0
            
        return perturbed_tokens
