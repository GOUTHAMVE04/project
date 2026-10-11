import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any, Optional
from config import Config
from models.llava_wrapper import LlavaModelWrapper
from perturbations.visual_perturbation import VisualPerturber

class VisualContrastiveDecoder:
    """
    Implements RQ3 Grounded Visual Contrastive Decoding (VCD).
    Contrasts original visual logit predictions against perturbed visual predictions 
    to filter out language-prior visual hallucinations.
    """
    def __init__(
        self, 
        model_wrapper: LlavaModelWrapper, 
        alpha: float = Config.VCD_ALPHA, 
        beta: float = Config.VCD_BETA,
        noise_std: float = Config.NOISE_STD
    ):
        self.wrapper = model_wrapper
        self.alpha = alpha
        self.beta = beta
        self.noise_std = noise_std
        self.perturber = VisualPerturber()

    def generate(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = Config.MAX_NEW_TOKENS,
        use_vcd: bool = True
    ) -> str:
        """
        Generates text output for a question given an image using Visual Contrastive Decoding.
        """
        # Instant returns for mock mode
        if getattr(self.wrapper, "use_mock", False):
            return "Yes, object verified via visual evidence contrast."

        # 1. Create perturbed image if VCD enabled
        if use_vcd:
            perturbed_image = self.perturber.add_gaussian_noise_pil(image, noise_std=self.noise_std)
            inputs_pert = self.wrapper.process_inputs(perturbed_image, question)
            pixel_values_pert = inputs_pert["pixel_values"]
        
        # 2. Process original image
        inputs_orig = self.wrapper.process_inputs(image, question)
        input_ids = inputs_orig["input_ids"]
        pixel_values_orig = inputs_orig["pixel_values"]

        generated_tokens = []
        cur_input_ids = input_ids.clone()
        cur_input_ids_pert = input_ids.clone() if use_vcd else None

        eos_token_id = getattr(self.wrapper.processor.tokenizer, "eos_token_id", 2)

        for step in range(max_new_tokens):
            with torch.no_grad():
                # Forward pass 1: Original image
                logits_orig = self.wrapper.model(
                    input_ids=cur_input_ids,
                    pixel_values=pixel_values_orig.to(dtype=self.wrapper.dtype),
                    return_dict=True
                ).logits[:, -1, :]

                if use_vcd:
                    # Forward pass 2: Perturbed image
                    logits_pert = self.wrapper.model(
                        input_ids=cur_input_ids_pert,
                        pixel_values=pixel_values_pert.to(dtype=self.wrapper.dtype),
                        return_dict=True
                    ).logits[:, -1, :]

                    next_token_logits = self._contrastive_calibrate(logits_orig, logits_pert)
                else:
                    next_token_logits = logits_orig

                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.item() == eos_token_id:
                    break

                generated_tokens.append(next_token.item())
                cur_input_ids = torch.cat([cur_input_ids, next_token], dim=-1)
                if use_vcd:
                    cur_input_ids_pert = torch.cat([cur_input_ids_pert, next_token], dim=-1)

        output_text = self.wrapper.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text.strip()

    def _contrastive_calibrate(self, logits_orig: torch.Tensor, logits_pert: torch.Tensor) -> torch.Tensor:
        """
        Applies mathematical VCD formulation with adaptive candidate truncation beta.
        Logits_final = (1 + alpha) * Logits_orig - alpha * Logits_pert
        """
        probs_orig = F.softmax(logits_orig, dim=-1)
        max_prob = torch.max(probs_orig, dim=-1, keepdim=True).values
        cutoff = self.beta * max_prob
        
        cutoff_mask = probs_orig >= cutoff
        calibrated_logits = (1 + self.alpha) * logits_orig - self.alpha * logits_pert
        calibrated_logits = torch.where(cutoff_mask, calibrated_logits, torch.tensor(float('-inf'), device=logits_orig.device))
        
        return calibrated_logits
