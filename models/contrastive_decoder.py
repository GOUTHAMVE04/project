import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any, Optional
from config import Config
from models.llava_wrapper import LlavaModelWrapper
from perturbations.visual_perturbation import VisualPerturber

class InstructionContrastiveDecoder:
    """
    Implements Instruction Contrastive Decoding (ICD) from:
    'Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding'
    Xintong Wang, Jingheng Pan, Liang Ding, Chris Biemann (ACL 2024 / arXiv:2403.18715).

    Contrasts token output distributions generated from standard instructions (X_ins)
    against disturbance instructions (X'_ins) that amplify hallucination language priors.
    Formula:
        p_icd(y_t | X_V, X_ins) = (1 + lambda) * logits_std(y_t) - lambda * logits_dist(y_t)
    subject to adaptive candidate cutoff threshold beta:
        cutoff = beta * max(probs_std)
    """
    def __init__(
        self, 
        model_wrapper: LlavaModelWrapper, 
        icd_lambda: float = Config.ICD_LAMBDA, 
        icd_beta: float = Config.ICD_BETA,
        disturbance_instruction: str = Config.DISTURBANCE_INSTRUCTION,
        noise_std: float = Config.NOISE_STD
    ):
        self.wrapper = model_wrapper
        self.icd_lambda = icd_lambda
        self.icd_beta = icd_beta
        self.disturbance_instruction = disturbance_instruction
        self.noise_std = noise_std
        self.perturber = VisualPerturber()

    def generate(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = Config.MAX_NEW_TOKENS,
        use_icd: bool = True,
        use_vcd: bool = False
    ) -> str:
        """
        Generates text output for a question given an image using Instruction Contrastive Decoding.
        """
        # Instant returns for mock mode
        if getattr(self.wrapper, "use_mock", False):
            return "Yes, verified by instruction contrastive decoding."

        # 1. Standard instruction inputs
        inputs_std = self.wrapper.process_inputs(image, question, is_disturbance=False)
        input_ids_std = inputs_std["input_ids"]
        pixel_values_std = inputs_std["pixel_values"]

        # 2. Disturbance instruction inputs (ICD branch)
        if use_icd:
            inputs_dist = self.wrapper.process_inputs(image, question, is_disturbance=True)
            input_ids_dist = inputs_dist["input_ids"]
            pixel_values_dist = inputs_dist["pixel_values"]

        # 3. Optional visual perturbation (Dual ICD + VCD)
        if use_vcd:
            perturbed_image = self.perturber.add_gaussian_noise_pil(image, noise_std=self.noise_std)
            inputs_pert = self.wrapper.process_inputs(perturbed_image, question, is_disturbance=False)
            input_ids_pert = inputs_pert["input_ids"]
            pixel_values_pert = inputs_pert["pixel_values"]

        generated_tokens = []
        cur_input_ids_std = input_ids_std.clone()
        cur_input_ids_dist = input_ids_dist.clone() if use_icd else None
        cur_input_ids_pert = input_ids_pert.clone() if use_vcd else None

        eos_token_id = getattr(self.wrapper.processor.tokenizer, "eos_token_id", 2)

        for step in range(max_new_tokens):
            with torch.no_grad():
                # Forward pass: Standard instruction
                logits_std = self.wrapper.model(
                    input_ids=cur_input_ids_std,
                    pixel_values=pixel_values_std.to(dtype=self.wrapper.dtype),
                    return_dict=True
                ).logits[:, -1, :]

                if use_icd:
                    # Forward pass: Disturbance instruction
                    logits_dist = self.wrapper.model(
                        input_ids=cur_input_ids_dist,
                        pixel_values=pixel_values_dist.to(dtype=self.wrapper.dtype),
                        return_dict=True
                    ).logits[:, -1, :]

                    next_token_logits = self._contrastive_calibrate(logits_std, logits_dist)
                elif use_vcd:
                    logits_pert = self.wrapper.model(
                        input_ids=cur_input_ids_pert,
                        pixel_values=pixel_values_pert.to(dtype=self.wrapper.dtype),
                        return_dict=True
                    ).logits[:, -1, :]
                    next_token_logits = self._contrastive_calibrate(logits_std, logits_pert)
                else:
                    next_token_logits = logits_std

                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.item() == eos_token_id:
                    break

                generated_tokens.append(next_token.item())
                cur_input_ids_std = torch.cat([cur_input_ids_std, next_token], dim=-1)
                if use_icd:
                    cur_input_ids_dist = torch.cat([cur_input_ids_dist, next_token], dim=-1)
                if use_vcd:
                    cur_input_ids_pert = torch.cat([cur_input_ids_pert, next_token], dim=-1)

        output_text = self.wrapper.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text.strip()

    def _contrastive_calibrate(self, logits_std: torch.Tensor, logits_dist: torch.Tensor) -> torch.Tensor:
        """
        Applies mathematical ICD formulation with adaptive candidate truncation beta.
        Logits_final = (1 + lambda) * Logits_std - lambda * Logits_dist
        """
        probs_std = F.softmax(logits_std, dim=-1)
        max_prob = torch.max(probs_std, dim=-1, keepdim=True).values
        cutoff = self.icd_beta * max_prob
        
        cutoff_mask = probs_std >= cutoff
        calibrated_logits = (1 + self.icd_lambda) * logits_std - self.icd_lambda * logits_dist
        calibrated_logits = torch.where(cutoff_mask, calibrated_logits, torch.tensor(float('-inf'), device=logits_std.device))
        
        return calibrated_logits

# Backward-compatibility alias
VisualContrastiveDecoder = InstructionContrastiveDecoder
