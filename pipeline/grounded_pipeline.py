import time
from PIL import Image
from typing import Optional, Callable, Dict, Any
from models.llava_wrapper import LlavaModelWrapper
from models.contrastive_decoder import InstructionContrastiveDecoder
from config import Config

class GroundedVLMInferencePipeline:
    """
    Unified Pipeline Framework integrating RQ1 (Adaptive Token Selection), 
    RQ2 (Fine-Grained Alignment), RQ3 (Instruction Contrastive Decoding - Wang et al., ACL 2024),
    and RQ4 (Joint Optimization).
    """
    def __init__(self, model_wrapper: Optional[LlavaModelWrapper] = None):
        self.wrapper = model_wrapper or LlavaModelWrapper()
        self.icd_decoder = InstructionContrastiveDecoder(self.wrapper)
        # Backward compatibility alias
        self.vcd_decoder = self.icd_decoder
        
        # Extension Hooks for RQ1 and RQ2 integration
        self.rq1_token_selector_hook: Optional[Callable] = None
        self.rq2_saliency_aligner_hook: Optional[Callable] = None

    def register_rq1_hook(self, selector_fn: Callable):
        """
        Registers RQ1 Adaptive Visual Token Pruner hook.
        selector_fn(pixel_values, question) -> pruned_pixel_values / token_mask
        """
        self.rq1_token_selector_hook = selector_fn
        print("[GroundedPipeline] Registered RQ1 Token Selection Hook.")

    def register_rq2_hook(self, aligner_fn: Callable):
        """
        Registers RQ2 Fine-Grained Alignment Saliency Scorer hook.
        aligner_fn(visual_features, query_tokens) -> saliency_map
        """
        self.rq2_saliency_aligner_hook = aligner_fn
        print("[GroundedPipeline] Registered RQ2 Saliency Alignment Hook.")

    def run_inference(
        self, 
        image: Image.Image, 
        question: str, 
        use_rq3_grounding: bool = True,
        use_icd: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference pass across active RQ modules.
        Uses Instruction Contrastive Decoding (ICD) by default.
        """
        start_time = time.time()
        
        # Stage 1: RQ1 Visual Token Selection (if hook registered)
        if self.rq1_token_selector_hook is not None:
            image = self.rq1_token_selector_hook(image, question)

        # Stage 2: RQ2 Fine-Grained Saliency Alignment (if hook registered)
        saliency_scores = None
        if self.rq2_saliency_aligner_hook is not None:
            saliency_scores = self.rq2_saliency_aligner_hook(image, question)

        # Stage 3: ICD Instruction Contrastive Evidence Verification Generation
        response = self.icd_decoder.generate(
            image=image, 
            question=question, 
            use_icd=use_rq3_grounding and use_icd
        )
        
        latency = time.time() - start_time
        
        return {
            "question": question,
            "response": response,
            "rq3_grounded": use_rq3_grounding,
            "use_icd": use_icd,
            "latency_seconds": round(latency, 4),
            "rq1_active": self.rq1_token_selector_hook is not None,
            "rq2_active": self.rq2_saliency_aligner_hook is not None
        }
