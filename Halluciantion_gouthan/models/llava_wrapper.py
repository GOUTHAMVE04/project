import torch
from PIL import Image
from typing import Optional, Dict, Any
from config import Config

class VLMModelWrapper:
    """
    Generalized Vision-Language Model Wrapper supporting both Real PyTorch/HuggingFace Models 
    and Instant Mock Execution for fast local debugging & demonstrations.
    """
    def __init__(self, model_id: str = Config.MODEL_ID, device: str = Config.DEVICE, dtype: torch.dtype = Config.DTYPE):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.use_mock = Config.USE_MOCK_MODEL
        
        if self.use_mock:
            print(f"[VLMModelWrapper] Running in INSTANT DEMO MODE (USE_MOCK_MODEL = True). No large weight download required.")
            self.processor = MockProcessor()
            self.model = MockModel(device, dtype)
        else:
            print(f"[VLMModelWrapper] Loading real weights for {model_id} from HuggingFace...")
            from transformers import AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration, SmolVLMForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
                    device_map=device if device != "cpu" else None
                )
            except Exception as e:
                if "smolvlm" in model_id.lower():
                    self.model = SmolVLMForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device if device != "cpu" else None)
                else:
                    self.model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device if device != "cpu" else None)
            self.model.eval()

    def format_prompt(self, question: str, is_disturbance: bool = False) -> str:
        if is_disturbance:
            return f"<image>User: {Config.DISTURBANCE_INSTRUCTION} {question}\nAssistant:"
        return f"<image>User: {question}\nAssistant:"

    def process_inputs(self, image: Image.Image, question: str, is_disturbance: bool = False) -> Dict[str, torch.Tensor]:
        if self.use_mock:
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]], device=self.device),
                "pixel_values": torch.zeros((1, 3, 224, 224), device=self.device, dtype=self.dtype)
            }
        prompt = self.format_prompt(question, is_disturbance=is_disturbance)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

class MockTokenizer:
    eos_token_id = 2
    def decode(self, tokens, **kwargs):
        return "Yes, verified by visual contrastive evidence."

class MockProcessor:
    tokenizer = MockTokenizer()

class MockModelOutput:
    def __init__(self, device, dtype):
        self.logits = torch.randn(1, 4, 1000, device=device, dtype=dtype)

class MockModel:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def eval(self):
        pass

    def __call__(self, input_ids=None, pixel_values=None, return_dict=True, **kwargs):
        return MockModelOutput(self.device, self.dtype)

# Backward compatibility alias
LlavaModelWrapper = VLMModelWrapper
