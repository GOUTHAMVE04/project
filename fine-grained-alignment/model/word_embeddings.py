"""
Frozen Word Embeddings Table for Fine-Grained Patch Alignment.
Loads GPT-2 token embedding matrix (dimension 768) and tokenizes/pools labels.
All parameters are frozen (requires_grad = False).
"""

from typing import List, Union
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class FrozenWordEmbeddings(nn.Module):
    """
    Frozen Language Model Word Embedding Table (GPT-2 small).
    Tokenizes input category labels and mean-pools subtoken embeddings into a single vector.
    """
    def __init__(self, model_name="gpt2", device="cpu"):
        super().__init__()
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Pad token convention for GPT-2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Extract only the token embedding table (wte: 50257 -> 768)
        full_gpt = GPT2Model.from_pretrained(model_name)
        self.wte = full_gpt.wte  # nn.Embedding(50257, 768)

        # Completely freeze all embedding parameters
        for param in self.wte.parameters():
            param.requires_grad = False

        self.eval()
        self.to(device)

        # In-memory embedding cache for category labels to avoid recomputing
        self._cache = {}

    def encode_single_label(self, label: str) -> torch.Tensor:
        """
        Tokenizes a single label string, extracts subtoken embeddings, and mean-pools them.
        Returns: Tensor of shape (768,)
        """
        clean_label = label.strip().lower()
        if clean_label in self._cache:
            return self._cache[clean_label].to(self.device)

        with torch.no_grad():
            token_ids = self.tokenizer.encode(clean_label, add_special_tokens=False)
            if len(token_ids) == 0:
                token_ids = [self.tokenizer.eos_token_id]

            input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            subtoken_embeds = self.wte(input_ids)  # (num_subtokens, 768)
            pooled_embed = subtoken_embeds.mean(dim=0)  # (768,)

            self._cache[clean_label] = pooled_embed.detach().cpu()
            return pooled_embed

    def encode_labels(self, labels: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes a list of label strings.
        Returns: Tensor of shape (len(labels), 768)
        """
        if isinstance(labels, str):
            labels = [labels]

        embed_list = [self.encode_single_label(label) for label in labels]
        return torch.stack(embed_list, dim=0)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
