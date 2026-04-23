"""
Phase 3 — Linguistic Encoder
Extracts semantic / contextual features from transcript text using BERT.
Returns both a pooled CLS embedding and full token-level hidden states
so the cross-attention layer can operate at token granularity.
"""

import os
import torch
import torch.nn as nn

BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

# Downloaded weights go here — never touches final_model/
_EMOTION_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_EMOTION_MODELS_DIR, exist_ok=True)


class LinguisticEncoder(nn.Module):
    """
    Wraps `bert-base-uncased` to produce linguistic embeddings.

    Outputs
    -------
    cls_embedding : Tensor [batch, 768]
        The [CLS] token from the last hidden layer.
    token_features : Tensor [batch, seq_len, 768]
        Full token-level hidden states (for cross-attention).
    """

    def __init__(self, model_name: str = BERT_MODEL_NAME, device: str | None = None):
        super().__init__()
        from transformers import BertModel, BertTokenizer

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        print(f"⏳ Loading Linguistic Encoder ({model_name})...")
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, cache_dir=_EMOTION_MODELS_DIR
        )
        self.encoder = BertModel.from_pretrained(
            model_name, cache_dir=_EMOTION_MODELS_DIR
        )
        self.encoder.eval()
        self.encoder.to(self.device)
        print("✅ Linguistic Encoder loaded.")

    @torch.no_grad()
    def forward(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize *text* and run through BERT.

        Parameters
        ----------
        text : str
            Transcript text (lowercase is fine — BERT uncased handles it).

        Returns
        -------
        cls_embedding : Tensor [1, 768]
        token_features : Tensor [1, seq_len, 768]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden = outputs.last_hidden_state          # [1, seq_len, 768]
        cls_embedding = last_hidden[:, 0, :]             # [1, 768]

        return cls_embedding.cpu(), last_hidden.cpu()

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------
    def encode(self, text: str) -> dict:
        """
        Returns dict with keys: cls_embedding, token_features.
        """
        cls_embedding, token_features = self.forward(text)
        return {
            "cls_embedding": cls_embedding,       # [1, 768]
            "token_features": token_features,     # [1, seq_len, 768]
        }
