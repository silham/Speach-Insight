"""
Phase 5 — Cross-Attention Fusion
Dynamically combines acoustic and linguistic representations using two
multi-head attention heads, then concatenates the attended outputs with
VADER sentiment features and projects to a shared 512-d representation.

Architecture
------------
Audio→Text : acoustic pooled (Q) attends over BERT token sequence (K, V)
Text→Audio : BERT CLS (Q) attends over Wav2Vec2 frame sequence (K, V)
Concat [audio_attended ‖ text_attended ‖ vader_4d] → Linear(1540→512)
→ LayerNorm → ReLU
"""

import torch
import torch.nn as nn

HIDDEN_DIM = 768
VADER_DIM = 4
FUSED_DIM = 512
NUM_HEADS = 8


class CrossAttentionFusion(nn.Module):
    """
    Fuses acoustic, linguistic, and sentiment features via cross-attention.

    Inputs
    ------
    audio_pooled       : [B, 768]          mean-pooled Wav2Vec2 embedding
    audio_frames       : [B, T_a, 768]     frame-level Wav2Vec2 hidden states
    text_cls           : [B, 768]          BERT [CLS] embedding
    text_tokens        : [B, T_t, 768]     BERT token-level hidden states
    vader_features     : [B, 4]            VADER [pos, neg, neu, compound]

    Output
    ------
    fused : [B, 512]   — ready for the emotion classifier heads.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        vader_dim: int = VADER_DIM,
        fused_dim: int = FUSED_DIM,
        num_heads: int = NUM_HEADS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Audio → Text cross-attention
        # Q = audio_pooled (unsqueeze to [B,1,768])
        # K,V = text_tokens [B, T_t, 768]
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Text → Audio cross-attention
        # Q = text_cls (unsqueeze to [B,1,768])
        # K,V = audio_frames [B, T_a, 768]
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Projection: (768 + 768 + 4) → 512
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + vader_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio_pooled: torch.Tensor,
        audio_frames: torch.Tensor,
        text_cls: torch.Tensor,
        text_tokens: torch.Tensor,
        vader_features: torch.Tensor,
    ) -> torch.Tensor:
        # Audio→Text: which words does the acoustic tone align with?
        audio_q = audio_pooled.unsqueeze(1)                     # [B, 1, 768]
        audio_attended, _ = self.audio_to_text_attn(
            query=audio_q, key=text_tokens, value=text_tokens,
        )                                                       # [B, 1, 768]
        audio_attended = audio_attended.squeeze(1)               # [B, 768]

        # Text→Audio: which acoustic frames does the semantic content align with?
        text_q = text_cls.unsqueeze(1)                           # [B, 1, 768]
        text_attended, _ = self.text_to_audio_attn(
            query=text_q, key=audio_frames, value=audio_frames,
        )                                                       # [B, 1, 768]
        text_attended = text_attended.squeeze(1)                 # [B, 768]

        # Ensure vader_features is [B, 4]
        if vader_features.dim() == 1:
            vader_features = vader_features.unsqueeze(0)

        # Concatenate + project
        combined = torch.cat(
            [audio_attended, text_attended, vader_features], dim=-1
        )                                                       # [B, 1540]
        fused = self.projection(combined)                       # [B, 512]

        return fused
