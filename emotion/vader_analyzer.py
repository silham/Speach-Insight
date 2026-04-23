"""
Phase 4 — VADER Sentiment Wrapper
Returns a normalised [4] tensor (positive, negative, neutral, compound)
from lexicon-based VADER sentiment analysis.  Used as an explicit feature
gate inside the fusion module — not as a standalone predictor.
"""

import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderAnalyzer:
    """
    Thin wrapper around VADER that returns a tensor compatible with the
    cross-attention fusion module.
    """

    def __init__(self):
        print("⏳ Initialising VADER Sentiment Analyzer...")
        self.analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER ready.")

    def analyze(self, text: str) -> dict:
        """
        Run VADER on *text*.

        Returns
        -------
        dict with keys
            scores : dict   — raw VADER scores {pos, neg, neu, compound}
            tensor : Tensor [4] — [pos, neg, neu, compound] as float32
        """
        scores = self.analyzer.polarity_scores(text)
        tensor = torch.tensor(
            [scores["pos"], scores["neg"], scores["neu"], scores["compound"]],
            dtype=torch.float32,
        )
        return {"scores": scores, "tensor": tensor}
