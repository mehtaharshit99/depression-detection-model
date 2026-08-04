"""
10_train_multihead_attention_pooling.py - multi-head attention-pooling model

Builds on the attention-pooling baseline with three independent audio pooling
heads. Each head can focus on different chunks before transcript fusion.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "attention_pooling_baseline",
    SCRIPT_DIR / "09_train_attention_pooling.py",
)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class MultiHeadAttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        text_dim: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int = 3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.audio_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.audio_head_projection = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_projection = (
            nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if text_dim > 0
            else None
        )
        fused_dim = hidden_dim + (hidden_dim if text_dim > 0 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, text_x: torch.Tensor, lengths: torch.Tensor):
        if torch.any(lengths == 0):
            raise ValueError("Zero-length sequence detected")

        projected = self.audio_projection(x)
        max_len = projected.size(1)
        mask = (
            torch.arange(max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(len(lengths), max_len)
            < lengths.unsqueeze(1)
        )
        attn_scores = self.audio_attention(projected).transpose(1, 2)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), -1e9)
        attn_weights = torch.softmax(attn_scores, dim=2)
        audio_contexts = torch.bmm(attn_weights, projected)
        audio_context = self.audio_head_projection(audio_contexts.flatten(start_dim=1))

        if self.text_projection is not None:
            text_context = self.text_projection(text_x)
            fused = torch.cat([audio_context, text_context], dim=1)
        else:
            fused = audio_context

        return self.classifier(fused)


if __name__ == "__main__":
    baseline.AttentionPoolingClassifier = MultiHeadAttentionPoolingClassifier
    baseline.RESULTS_PATH = baseline.MODEL_DIR / "cv_results_multihead_attention_pooling.csv"
    baseline.MODEL_PATH = baseline.MODEL_DIR / "multihead_attention_pooling_multimodal.pkl"
    baseline.CHECKPOINT_PREFIX = "multihead_attention_pooling"
    if "--seed" not in sys.argv:
        sys.argv.extend(["--seed", "99"])
    baseline.main()
