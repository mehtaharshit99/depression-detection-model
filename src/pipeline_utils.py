"""
pipeline_utils.py — Shared utilities and sequence model components
==================================================================
Contains:
  - audio constants and chunking helpers
  - optional prosody feature extraction
  - participant-level sequence dataset
  - collate function for variable-length sequences
  - BiGRU + attention classifier
"""

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ─────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────
CHUNK_SEC = 12
TARGET_SR = 16_000
PROSODY_DIM = 13


# ─────────────────────────────────────────────
# CHUNKING HELPER
# ─────────────────────────────────────────────
def chunk_waveform(waveform_np: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Split a mono waveform into non-overlapping CHUNK_SEC-second segments.
    Segments shorter than 1 second are discarded.

    Args:
        waveform_np : 1-D float32 numpy array (mono)
        sr          : sample rate

    Returns:
        list of 1-D float32 numpy arrays, each ~CHUNK_SEC * TARGET_SR samples long
    """
    import torchaudio

    if sr != TARGET_SR:
        wav_t = torch.from_numpy(waveform_np).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, TARGET_SR)
        waveform_np = wav_t.squeeze(0).numpy()

    total_sec = len(waveform_np) / TARGET_SR
    chunks = []

    for start in np.arange(0, total_sec, CHUNK_SEC):
        end = min(start + CHUNK_SEC, total_sec)
        seg = waveform_np[int(start * TARGET_SR): int(end * TARGET_SR)]
        if len(seg) < TARGET_SR:
            continue
        chunks.append(seg.astype(np.float32))

    return chunks


# ─────────────────────────────────────────────
# OPTIONAL PROSODY FEATURES
# ─────────────────────────────────────────────
def extract_prosody_features(waveform_np: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract 13 lightweight prosodic features from a mono audio chunk.

    Returns:
        np.ndarray of shape (PROSODY_DIM,), float32, NaN-safe
    """
    features = np.zeros(PROSODY_DIM, dtype=np.float32)

    if waveform_np is None or len(waveform_np) < sr * 0.1:
        return features

    try:
        y = waveform_np.astype(np.float32)

        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced_flag = f0 > 0
        f0_voiced = f0[voiced_flag] if voiced_flag.any() else np.array([0.0])

        features[0] = float(np.mean(f0_voiced))
        features[1] = float(np.std(f0_voiced))
        features[2] = float(np.min(f0_voiced))
        features[3] = float(np.max(f0_voiced))
        features[4] = float(features[3] - features[2])

        rms = librosa.feature.rms(y=y)[0]
        features[5] = float(np.mean(rms))
        features[6] = float(np.std(rms))

        features[7] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
        features[8] = float(voiced_flag.sum() / len(voiced_flag))
        features[9] = 1.0 - features[8]
        features[10] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
        features[11] = float(np.mean(np.abs(np.diff(f0_voiced)))) if len(f0_voiced) > 1 else 0.0
        features[12] = float(np.mean(np.abs(np.diff(rms)))) if len(rms) > 1 else 0.0

    except Exception:
        pass

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ParticipantSequenceDataset(Dataset):
    """
    One sample = one participant sequence of chunk-level Wav2Vec2 features.

    Expected columns:
      - participant_id
      - chunk_idx
      - label
      - w2v_0 ... w2v_767
    """

    def __init__(self, df: pd.DataFrame):
        self.samples = []
        self.participant_ids = []
        self.labels = []

        feature_cols = sorted(
            [c for c in df.columns if c.startswith("w2v_")],
            key=lambda c: int(c.split("_")[1]),
        )

        if not feature_cols:
            raise ValueError("No Wav2Vec2 feature columns found (expected w2v_*).")

        grouped = df.groupby("participant_id")

        for pid, group in grouped:
            group = group.sort_values("chunk_idx")

            if group.empty:
                continue

            features = group[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            features = features.values.astype(np.float32)

            if features.shape[0] == 0:
                continue

            label = int(group["label"].iloc[0])

            self.samples.append((features, label))
            self.participant_ids.append(str(pid))
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]

        return {
            "features": torch.from_numpy(x),
            "label": torch.tensor(float(y), dtype=torch.float32),
            "participant_id": self.participant_ids[idx],
        }

    def apply_standardization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Standardize every participant sequence in-place using train-fold statistics.
        """
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        mean = mean.astype(np.float32)

        updated_samples = []
        for features, label in self.samples:
            features = ((features - mean) / std).astype(np.float32)
            updated_samples.append((features, label))

        self.samples = updated_samples


# ─────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────
def collate_fn(batch):
    """
    Pad variable-length participant sequences to the max sequence length in batch.
    """
    filtered = [item for item in batch if item["features"].shape[0] > 0]

    if not filtered:
        return None

    xs = [item["features"] for item in filtered]
    ys = torch.stack([item["label"] for item in filtered])
    pids = [item["participant_id"] for item in filtered]

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = xs[0].shape[1]

    padded = torch.zeros(len(xs), max_len, feat_dim, dtype=torch.float32)

    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x

    return padded, ys, lengths, pids


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class GRUSequenceClassifier(nn.Module):
    """
    BiGRU + attention pooling for participant-level depression classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if torch.any(lengths == 0):
            raise ValueError("Zero-length sequence detected")

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )

        max_len = out.size(1)
        mask = (
            torch.arange(max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(len(lengths), max_len)
            < lengths.unsqueeze(1)
        )

        attn_scores = self.attention(out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        context = self.dropout(context)

        return self.classifier(context)
