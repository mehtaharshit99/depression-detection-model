"""
pipeline_utils.py — Shared utilities and sequence model components
==================================================================
Contains:
  - audio constants and chunking helpers
  - canonical skip-reason constants + deterministic file discovery
  - lightweight transcript text cleaner
  - optional prosody feature extraction
  - participant-level sequence dataset
  - collate function for variable-length sequences
  - BiGRU + attention classifier
  - lightweight multi-head attention-pooling classifier
"""

import re

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset

CHUNK_SEC = 12
TARGET_SR = 16_000
PROSODY_DIM = 13

# ---------------------------------------------------------------------------
# Canonical skip-reason constants
# ---------------------------------------------------------------------------
SKIP_NOT_IN_LABEL_MAP         = "not_in_label_map"
SKIP_ALREADY_EXISTS           = "already_exists"
SKIP_MISSING_AUDIO_AND_TRANS  = "missing_audio_and_transcript"
SKIP_MISSING_AUDIO            = "missing_audio"
SKIP_MISSING_TRANSCRIPT       = "missing_transcript"
SKIP_TRANSCRIPT_PARSE_FAILURE = "transcript_parse_failure"
SKIP_NO_PARTICIPANT_ROWS      = "no_participant_rows"
SKIP_NO_USABLE_SEGMENTS       = "no_usable_segments"
SKIP_ALL_CHUNKS_SILENT        = "all_chunks_invalid_silent"
SKIP_AUDIO_LOAD_FAILURE       = "audio_load_failure"
SKIP_METADATA_COLLISION       = "metadata_file_collision"

# Quality-filter skip reasons
SKIP_TOO_FEW_CHUNKS           = "too_few_valid_chunks"
SKIP_TOO_SHORT_SPEECH         = "participant_speech_too_short"

SUCCESS = "success"


# ---------------------------------------------------------------------------
# Deterministic file-discovery helper (ignores macOS ._* files)
# ---------------------------------------------------------------------------

def find_participant_files(folder):
    """
    Return (audio_path | None, transcript_path | None) for a participant
    folder, ignoring all paths whose name starts with '._'.

    Both lists are sorted before taking the first element so results are
    deterministic across operating systems and Python versions.
    """
    audio_files = sorted(
        f for f in folder.glob("*_AUDIO.wav") if not f.name.startswith("._")
    )
    transcript_files = sorted(
        f for f in folder.glob("*TRANSCRIPT*.csv") if not f.name.startswith("._")
    )
    audio_path      = audio_files[0]      if audio_files      else None
    transcript_path = transcript_files[0] if transcript_files  else None
    return audio_path, transcript_path


# ---------------------------------------------------------------------------
# Transcript text cleaner
# ---------------------------------------------------------------------------

# Patterns that are not real speech content.
_TRANSCRIPT_ARTIFACT_RE = re.compile(
    r"""
    <[^>]*>              |
    \[[^\]]*\]           |
    \([^)]*\)            |
    \b(um+|uh+|hmm+|mhm+|erm+|ah+)\b |
    [^\x00-\x7F]+        |
    [^\w\s'.,?!-]
    """,
    re.IGNORECASE | re.VERBOSE,
)
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_transcript_text(raw_text: str) -> str:
    """
    lightweight cleaning of a single participant utterance before
    sentence-embedding extraction.

    Steps (conservative — no real words are removed):
      1. Strip XML/HTML-style control tokens  e.g. <synch>, <laughter>
      2. Normalise whitespace
      3. Return empty string if no non-filler alphabetic tokens remain

    Identical logic must be used in both training extraction and runtime
    inference so model training and serving are consistent.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = _TRANSCRIPT_ARTIFACT_RE.sub(" ", raw_text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def clean_participant_utterances(rows: pd.DataFrame, value_col: str = "value") -> str:
    """
    Clean and join all participant utterances for text embedding.

    Args:
        rows      : DataFrame rows for the participant speaker.
        value_col : Column containing utterance text.

    Returns:
        Single cleaned, joined string ready for the sentence encoder.
    """
    if value_col not in rows.columns:
        return ""

    cleaned_parts = []
    for raw in rows[value_col].dropna().astype(str):
        cleaned = clean_transcript_text(raw)
        if cleaned:
            cleaned_parts.append(cleaned)

    return " ".join(cleaned_parts)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def chunk_waveform(waveform_np: np.ndarray, sr: int) -> list[np.ndarray]:
    """
    Split a mono waveform into non-overlapping CHUNK_SEC-second segments.
    Segments shorter than 1 second are discarded.

    Args:
        waveform_np : 1-D float32 numpy array (mono)
        sr          : sample rate

    Returns:
        list of 1-D float32 numpy arrays, each ~CHUNK_SEC * TARGET_SR samples
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


# ---------------------------------------------------------------------------
# Dataset / DataLoader helpers
# ---------------------------------------------------------------------------

class ParticipantSequenceDataset(Dataset):
    """
    One sample = one participant sequence of chunk-level Wav2Vec2 features.

    Expected columns:
      - participant_id
      - chunk_idx
      - label
      - w2v_0 ... w2v_767
      - optional text_0 ... text_n
    """

    def __init__(self, df: pd.DataFrame):
        self.samples = []
        self.participant_ids = []
        self.labels = []

        audio_cols = sorted(
            [c for c in df.columns if c.startswith("w2v_")],
            key=lambda c: int(c.split("_")[1]),
        )
        text_cols = sorted(
            [c for c in df.columns if c.startswith("text_")],
            key=lambda c: int(c.split("_")[1]),
        )

        if not audio_cols:
            raise ValueError("No Wav2Vec2 feature columns found (expected w2v_*).")

        grouped = df.groupby("participant_id")

        for pid, group in grouped:
            group = group.sort_values("chunk_idx")

            if group.empty:
                continue

            audio_features = group[audio_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            audio_features = audio_features.values.astype(np.float32)

            if audio_features.shape[0] == 0:
                continue

            if text_cols:
                text_features = group[text_cols].iloc[0].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                text_features = text_features.values.astype(np.float32)
            else:
                text_features = np.zeros(0, dtype=np.float32)

            label = int(group["label"].iloc[0])

            self.samples.append((audio_features, text_features, label))
            self.participant_ids.append(str(pid))
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, text, y = self.samples[idx]

        return {
            "features": torch.from_numpy(x),
            "text_features": torch.from_numpy(text),
            "label": torch.tensor(float(y), dtype=torch.float32),
            "participant_id": self.participant_ids[idx],
        }

    def apply_standardization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Standardize every participant sequence in-place."""
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        mean = mean.astype(np.float32)

        updated_samples = []
        for features, text_features, label in self.samples:
            features = ((features - mean) / std).astype(np.float32)
            updated_samples.append((features, text_features, label))

        self.samples = updated_samples


def collate_fn(batch):
    """
    Pad variable-length participant sequences to the max sequence length in batch.
    """
    filtered = [item for item in batch if item["features"].shape[0] > 0]

    if not filtered:
        return None

    xs       = [item["features"]      for item in filtered]
    text_xs  = [item["text_features"] for item in filtered]
    ys       = torch.stack([item["label"] for item in filtered])
    pids     = [item["participant_id"] for item in filtered]

    lengths  = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_len  = int(lengths.max().item())
    feat_dim = xs[0].shape[1]

    padded = torch.zeros(len(xs), max_len, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x

    text_dim = text_xs[0].shape[0] if text_xs else 0
    text_padded = torch.zeros(len(text_xs), text_dim, dtype=torch.float32)
    for i, text_x in enumerate(text_xs):
        if text_dim > 0:
            text_padded[i] = text_x

    return padded, text_padded, ys, lengths, pids


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class GRUSequenceClassifier(nn.Module):
    """
    BiGRU + attention pooling — audio-only participant-level classifier.
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
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

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


class MultimodalGRUSequenceClassifier(nn.Module):
    """
    Audio BiGRU + attention fused with a participant-level text embedding.
    Fusion: simple concatenation.
    """

    def __init__(
        self,
        input_dim: int,
        text_dim: int,
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

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ) if text_dim > 0 else None

        fused_dim = hidden_dim * 2 + (hidden_dim if text_dim > 0 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, text_x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if torch.any(lengths == 0):
            raise ValueError("Zero-length sequence detected")

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

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

        audio_context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)

        if self.text_projection is not None:
            text_context = self.text_projection(text_x)
            fused = torch.cat([audio_context, text_context], dim=1)
        else:
            fused = audio_context

        fused = self.dropout(fused)
        return self.classifier(fused)


class MultimodalAttentionPoolingClassifier(nn.Module):
    """
    Project chunk-level audio embeddings, pool them with independent attention
    heads, then fuse the pooled audio context with the participant text embedding.
    """

    def __init__(
        self,
        input_dim: int,
        text_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.45,
        attention_heads: int = 3,
    ):
        super().__init__()

        if attention_heads < 1:
            raise ValueError("attention_heads must be at least 1.")

        self.attention_heads = attention_heads
        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.audio_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, attention_heads),
        )
        self.audio_head_projection = nn.Sequential(
            nn.Linear(hidden_dim * attention_heads, hidden_dim),
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

    def forward(self, x: torch.Tensor, text_x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
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
