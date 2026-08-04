"""
09_train_attention_pooling.py - lightweight multimodal attention-pooling model

Trains a participant-level classifier that attends directly over projected
chunk embeddings, then fuses the resulting audio context with the transcript
embedding. This keeps chunk-level information without the recurrent GRU layer.
"""

import argparse
import copy
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pipeline_utils import ParticipantSequenceDataset, collate_fn

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE_DIR / "data" / "features_multimodal"
MODEL_DIR = BASE_DIR / "models"
RESULTS_PATH = MODEL_DIR / "cv_results_attention_pooling.csv"
MODEL_PATH = MODEL_DIR / "attention_pooling_multimodal.pkl"
CHECKPOINT_PREFIX = "attention_pooling"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionPoolingClassifier(nn.Module):
    def __init__(self, input_dim: int, text_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.audio_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.audio_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
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

        attn_scores = self.audio_attention(projected).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        audio_context = torch.bmm(attn_weights.unsqueeze(1), projected).squeeze(1)

        if self.text_projection is not None:
            text_context = self.text_projection(text_x)
            fused = torch.cat([audio_context, text_context], dim=1)
        else:
            fused = audio_context

        return self.classifier(fused)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_feature_dir(path: Path) -> Path:
    return path if path.is_absolute() else BASE_DIR / path


def load_feature_dataframe(feature_dir: Path) -> pd.DataFrame:
    files = sorted(feature_dir.glob("*_multimodal_embeddings.csv"))
    if not files:
        raise FileNotFoundError(
            f"No feature CSVs found in {feature_dir}. Run 02_extract_features.py first."
        )

    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df["participant_id"] = df["participant_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def standardize_fold_features(train_df: pd.DataFrame, val_df: pd.DataFrame):
    w2v_cols = sorted(
        [c for c in train_df.columns if c.startswith("w2v_")],
        key=lambda c: int(c.split("_")[1]),
    )
    text_cols = sorted(
        [c for c in train_df.columns if c.startswith("text_")],
        key=lambda c: int(c.split("_")[1]),
    )
    feature_cols = w2v_cols + text_cols
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float32))

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df.loc[:, feature_cols] = scaler.transform(
        train_df[feature_cols].values.astype(np.float32)
    )
    val_df.loc[:, feature_cols] = scaler.transform(
        val_df[feature_cols].values.astype(np.float32)
    )
    return train_df, val_df, scaler, feature_cols, len(w2v_cols), len(text_cols)


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= 0.5).astype(np.int32)
    metrics = {
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "uar": recall_score(labels, preds, average="macro", zero_division=0),
        "acc": accuracy_score(labels, preds),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        if batch is None:
            continue
        x, text_x, y, lengths, _ = batch
        x = x.to(DEVICE)
        text_x = text_x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, text_x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_probs = []
    all_labels = []
    all_pids = []
    for batch in loader:
        if batch is None:
            continue
        x, text_x, y, lengths, pids = batch
        x = x.to(DEVICE)
        text_x = text_x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)
        lengths = lengths.to(DEVICE)

        logits = model(x, text_x, lengths)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        total_loss += loss.item()
        total_batches += 1
        all_probs.extend(probs.tolist())
        all_labels.extend(y.squeeze(1).cpu().numpy().tolist())
        all_pids.extend(pids)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(1, total_batches)
    metrics["participant_ids"] = all_pids
    metrics["labels"] = all_labels
    metrics["probs"] = all_probs
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    feature_dir = resolve_feature_dir(args.feature_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_dataframe(feature_dir)
    dataset = ParticipantSequenceDataset(df)
    labels = np.asarray(dataset.labels, dtype=int)
    participant_ids = np.asarray(dataset.participant_ids)

    if len(dataset) < args.folds:
        raise ValueError(f"Not enough participants ({len(dataset)}) for {args.folds}-fold CV.")

    input_dim = dataset[0]["features"].shape[1]
    text_dim = dataset[0]["text_features"].shape[0]

    print(f"Device        : {DEVICE}")
    print(f"Feature dir   : {feature_dir}")
    print(f"Participants  : {len(dataset)}")
    print(f"Depressed     : {(labels == 1).sum()}")
    print(f"Non-depressed : {(labels == 0).sum()}")
    print(f"Audio dim     : {input_dim}")
    print(f"Text dim      : {text_dim}")
    print(f"Hidden dim    : {args.hidden_dim}")
    print(f"Dropout       : {args.dropout}")
    print(f"Seed          : {args.seed}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    last_scaler_payload = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(participant_ids, labels)):
        print("\n" + "=" * 60)
        print(f"Fold {fold + 1}/{args.folds} | train={len(train_idx)} val={len(val_idx)}")

        train_ids = set(participant_ids[train_idx])
        val_ids = set(participant_ids[val_idx])
        train_df = df[df["participant_id"].isin(train_ids)]
        val_df = df[df["participant_id"].isin(val_ids)]
        train_df, val_df, scaler, feature_cols, audio_dim, text_dim = standardize_fold_features(
            train_df,
            val_df,
        )
        last_scaler_payload = {
            "scaler": scaler,
            "feature_cols": feature_cols,
            "audio_dim": audio_dim,
            "text_dim": text_dim,
        }

        train_ds = ParticipantSequenceDataset(train_df)
        val_ds = ParticipantSequenceDataset(val_df)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = AttentionPoolingClassifier(input_dim, text_dim, args.hidden_dim, args.dropout).to(DEVICE)
        pos_count = int((np.asarray(train_ds.labels) == 1).sum())
        neg_count = int((np.asarray(train_ds.labels) == 0).sum())
        pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

        best_state = None
        best_metrics = None
        best_score = -1.0
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_metrics = evaluate(model, val_loader, criterion)

            if val_metrics["f1"] > best_score:
                best_score = val_metrics["f1"]
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics.copy()
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"Epoch {epoch:02d} | TrLoss {train_loss:.4f} | "
                f"ValLoss {val_metrics['loss']:.4f} | F1 {val_metrics['f1']:.4f} | "
                f"UAR {val_metrics['uar']:.4f} | AUC {val_metrics['auc']:.4f} | "
                f"Acc {val_metrics['acc']:.4f} | {time.time() - t0:.1f}s"
            )
            if no_improve >= args.patience:
                print("Early stopping triggered.")
                break

        torch.save(
            {
                "model_state": best_state,
                "input_dim": input_dim,
                "text_dim": text_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "seed": args.seed,
            },
            MODEL_DIR / f"{CHECKPOINT_PREFIX}_fold{fold}.pt",
        )

        pred_df = pd.DataFrame(
            {
                "participant_id": best_metrics["participant_ids"],
                "label": best_metrics["labels"],
                "probability": best_metrics["probs"],
                "prediction": (np.asarray(best_metrics["probs"]) >= 0.5).astype(int),
                "fold": fold,
            }
        )
        pred_df.to_csv(MODEL_DIR / f"val_predictions_{CHECKPOINT_PREFIX}_fold{fold}.csv", index=False)

        fold_results.append(
            {
                "fold": fold,
                "f1": best_metrics["f1"],
                "uar": best_metrics["uar"],
                "auc": best_metrics["auc"],
                "acc": best_metrics["acc"],
            }
        )
        print(
            f"Best Fold {fold + 1} | F1 {best_metrics['f1']:.4f} | "
            f"UAR {best_metrics['uar']:.4f} | AUC {best_metrics['auc']:.4f} | "
            f"Acc {best_metrics['acc']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(RESULTS_PATH, index=False)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(
            {
                "input_dim": input_dim,
                "text_dim": text_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "seed": args.seed,
                "scaler": last_scaler_payload,
            },
            fh,
        )

    print("\nCross-validation complete")
    print(f"Macro F1 : {results_df['f1'].mean():.4f} +/- {results_df['f1'].std():.4f}")
    print(f"UAR      : {results_df['uar'].mean():.4f} +/- {results_df['uar'].std():.4f}")
    print(f"AUC      : {results_df['auc'].mean():.4f} +/- {results_df['auc'].std():.4f}")
    print(f"Accuracy : {results_df['acc'].mean():.4f} +/- {results_df['acc'].std():.4f}")
    print(f"Saved CV summary : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
