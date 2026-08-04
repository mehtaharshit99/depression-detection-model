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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from pipeline_utils import (
    ParticipantSequenceDataset,
    collate_fn,
    MultimodalAttentionPoolingClassifier,
    MultimodalGRUSequenceClassifier,
)

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE_DIR / "data" / "features_multimodal"
MODEL_DIR = BASE_DIR / "models"
MODEL_PREFIX = "best_multimodal_fold"
SCALER_PATH = MODEL_DIR / "multimodal_inference_scaler.pkl"
CV_RESULTS_PATH = MODEL_DIR / "cv_results_multimodal.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multimodal depression classifier.")
    parser.add_argument("--feature_dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--model", choices=["attention_pooling", "gru"], default="attention_pooling")
    parser.add_argument("--attention_heads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_feature_dir(path: Path) -> Path:
    return path if path.is_absolute() else BASE_DIR / path


def sorted_feature_columns(df: pd.DataFrame):
    audio_cols = sorted(
        [c for c in df.columns if c.startswith("w2v_")],
        key=lambda c: int(c.split("_")[1]),
    )
    text_cols = sorted(
        [c for c in df.columns if c.startswith("text_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not audio_cols:
        raise ValueError("No w2v_* feature columns found.")
    return audio_cols, text_cols, audio_cols + text_cols


def load_feature_dataframe(feature_dir: Path) -> pd.DataFrame:
    files = sorted(feature_dir.glob("*_multimodal_embeddings.csv"))
    if not files:
        raise FileNotFoundError(
            f"No feature CSVs found in {feature_dir}. Run src/02_extract_features.py first."
        )

    print(f"Loading {len(files)} participant feature files...")
    df = pd.concat([pd.read_csv(file_path) for file_path in files], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df["participant_id"] = df["participant_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def standardize_fold_features(train_df: pd.DataFrame, val_df: pd.DataFrame):
    audio_cols, text_cols, feature_cols = sorted_feature_columns(train_df)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float32))

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols].values.astype(np.float32))
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    return train_df, val_df, scaler, feature_cols, len(audio_cols), len(text_cols)


def save_inference_scaler(df: pd.DataFrame, args) -> None:
    audio_cols, text_cols, feature_cols = sorted_feature_columns(df)
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values.astype(np.float32))

    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(
            {
                "scaler": scaler,
                "feature_cols": feature_cols,
                "audio_dim": len(audio_cols),
                "text_dim": len(text_cols),
                "model_type": args.model,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "attention_heads": args.attention_heads,
                "seed": args.seed,
            },
            fh,
        )


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= threshold).astype(np.int32)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "uar": recall_score(labels, preds, average="macro", zero_division=0),
        "auc": safe_auc(labels, probs),
    }


def safe_auc(labels, probs) -> float:
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return 0.0


def build_model(args, input_dim: int, text_dim: int) -> nn.Module:
    if args.model == "attention_pooling":
        return MultimodalAttentionPoolingClassifier(
            input_dim=input_dim,
            text_dim=text_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            attention_heads=args.attention_heads,
        )

    return MultimodalGRUSequenceClassifier(
        input_dim=input_dim,
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


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

    print(f"Device          : {DEVICE}")
    print(f"Feature dir     : {feature_dir}")
    print(f"Model           : {args.model}")
    print(f"Attention heads : {args.attention_heads if args.model == 'attention_pooling' else 'n/a'}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"LR              : {args.lr}")
    print(f"Folds           : {args.folds}")
    print(f"Hidden dim      : {args.hidden_dim}")
    print(f"Dropout         : {args.dropout}")
    print(f"Patience        : {args.patience}")
    print(f"Seed            : {args.seed}")

    df = load_feature_dataframe(feature_dir)
    dataset = ParticipantSequenceDataset(df)
    if len(dataset) < args.folds:
        raise ValueError(f"Not enough participants ({len(dataset)}) for {args.folds}-fold CV.")

    input_dim = dataset[0]["features"].shape[1]
    text_dim = dataset[0]["text_features"].shape[0]
    labels = np.asarray(dataset.labels, dtype=int)
    participant_ids = np.asarray(dataset.participant_ids)

    print()
    print(f"Total participants : {len(dataset)}")
    print(f"Depressed          : {(labels == 1).sum()}")
    print(f"Non-depressed      : {(labels == 0).sum()}")
    print(f"Audio input dim    : {input_dim}")
    print(f"Text input dim     : {text_dim}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(participant_ids, labels)):
        print()
        print("=" * 60)
        print(f"Fold {fold + 1}/{args.folds} | train={len(train_idx)} val={len(val_idx)}")

        train_ids = set(participant_ids[train_idx])
        val_ids = set(participant_ids[val_idx])
        train_df = df[df["participant_id"].isin(train_ids)]
        val_df = df[df["participant_id"].isin(val_ids)]
        train_df, val_df, _, _, _, _ = standardize_fold_features(train_df, val_df)

        train_ds = ParticipantSequenceDataset(train_df)
        val_ds = ParticipantSequenceDataset(val_df)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = build_model(args, input_dim, text_dim).to(DEVICE)
        fold_labels = np.asarray(train_ds.labels, dtype=np.int32)
        pos_count = int((fold_labels == 1).sum())
        neg_count = int((fold_labels == 0).sum())
        pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

        best_state = None
        best_metrics = None
        best_score = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_metrics = evaluate(model, val_loader, criterion)
            current_score = val_metrics["macro_f1"]

            if current_score > best_score:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch:02d} | TrLoss {train_loss:.4f} | ValLoss {val_metrics['loss']:.4f} | "
                f"F1 {val_metrics['macro_f1']:.4f} | UAR {val_metrics['uar']:.4f} | "
                f"AUC {val_metrics['auc']:.4f} | Acc {val_metrics['accuracy']:.4f} | "
                f"{time.time() - t0:.1f}s"
            )
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

        if best_state is None or best_metrics is None:
            raise RuntimeError(f"Fold {fold} did not produce a valid checkpoint.")

        checkpoint = {
            "model_state": best_state,
            "model_type": args.model,
            "input_dim": input_dim,
            "text_dim": text_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "attention_heads": args.attention_heads,
            "seed": args.seed,
        }
        torch.save(checkpoint, MODEL_DIR / f"{MODEL_PREFIX}{fold}.pt")

        pred_df = pd.DataFrame(
            {
                "participant_id": best_metrics["participant_ids"],
                "label": best_metrics["labels"],
                "probability": best_metrics["probs"],
                "prediction": (np.asarray(best_metrics["probs"]) >= 0.5).astype(int),
                "fold": fold,
            }
        )
        pred_df.to_csv(MODEL_DIR / f"val_predictions_multimodal_fold{fold}.csv", index=False)

        fold_results.append(
            {
                "fold": fold,
                "accuracy": best_metrics["accuracy"],
                "macro_f1": best_metrics["macro_f1"],
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "uar": best_metrics["uar"],
                "auc": best_metrics["auc"],
            }
        )
        print(
            f"Best Fold {fold + 1} | F1 {best_metrics['macro_f1']:.4f} | "
            f"Precision {best_metrics['precision']:.4f} | Recall {best_metrics['recall']:.4f} | "
            f"UAR {best_metrics['uar']:.4f} | AUC {best_metrics['auc']:.4f} | "
            f"Acc {best_metrics['accuracy']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(CV_RESULTS_PATH, index=False)
    save_inference_scaler(df, args)

    print()
    print("=" * 60)
    print("Cross-validation complete")
    print(f"Accuracy : {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
    print(f"Macro F1 : {results_df['macro_f1'].mean():.4f} +/- {results_df['macro_f1'].std():.4f}")
    print(f"Precision: {results_df['precision'].mean():.4f} +/- {results_df['precision'].std():.4f}")
    print(f"Recall   : {results_df['recall'].mean():.4f} +/- {results_df['recall'].std():.4f}")
    print(f"UAR      : {results_df['uar'].mean():.4f} +/- {results_df['uar'].std():.4f}")
    print(f"AUC      : {results_df['auc'].mean():.4f} +/- {results_df['auc'].std():.4f}")
    print(f"Saved CV summary      : {CV_RESULTS_PATH}")
    print(f"Saved inference scaler: {SCALER_PATH}")


if __name__ == "__main__":
    main()
