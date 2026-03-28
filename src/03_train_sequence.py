import argparse
import copy
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

from pipeline_utils import (
    ParticipantSequenceDataset,
    collate_fn,
    GRUSequenceClassifier,
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = BASE_DIR / "data" / "features_turn_level"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def load_feature_dataframe() -> pd.DataFrame:
    files = sorted(FEATURE_DIR.glob("*_chunk_embeddings.csv"))

    if not files:
        raise FileNotFoundError(
            f"No feature CSVs found in {FEATURE_DIR}. Run 02_extract_features.py first."
        )

    print(f"Loading {len(files)} participant feature files...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df["participant_id"] = df["participant_id"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def standardize_fold_features(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Fit StandardScaler on train-fold chunk features only, then transform both splits.
    """
    feature_cols = sorted(
        [c for c in train_df.columns if c.startswith("w2v_")],
        key=lambda c: int(c.split("_")[1]),
    )

    if not feature_cols:
        raise ValueError("No w2v_* feature columns found for standardization.")

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

    return train_df, val_df


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    preds = (probs >= threshold).astype(np.int32)

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


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        if batch is None:
            continue

        x, y, lengths, _ = batch

        x = x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x, lengths)
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

        x, y, lengths, pids = batch

        x = x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)
        lengths = lengths.to(DEVICE)

        logits = model(x, lengths)
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"LR          : {args.lr}")
    print(f"Folds       : {args.folds}")
    print(f"Hidden dim  : {args.hidden_dim}")
    print(f"Num layers  : {args.num_layers}")
    print(f"Dropout     : {args.dropout}")
    print(f"Patience    : {args.patience}")

    df = load_feature_dataframe()
    dataset = ParticipantSequenceDataset(df)

    if len(dataset) < args.folds:
        raise ValueError(
            f"Not enough participants ({len(dataset)}) for {args.folds}-fold CV."
        )

    input_dim = dataset[0]["features"].shape[1]
    labels = np.array(dataset.labels)

    print(f"\nTotal participants : {len(dataset)}")
    print(f"Depressed          : {(labels == 1).sum()}")
    print(f"Non-depressed      : {(labels == 0).sum()}")
    print(f"Input dim          : {input_dim}")

    skf = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{args.folds}")
        print(f"Train participants : {len(train_idx)}")
        print(f"Val participants   : {len(val_idx)}")

        train_pids = {dataset.participant_ids[i] for i in train_idx}
        val_pids = {dataset.participant_ids[i] for i in val_idx}

        train_df = df[df["participant_id"].isin(train_pids)].copy()
        val_df = df[df["participant_id"].isin(val_pids)].copy()
        train_df, val_df = standardize_fold_features(train_df, val_df)

        fold_train_dataset = ParticipantSequenceDataset(train_df)
        fold_val_dataset = ParticipantSequenceDataset(val_df)

        train_loader = DataLoader(
            fold_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            fold_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = GRUSequenceClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(DEVICE)

        fold_labels = np.array(fold_train_dataset.labels, dtype=np.int32)
        pos_count = int((fold_labels == 1).sum())
        neg_count = int((fold_labels == 0).sum())
        pos_weight = torch.tensor(
            [neg_count / max(1, pos_count)],
            dtype=torch.float32,
            device=DEVICE,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        print(f"pos_weight       : {pos_weight.item():.4f}")

        best_state = None
        best_metrics = None
        best_score = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_metrics = evaluate(model, val_loader, criterion)
            scheduler.step(val_metrics["f1"])

            current_score = val_metrics["f1"]

            if current_score > best_score:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics.copy()
                epochs_without_improvement = 0

                torch.save(best_state, MODEL_DIR / f"best_bigru_fold{fold}.pt")
            else:
                epochs_without_improvement += 1

            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss {train_loss:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} | "
                f"F1 {val_metrics['f1']:.4f} | "
                f"UAR {val_metrics['uar']:.4f} | "
                f"AUC {val_metrics['auc']:.4f} | "
                f"Acc {val_metrics['acc']:.4f} | "
                f"LR {lr_now:.2e} | "
                f"{time.time() - t0:.1f}s"
            )

            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

        if best_state is None or best_metrics is None:
            raise RuntimeError(f"Fold {fold} did not produce a valid checkpoint.")

        pred_df = pd.DataFrame(
            {
                "participant_id": best_metrics["participant_ids"],
                "label": best_metrics["labels"],
                "probability": best_metrics["probs"],
                "prediction": (np.array(best_metrics["probs"]) >= 0.5).astype(int),
                "fold": fold,
            }
        )
        pred_df.to_csv(MODEL_DIR / f"val_predictions_fold{fold}.csv", index=False)

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
            f"Best Fold {fold + 1} | "
            f"F1 {best_metrics['f1']:.4f} | "
            f"UAR {best_metrics['uar']:.4f} | "
            f"AUC {best_metrics['auc']:.4f} | "
            f"Acc {best_metrics['acc']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(MODEL_DIR / "cv_results_sequence.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Cross-validation complete")
    print(f"Macro F1 : {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    print(f"UAR      : {results_df['uar'].mean():.4f} ± {results_df['uar'].std():.4f}")
    print(f"AUC      : {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"Accuracy : {results_df['acc'].mean():.4f} ± {results_df['acc'].std():.4f}")
    print(f"Saved CV summary to: {MODEL_DIR / 'cv_results_sequence.csv'}")


if __name__ == "__main__":
    main()
