# ──────────────────────────────────────────────────────────────────
#  train.py  –  Training loop for EEG spectrogram CNN
# ──────────────────────────────────────────────────────────────────
#
#  Usage:
#      python train.py
#
#  What this file does:
#      1. Seeds everything for reproducibility.
#      2. Builds train / val / test dataloaders from raw_data/.
#      3. Instantiates EEG_CNN and a weighted loss function.
#      4. Runs the training loop, tracking loss and accuracy.
#      5. Saves the best model (by val accuracy) to checkpoints/.
#      6. Plots and saves training curves after training completes.
#      7. Runs a final evaluation on the test set.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")           # non-interactive — safe on all platforms
import matplotlib.pyplot as plt

from config import (
    epochs, learning_rate, CLASS_WEIGHTS,
    CHECKPOINT_DIR, WEIGHTS_FILE, RANDOM_SEED,
)
from dataset import seed_everything, get_dataloaders
from model import EEG_CNN


# ── Training loop ─────────────────────────────────────────────────
def train_model(
    model:        nn.Module,
    num_epochs:   int,
    train_loader: torch.utils.data.DataLoader,
    loss_fn:      nn.Module,
    optimizer:    torch.optim.Optimizer,
    val_loader:   torch.utils.data.DataLoader = None,
    device:       torch.device               = None,
) -> dict:
    """
    Run the full training loop with optional validation each epoch.

    Tracks per-epoch train loss, train accuracy, and (if val_loader
    is provided) validation accuracy. Saves the best checkpoint to
    WEIGHTS_FILE whenever validation accuracy improves.

    Args:
        model        : EEG_CNN instance (or any nn.Module).
        num_epochs   : Total epochs to train.
        train_loader : DataLoader for the training split.
        loss_fn      : Loss function (CrossEntropyLoss with class weights).
        optimizer    : Optimiser (Adam).
        val_loader   : DataLoader for the validation split (optional).
        device       : torch.device. Auto-detected if None.

    Returns:
        Dictionary with keys:
            "train_losses"    – list of average loss per epoch
            "train_accuracies"– list of train accuracy per epoch
            "val_accuracies"  – list of val accuracy per epoch (empty if no val_loader)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_losses:     list[float] = []
    train_accuracies: list[float] = []
    val_accuracies:   list[float] = []
    best_val_acc: float = 0.0

    print(f"Training on : {device}")
    print(f"Epochs      : {num_epochs}")
    print(f"Train batches: {len(train_loader)}  |  "
          f"Val batches: {len(val_loader) if val_loader else 'N/A'}")
    print("─" * 60)

    for epoch in range(1, num_epochs + 1):

        # ── Training phase ────────────────────────────────────────
        model.train()
        running_loss    = 0.0
        correct_train   = 0
        total_train     = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item() * imgs.size(0)
            preds          = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

        epoch_loss     = running_loss  / total_train
        epoch_train_acc = correct_train / total_train * 100
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)

        # ── Validation phase ──────────────────────────────────────
        epoch_val_acc = 0.0
        if val_loader is not None:
            model.eval()
            correct_val = 0
            total_val   = 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits       = model(imgs)
                    preds        = logits.argmax(dim=1)
                    correct_val += (preds == labels).sum().item()
                    total_val   += labels.size(0)

            epoch_val_acc = correct_val / total_val * 100
            val_accuracies.append(epoch_val_acc)

            # Save best checkpoint
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), WEIGHTS_FILE)
                saved_marker = "  ✓ saved"
            else:
                saved_marker = ""

            print(
                f"Epoch [{epoch:>3}/{num_epochs}]  "
                f"Loss: {epoch_loss:.4f}  "
                f"Train Acc: {epoch_train_acc:.1f}%  "
                f"Val Acc: {epoch_val_acc:.1f}%"
                f"{saved_marker}"
            )
        else:
            print(
                f"Epoch [{epoch:>3}/{num_epochs}]  "
                f"Loss: {epoch_loss:.4f}  "
                f"Train Acc: {epoch_train_acc:.1f}%"
            )

    print("─" * 60)
    print(f"Training complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Weights saved to : {WEIGHTS_FILE}")

    return {
        "train_losses":     train_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies":   val_accuracies,
    }


# ── Plot training curves ──────────────────────────────────────────
def plot_training_curves(history: dict, save_path: str = "training_curves.png") -> None:
    """
    Plot and save loss + accuracy curves from the training history dict.

    Args:
        history  : Dict returned by train_model().
        save_path: Where to save the figure.
    """
    epochs_range = range(1, len(history["train_losses"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs_range, history["train_losses"], label="Train Loss", marker="o")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(epochs_range, history["train_accuracies"], label="Train Accuracy", marker="o")
    if history["val_accuracies"]:
        ax2.plot(epochs_range, history["val_accuracies"], label="Val Accuracy", marker="s")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


# ── Test set evaluation ───────────────────────────────────────────
def evaluate_on_test(
    model:       nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device:      torch.device = None,
) -> None:
    """
    Load the best saved weights and evaluate on the test set.
    Prints overall accuracy and a confusion matrix breakdown.

    Args:
        model      : EEG_CNN instance (architecture must match saved weights).
        test_loader: DataLoader for the test split.
        device     : torch.device. Auto-detected if None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model.to(device)
    model.eval()

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds        = model(imgs).argmax(dim=1)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    total    = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0.0

    print("\n── Test Set Results ──────────────────────────────────")
    print(f"  Accuracy : {accuracy:.1f}%  ({tp + tn}/{total} correct)")
    print(f"  TP (seizure correctly detected)    : {tp}")
    print(f"  TN (non-seizure correctly detected): {tn}")
    print(f"  FP (non-seizure called seizure)    : {fp}")
    print(f"  FN (seizure missed)                : {fn}")
    print("─" * 54)


# ── Alias for interface.py ────────────────────────────────────────
the_trainer = train_model


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    seed_everything(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Model
    model = EEG_CNN()

    # Weighted loss — upweights seizure (minority class)
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # Optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    history = train_model(
        model        = model,
        num_epochs   = epochs,
        train_loader = train_loader,
        loss_fn      = loss_fn,
        optimizer    = optimizer,
        val_loader   = val_loader,
        device       = device,
    )

    # Plot training curves
    plot_training_curves(history)

    # Evaluate on test set using best saved weights
    evaluate_on_test(model, test_loader, device)