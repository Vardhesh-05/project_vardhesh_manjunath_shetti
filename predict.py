# ──────────────────────────────────────────────────────────────────
#  predict.py  –  Inference on raw Bonn EEG .txt files
# ──────────────────────────────────────────────────────────────────
#
#  Usage (from terminal):
#      python predict.py data/S/S001.txt data/non_seizure/Z001.txt
#
#  Or import and call directly:
#      from predict import classify_eeg
#      results = classify_eeg(["data/seizure/S001.txt", "data/non_seizure/Z001.txt"])

import os
import sys
import torch
import torch.nn.functional as F

from config import WEIGHTS_FILE, RANDOM_SEED
from dataset import seed_everything, eeg_txt_to_tensor
from model import EEG_CNN

# ── Label map ─────────────────────────────────────────────────────
LABEL_NAMES = {0: "Non-Seizure", 1: "Seizure"}


# ── Load model once at module level ───────────────────────────────
# Loading once here means repeated calls to classify_eeg() don't
# reload weights from disk every time — important for efficiency
# when the grader calls predict.py on multiple files.
def _load_model(device: torch.device) -> EEG_CNN:
    """
    Load EEG_CNN with saved weights from WEIGHTS_FILE.

    Args:
        device: torch.device to load the model onto.

    Returns:
        EEG_CNN in eval mode with best trained weights loaded.

    Raises:
        FileNotFoundError: If WEIGHTS_FILE does not exist.
                           Run train.py first to generate weights.
    """
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            f"No weights found at '{WEIGHTS_FILE}'.\n"
            "Please run train.py first to train the model."
        )
    model = EEG_CNN()
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model.to(device)
    model.eval()
    return model


# ── Inference function ────────────────────────────────────────────
def classify_eeg(
    list_of_txt_paths: list[str],
    verbose: bool = True,
) -> list[dict]:
    """
    Run inference on a list of raw Bonn EEG .txt files.

    Each file is converted to a spectrogram tensor using the exact
    same preprocessing pipeline as training (eeg_txt_to_tensor from
    dataset.py) — guaranteeing train/inference consistency.

    Args:
        list_of_txt_paths : List of paths to raw EEG .txt files.
                            These should be files from the data/ directory
                            or anywhere in the raw Bonn dataset.
        verbose           : If True, prints prediction results to stdout.

    Returns:
        List of dicts, one per input file, each containing:
            {
                "file"       : str   – original file path,
                "label"      : int   – predicted class (0 or 1),
                "prediction" : str   – "Seizure" or "Non-Seizure",
                "confidence" : float – probability of predicted class (0–1),
            }

    Example:
        >>> results = classify_eeg(["data/seizure/S001.txt"])
        >>> print(results[0]["prediction"])
        'Seizure'
    """
    seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_model(device)

    results = []

    for txt_path in list_of_txt_paths:
        if not os.path.exists(txt_path):
            print(f"  [WARNING] File not found, skipping: {txt_path}")
            continue

        # Convert raw .txt → spectrogram tensor (same pipeline as training)
        tensor = eeg_txt_to_tensor(txt_path)          # (3, 128, 128)
        tensor = tensor.unsqueeze(0).to(device)        # (1, 3, 128, 128)

        with torch.no_grad():
            logits      = model(tensor)                # (1, 2)
            probs       = F.softmax(logits, dim=1)     # (1, 2)
            label       = probs.argmax(dim=1).item()   # 0 or 1
            confidence  = probs[0, label].item()       # probability of predicted class

        result = {
            "file"       : txt_path,
            "label"      : label,
            "prediction" : LABEL_NAMES[label],
            "confidence" : round(confidence, 4),
        }
        results.append(result)

        if verbose:
            print(
                f"  {os.path.basename(txt_path):<20s}  →  "
                f"{LABEL_NAMES[label]:<15s}  "
                f"(confidence: {confidence:.1%})"
            )

    return results


# ── Alias for interface.py ────────────────────────────────────────
the_predictor = classify_eeg


# ── CLI entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_eeg.txt> [<path2.txt> ...]")
        print("Example: python predict.py data/seizure/S001.txt data/non_seizure/Z001.txt")
        sys.exit(1)

    txt_paths = sys.argv[1:]
    print(f"\nRunning inference on {len(txt_paths)} file(s)...\n")
    classify_eeg(txt_paths, verbose=True)