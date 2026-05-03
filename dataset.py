# ──────────────────────────────────────────────────────────────────
#  dataset.py  –  Spectrogram generation, Dataset class, Dataloaders
# ──────────────────────────────────────────────────────────────────
#
#  Pipeline (no pre-generation step needed):
#    raw_data/{S,Z,O,N,F}/*.txt
#        → spectrogram generated in memory
#        → normalised RGB tensor
#        → train / val / test DataLoaders via random_split

import os
import random
import io

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")           # non-interactive backend, safe for all platforms
import matplotlib.pyplot as plt
from scipy import signal

from config import (
    resize_x, resize_y, input_channels,
    batch_size, RANDOM_SEED,
    NPERSEG, BONN_CLASS_MAP, RAW_DATA_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)


# ── Reproducibility ───────────────────────────────────────────────
def seed_everything(seed: int = RANDOM_SEED) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU + GPU) for full reproducibility.
    Call this once at the very top of train.py and predict.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Spectrogram generation ────────────────────────────────────────
# Shared transform applied to every spectrogram image.
# Defined once here so dataset.py and predict.py use the exact same
# preprocessing — guarantees train/inference consistency.
SPECTROGRAM_TRANSFORM = transforms.Compose([
    transforms.Resize((resize_y, resize_x)),
    transforms.ToTensor(),                      # PIL → tensor, values in [0, 1]
    transforms.Normalize(                       # ImageNet mean/std — stabilises
        mean=[0.485, 0.456, 0.406],             # training for viridis RGB images
        std=[0.229, 0.224, 0.225]
    ),
])


def eeg_txt_to_tensor(txt_path: str) -> torch.Tensor:
    """
    Convert a raw Bonn EEG .txt file into a normalised RGB spectrogram tensor.

    Steps:
      1. Load raw 1-D signal from .txt (one sample per line).
      2. Compute spectrogram with a fixed window (NPERSEG from config)
         so all signals produce identically-shaped outputs.
      3. Log-scale: log(Sxx + 1e-10) compresses the dynamic range.
      4. Render to an in-memory RGB PNG using the viridis colormap.
         Colour encodes frequency intensity — preserving this is intentional.
      5. Resize to (resize_y, resize_x) and normalise via SPECTROGRAM_TRANSFORM.

    Args:
        txt_path : Absolute or relative path to a Bonn EEG .txt file.

    Returns:
        Tensor of shape (3, resize_y, resize_x) with normalised float values.
    """
    # 1. Load
    raw = np.loadtxt(txt_path)

    # 2. Spectrogram
    _, _, Sxx = signal.spectrogram(raw, nperseg=NPERSEG)

    # 3. Log scale
    Sxx_log = np.log(Sxx + 1e-10)

    # 4. Render to in-memory RGB image (no disk I/O)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.imshow(Sxx_log, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")

    # 5. Resize + normalise
    return SPECTROGRAM_TRANSFORM(img)   # shape: (3, resize_y, resize_x)


# ── Dataset class ─────────────────────────────────────────────────
class EEGSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for the Bonn EEG corpus.

    Reads raw .txt EEG files directly from raw_data/ and generates
    spectrograms on-the-fly. No pre-generation or folder-splitting needed.

    Expected structure:
        raw_data/
            S/   ← seizure       (label 1)
            Z/   ← non-seizure   (label 0)
            O/   ← non-seizure   (label 0)
            N/   ← non-seizure   (label 0)
            F/   ← non-seizure   (label 0)

    The folder → label mapping is defined in BONN_CLASS_MAP (config.py).

    Args:
        root_dir : Path to the raw_data directory.
    """

    def __init__(self, root_dir: str = RAW_DATA_DIR):
        self.samples: list[tuple[str, int]] = []   # (file_path, label)

        for folder, label in BONN_CLASS_MAP.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"Expected folder '{folder_path}' not found.\n"
                    f"Download the Bonn EEG dataset and place it at '{root_dir}/'.\n"
                    f"See README.md for instructions."
                )
            for fname in sorted(os.listdir(folder_path)):
                if fname.upper().endswith(".TXT"):
                    self.samples.append(
                        (os.path.join(folder_path, fname), label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .txt files found in '{root_dir}'. "
                "Check that the Bonn dataset folders contain .txt files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        return eeg_txt_to_tensor(path), label


# ── Dataloader factory ────────────────────────────────────────────
def _stratified_split(
    samples: list[tuple[str, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset indices in a stratified manner — each class is split
    independently so that train / val / test all contain proportional
    representation of seizure and non-seizure samples.

    This prevents the random bad luck of val/test sets having too few
    (or zero) seizure samples, which caused unstable val accuracy.

    Args:
        samples     : List of (file_path, label) from EEGSpectrogramDataset.
        train_ratio : Fraction of data for training.
        val_ratio   : Fraction of data for validation.
        seed        : Random seed for reproducibility.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = random.Random(seed)

    # Group indices by class label
    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(samples):
        class_indices.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for label, indices in class_indices.items():
        indices = indices.copy()
        rng.shuffle(indices)

        n       = len(indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        # remainder goes to test

        train_idx.extend(indices[:n_train])
        val_idx.extend(  indices[n_train : n_train + n_val])
        test_idx.extend( indices[n_train + n_val :])

    # Shuffle each split so batches aren't all-one-class
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def get_dataloaders(
    root_dir: str    = RAW_DATA_DIR,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build stratified train / val / test DataLoaders from the raw Bonn dataset.

    Each class (seizure / non-seizure) is split independently so every
    split contains a proportional mix — prevents val/test sets from
    having too few seizure samples by random chance.

    Ratios (from config.py):
        Train : Val : Test = 70% : 15% : 15%

    Args:
        root_dir   : Path to raw_data directory.
        num_workers: Parallel workers for loading. Keep 0 on Windows.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    full_dataset = EEGSpectrogramDataset(root_dir=root_dir)

    train_idx, val_idx, test_idx = _stratified_split(
        samples     = full_dataset.samples,
        train_ratio = TRAIN_RATIO,
        val_ratio   = VAL_RATIO,
        seed        = RANDOM_SEED,
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    test_ds  = Subset(full_dataset, test_idx)

    def _worker_init(worker_id):
        np.random.seed(RANDOM_SEED + worker_id)

    shared = dict(num_workers=num_workers, worker_init_fn=_worker_init,
                  pin_memory=torch.cuda.is_available())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **shared)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **shared)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **shared)

    return train_loader, val_loader, test_loader


# ── Aliases for interface.py ──────────────────────────────────────
TheDataset     = EEGSpectrogramDataset
the_dataloader = get_dataloaders


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    seed_everything()
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # Verify stratification — each split should have both classes
    from collections import Counter
    for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        all_labels = [label for _, labels in loader for label in labels.tolist()]
        c = Counter(all_labels)
        print(f"{name:5s} class dist → non-seizure: {c[0]:3d}  seizure: {c[1]:3d}")

    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape : {imgs.shape}")    # expect (32, 3, 128, 128)
    print(f"Labels      : {labels.tolist()}")