# ──────────────────────────────────────────────────────────────────
#  interface.py  –  Standardised entry points for the grading program
# ──────────────────────────────────────────────────────────────────
#
#  This file does NOT contain any logic.
#  It only imports from the other project files and re-exports them
#  under the standardised names the grading program expects.
#
#  DO NOT modify the names on the right-hand side of each `as` clause.
#  If you rename anything in the source files, update the left-hand
#  side (the import name) to match — never the right-hand side.

# The model architecture
from model import EEG_CNN as TheModel

# The function that runs the training loop
from train import train_model as the_trainer

# The function that runs inference on a list of raw .txt file paths
from predict import classify_eeg as the_predictor

# The custom Dataset class
from dataset import EEGSpectrogramDataset as TheDataset

# The function that returns (train_loader, val_loader, test_loader)
from dataset import get_dataloaders as the_dataloader

# Training hyperparameters
from config import batch_size as the_batch_size
from config import epochs as total_epochs