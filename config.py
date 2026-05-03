# ─────────────────────────────────────────────
#  config.py  –  All hyperparameters & settings
# ─────────────────────────────────────────────
#
#  Every other file imports from here.
#  Change a value here and it propagates everywhere.
#  Nothing is hardcoded anywhere else.

# ── Bonn EEG dataset folder → binary label mapping ───────────────
#    S = ictal (seizure)          → label 1
#    Z, O, N, F = non-seizure     → label 0
BONN_CLASS_MAP = {
    "S": 1,   # seizure
    "Z": 0,   # healthy, eyes open
    "O": 0,   # healthy, eyes closed
    "N": 0,   # interictal, opposite hemisphere
    "F": 0,   # interictal, epileptogenic zone
}

# ── Image dimensions ──────────────────────────────────────────────
resize_x       = 128    # width  (pixels) fed into the CNN
resize_y       = 128    # height (pixels) fed into the CNN
input_channels = 3      # RGB — viridis colormap encodes frequency intensity

# ── Spectrogram ───────────────────────────────────────────────────
NPERSEG = 256           # scipy spectrogram window size
                        # fixed so every signal produces identical dimensions

# ── Training hyperparameters ──────────────────────────────────────
batch_size     = 32
epochs         = 20
learning_rate  = 0.001

# ── Class imbalance weights ───────────────────────────────────────
#    ~400 non-seizure vs ~100 seizure samples in the Bonn dataset.
#    Upweighting seizure (minority class) prevents the model from
#    ignoring it and predicting non-seizure for everything.
#    Format: [weight_for_class_0, weight_for_class_1]
CLASS_WEIGHTS  = [1.0, 4.0]

# ── Dataset split ratios ──────────────────────────────────────────
#    Split is done in code via random_split (no folder pre-splitting).
#    Must sum to 1.0.
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15

# ── Paths ─────────────────────────────────────────────────────────
RAW_DATA_DIR   = "raw_data"                   # full Bonn dataset (.gitignored)
CHECKPOINT_DIR = "checkpoints"
WEIGHTS_FILE   = "checkpoints/final_weights.pth"

# ── Reproducibility ───────────────────────────────────────────────
RANDOM_SEED    = 42