# ──────────────────────────────────────────────────────────────────
#  model.py  –  CNN architecture for EEG spectrogram classification
# ──────────────────────────────────────────────────────────────────
#
#  Input : RGB spectrogram tensor of shape (batch, 3, 128, 128)
#  Output: Logits of shape (batch, 2)  →  [non-seizure, seizure]

import torch
import torch.nn as nn

from config import input_channels, resize_x, resize_y


class EEG_CNN(nn.Module):
    """
    Lightweight custom CNN for binary EEG spectrogram classification.

    Architecture:
        Block 1 : Conv(3→32)  → BN → ReLU → MaxPool   → Dropout
        Block 2 : Conv(32→64) → BN → ReLU → MaxPool   → Dropout
        Block 3 : Conv(64→128)→ BN → ReLU → MaxPool   → Dropout
        Head    : AdaptiveAvgPool → Flatten → FC(128→64) → ReLU → FC(64→2)

    Design choices:
        - Batch Normalisation after every conv layer stabilises training
          and reduces sensitivity to learning rate.
        - AdaptiveAvgPool(1,1) before the FC head makes the architecture
          input-size agnostic — resize_x/resize_y can be changed in
          config.py without breaking the model.
        - Dropout (p=0.5) on conv outputs + FC hidden layer to regularise
          a dataset of only 500 samples.
        - Two output logits fed into CrossEntropyLoss (in train.py),
          which internally applies softmax — no activation here.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Convolutional blocks ──────────────────────────────────
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 128 → 64
            nn.Dropout2d(p=0.25),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64 → 32
            nn.Dropout2d(p=0.25),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 → 16
            nn.Dropout2d(p=0.25),
        )

        # ── Classification head ───────────────────────────────────
        # AdaptiveAvgPool collapses spatial dims to (1,1) regardless
        # of input resolution → 128 features into the FC layers.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),                        # 2 logits: [non-seizure, seizure]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_channels, resize_y, resize_x).

        Returns:
            Logits tensor of shape (batch, 2).
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# ── Alias for interface.py ────────────────────────────────────────
TheModel = EEG_CNN


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    model = EEG_CNN()
    print(model)
    print()

    dummy = torch.zeros(4, input_channels, resize_y, resize_x)
    logits = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {logits.shape}")   # expect (4, 2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")