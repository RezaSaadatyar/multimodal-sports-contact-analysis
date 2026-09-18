# ========================= Presented by: Reza Saadatyar (2026) ================================
# ============================= E-mail: R.Saadatyar90@gmail.com ==================================
# Input(batch, features) → BatchNorm → Linear(64) → ReLU → Dropout → Linear(32) → ReLU → Linear(1)

import torch
import torch.nn as nn


class TrackingMLP(nn.Module):
    """
    Simple PyTorch baseline for binary contact-event classification.

    The model is intentionally small because the first objective is to build
    a transparent tracking-only baseline before introducing video features.

    Args:
        n_features (int): Number of input tracking features.
        dropout    (float): Dropout probability.
    """

    def __init__(self, n_features, dropout=0.3):
        super(TrackingMLP, self).__init__()

        self.network = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Shape (batch, n_features).

        Returns:
            torch.Tensor: Raw logits with shape (batch,).
        """
        logits = self.network(x)
        return logits.squeeze(-1)


def build_tracking_mlp(n_features, dropout=0.3):
    """
    Factory function for the tracking-only baseline.
    """
    return TrackingMLP(
        n_features=n_features,
        dropout=dropout,
    )
