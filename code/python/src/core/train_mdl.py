# ========================= Presented by: Reza Saadatyar (2026) ================================
# ============================= E-mail: R.Saadatyar90@gmail.com ==================================

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import evaluate_binary_metrics


def make_dataloader(X, y, batch_size=256, shuffle=True):
    """
    Convert NumPy arrays into a PyTorch DataLoader.
    """
    X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_mdl(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=30,
    batch_size=256,
    learning_rate=1e-3,
    pos_weight=None,
    device=None,
    patience=5,
):
    """
    Train a binary PyTorch contact-classification model.

    Args:
        model       (nn.Module): PyTorch model.
        X_train  (array-like): Training features.
        y_train  (array-like): Training labels.
        X_val    (array-like): Optional validation features.
        y_val    (array-like): Optional validation labels.
        epochs          (int): Maximum number of epochs.
        batch_size      (int): Batch size.
        learning_rate (float): Adam learning rate.
        pos_weight    (float): Optional positive-class weight for BCEWithLogitsLoss.
        device  (torch.device): CUDA or CPU.
        patience        (int): Early-stopping patience.

    Returns:
        model   (nn.Module): Model restored to the best validation state.
        history      (dict): Training and validation history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if pos_weight is not None:
        positive_weight = torch.tensor([float(pos_weight)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    train_loader = make_dataloader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auprc": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # ======================================== Training Loop ========================================
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # -------------------------------- Optional Validation --------------------------------
        if X_val is not None and y_val is not None:
            val_loss, val_metrics = evaluate_mdl(
                model=model,
                X=X_val,
                y=y_val,
                criterion=criterion,
                batch_size=batch_size,
                device=device,
            )

            history["val_loss"].append(val_loss)
            history["val_auprc"].append(val_metrics["auprc"])

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUPRC: {val_metrics['auprc']:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("⏹️ Early stopping triggered.")
                break
        else:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}"
            )

    model.load_state_dict(best_state)
    return model, history


def evaluate_mdl(model, X, y, criterion=None, batch_size=512, device=None):
    """
    Evaluate the model and return binary metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    loader = make_dataloader(
        X,
        y,
        batch_size=batch_size,
        shuffle=False,
    )

    all_probabilities = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            probabilities = torch.sigmoid(logits)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * features.size(0)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probabilities)
    y_true = np.concatenate(all_labels)

    metrics = evaluate_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
    )

    mean_loss = (
        total_loss / len(loader.dataset)
        if criterion is not None
        else np.nan
    )

    return mean_loss, metrics
