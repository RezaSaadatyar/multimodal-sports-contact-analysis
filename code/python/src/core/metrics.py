# ========================= Presented by: Reza Saadatyar (2026) ================================
# ============================= E-mail: R.Saadatyar90@gmail.com ==================================

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute class-aware metrics for contact-event detection.

    Args:
        y_true    (array-like): Ground-truth binary labels.
        y_prob    (array-like): Predicted positive-class probabilities.
        threshold     (float): Probability threshold for class prediction.

    Returns:
        dict: Accuracy, balanced accuracy, sensitivity/recall, specificity,
              precision/PPV, F1, AUROC, AUPRC, MCC, FPR and FNR.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = precision_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": auroc,
        "auprc": auprc,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }
