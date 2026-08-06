"""Evaluation utilities: metrics, confusion, and the late-fusion hybrid."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)


def metrics_from_proba(proba, y_true, class_names=None):
    """Compute accuracy + macro precision/recall/F1 and per-class report."""
    y_pred = proba.argmax(1)
    n_classes = proba.shape[1]
    labels = list(range(n_classes))
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels)
    return {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "report": classification_report(
            y_true, y_pred,
            target_names=class_names, zero_division=0,
            labels=list(range(proba.shape[1]))),
        "y_pred": y_pred,
    }


# --------------------------------------------------------------------------
# Late-fusion hybrid (report Section 6.3): weighted soft-voting, no training
# --------------------------------------------------------------------------
def fuse(proba_cnn, proba_lstm, w_cnn=0.55):
    """Weighted average of the two softmax vectors."""
    return w_cnn * proba_cnn + (1.0 - w_cnn) * proba_lstm


def tune_fusion_weight(proba_cnn_val, proba_lstm_val, y_val, grid=None):
    """Scan the CNN weight on the validation set; return (best_w, best_acc, table)."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    table = []
    best_w, best_acc = 0.5, -1.0
    for w in grid:
        acc = accuracy_score(y_val, fuse(proba_cnn_val, proba_lstm_val, w).argmax(1))
        table.append((float(w), float(acc)))
        if acc > best_acc:
            best_acc, best_w = acc, float(w)
    return best_w, best_acc, table
