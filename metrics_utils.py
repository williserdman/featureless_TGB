from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor


def compute_auroc_ap(*, y_true: Tensor, y_pred: Tensor) -> tuple[float, float]:
    """Compute micro-averaged AUROC and AP over all labels.

    Returns NaN when the metric is undefined (for example no positives or one class only).
    """

    true_np = y_true.detach().cpu().float().numpy()
    pred_np = y_pred.detach().cpu().float().numpy()

    if true_np.ndim == 1:
        true_np = true_np.reshape(-1, 1)
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(-1, 1)

    true_flat = true_np.reshape(-1)
    pred_flat = pred_np.reshape(-1)

    finite = np.isfinite(pred_flat)
    if not np.all(finite):
        pred_flat = np.where(finite, pred_flat, 0.5)

    # TGBn labels are relevance scores for ranking; convert to binary relevance
    # for supplemental AUROC/AP diagnostics.
    true_binary = (true_flat > 0).astype(np.int32)

    auroc = float("nan")
    ap = float("nan")

    if np.unique(true_binary).size >= 2:
        auroc = float(roc_auc_score(true_binary, pred_flat))

    if np.sum(true_binary) > 0:
        ap = float(average_precision_score(true_binary, pred_flat))

    return auroc, ap
