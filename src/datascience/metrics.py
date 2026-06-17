"""Metric, calibration and uncertainty helpers (task-agnostic, numpy/sklearn).

Each is a small standalone function so the evaluator just composes them. To add
a metric, add a branch in `compute_scalar_metrics` and list it in
`params.yaml: metrics.compute`.
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_scalar_metrics(names, y_true, y_pred, probs=None) -> dict:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef
    from sklearn.preprocessing import label_binarize

    out = {}
    for n in names:
        if n == "accuracy":
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
        elif n == "macro_f1":
            out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif n == "micro_f1":
            out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        elif n == "mcc":
            out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        elif n == "pr_auc":
            if probs is None:
                continue
            n_classes = probs.shape[1]
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            if n_classes == 2:
                y_bin = np.hstack([1 - y_bin, y_bin])
            out["pr_auc"] = float(average_precision_score(y_bin, probs, average="macro"))
    return out


def per_class_report(y_true, y_pred, id2label):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    ids = sorted(id2label)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    report = {
        id2label[cid]: {
            "precision": float(p[k]),
            "recall": float(r[k]),
            "f1": float(f[k]),
            "support": int(s[k]),
        }
        for k, cid in enumerate(ids)
    }
    cm = confusion_matrix(y_true, y_pred, labels=ids).tolist()
    return report, cm, [id2label[i] for i in ids]


def expected_calibration_error(probs, y_true, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(val_logits, val_labels, max_iter: int = 200) -> float:
    """Temperature scaling: fit a single scalar T>0 on val by minimizing NLL."""
    import torch

    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.clamp(min=1e-3).item())


def bootstrap_ci(metric_fn, y_true, y_pred, n: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI for a metric on the test set."""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y_true))
    vals = [metric_fn(y_true[idx], y_pred[idx]) for idx in (rng.choice(idx_all, len(idx_all), replace=True) for _ in range(n))]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
