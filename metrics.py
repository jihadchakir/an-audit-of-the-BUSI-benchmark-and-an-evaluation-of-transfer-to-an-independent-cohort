from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------
def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, positive: int) -> Dict[str, float]:
    pos = y_true == positive
    pred_pos = y_pred == positive
    tp = int((pos & pred_pos).sum())
    fn = int((pos & ~pred_pos).sum())
    tn = int((~pos & ~pred_pos).sum())
    fp = int((~pos & pred_pos).sum())
    return {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    class_names: Sequence[str],
    malignant_index: int,
    answered: Optional[np.ndarray] = None,
) -> Dict:
    """Full metric block for one split.

    If `answered` is given, accuracy-style metrics are computed on answered
    cases only and `coverage` records what fraction that was. Discrimination
    metrics (AUC) are computed on all cases, since they do not depend on the
    decision rule.
    """
    n_classes = len(class_names)
    out: Dict = {"n": int(len(y_true))}

    if answered is None:
        answered = np.ones(len(y_true), dtype=bool)
    out["coverage"] = float(answered.mean())
    out["n_answered"] = int(answered.sum())

    yt, yp = y_true[answered], y_pred[answered]
    if len(yt) == 0:
        return out

    out["accuracy"] = float((yt == yp).mean())
    out["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))
    out["macro_f1"] = float(f1_score(yt, yp, average="macro", labels=list(range(n_classes)), zero_division=0))
    out["confusion_matrix"] = confusion_matrix(yt, yp, labels=list(range(n_classes))).tolist()
    out["malignant"] = sensitivity_specificity(yt, yp, malignant_index)

    per_class = {}
    for c, name in enumerate(class_names):
        per_class[name] = sensitivity_specificity(yt, yp, c)
    out["per_class"] = per_class

    # discrimination, decision-rule independent, all cases
    try:
        if n_classes == 2:
            out["auroc"] = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            out["auroc_ovr_macro"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                              labels=list(range(n_classes)))
            )
        mal_bin = (y_true == malignant_index).astype(int)
        if 0 < mal_bin.sum() < len(mal_bin):
            out["malignant_auroc"] = float(roc_auc_score(mal_bin, proba[:, malignant_index]))
            out["malignant_auprc"] = float(average_precision_score(mal_bin, proba[:, malignant_index]))
            out["malignant_brier"] = float(brier_score_loss(mal_bin, proba[:, malignant_index]))
    except ValueError as exc:
        out["auroc_error"] = str(exc)

    out["ece"] = expected_calibration_error(y_true, proba)
    out["brier_multiclass"] = multiclass_brier(y_true, proba, n_classes)
    return out


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1337,
    stratified: bool = True,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Percentile bootstrap CI.

    stratified=True resamples within each true class, which keeps the class
    prevalence of the resamples equal to the observed one. That is the right
    choice for sensitivity/specificity (they are conditional on the true class)
    and it is what should be stated in the caption for accuracy too, since
    accuracy is then conditioned on the observed prevalence.

    `groups` switches to a CLUSTER bootstrap: whole groups are resampled, not
    individual images. Use it whenever images are not independent, which is
    almost always in this project:

      * BUS-BRA has 1875 images from 1064 patients. 811 patients contribute two
        views of the same lesion. Resampling images pretends those 1875 are
        independent draws and shrinks every interval by roughly sqrt(1875/1064),
        about 30%.
      * BUSI has duplicate groups from the audit, same argument.

    An interval that is 30% too narrow is exactly the kind of thing a reviewer
    checks when the rest of the paper has already been questioned.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: List[float] = []

    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        # each group gets one stratum, its modal true class
        g_stratum = np.array([np.bincount(y_true[members[g]]).argmax() for g in uniq])
        pools = ([uniq[g_stratum == s] for s in np.unique(g_stratum)]
                 if stratified else [uniq])
        for _ in range(n_boot):
            picked = np.concatenate([rng.choice(p, size=len(p), replace=True) for p in pools])
            idx = np.concatenate([members[g] for g in picked])
            try:
                stats.append(metric_fn(y_true[idx], y_pred[idx], proba[idx]))
            except ValueError:
                continue
    else:
        if stratified:
            class_idx = [np.where(y_true == c)[0] for c in np.unique(y_true)]
        for _ in range(n_boot):
            if stratified:
                idx = np.concatenate([rng.choice(ci, size=len(ci), replace=True) for ci in class_idx])
            else:
                idx = rng.integers(0, n, size=n)
            try:
                stats.append(metric_fn(y_true[idx], y_pred[idx], proba[idx]))
            except ValueError:
                continue

    if not stats:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}

    arr = np.array(stats, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "point": float(metric_fn(y_true, y_pred, proba)),
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n_boot": int(len(arr)),
        "unit": "cluster" if groups is not None else "image",
        "n_clusters": int(len(np.unique(groups))) if groups is not None else None,
    }


def standard_ci_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    malignant_index: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1337,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """The CIs that belong in the main results table.

    Pass `groups` (patient id, or duplicate-group id) unless the images really
    are independent. They are not, in either cohort used here.
    """
    fns = {
        "accuracy": lambda t, p, _: float((t == p).mean()),
        "balanced_accuracy": lambda t, p, _: float(balanced_accuracy_score(t, p)),
        "macro_f1": lambda t, p, _: float(f1_score(t, p, average="macro", zero_division=0)),
        "malignant_sensitivity": lambda t, p, _: sensitivity_specificity(t, p, malignant_index)["sensitivity"],
        "malignant_specificity": lambda t, p, _: sensitivity_specificity(t, p, malignant_index)["specificity"],
        "malignant_ppv": lambda t, p, _: sensitivity_specificity(t, p, malignant_index)["ppv"],
    }
    out = {}
    for name, fn in fns.items():
        out[name] = bootstrap_ci(fn, y_true, y_pred, proba, n_boot=n_boot,
                                 alpha=alpha, seed=seed, groups=groups)
    return out


def subgroup_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    subgroup: np.ndarray,
    class_names: Sequence[str],
    malignant_index: int,
    min_n: int = 20,
) -> Dict[str, Dict]:
    """Metrics sliced by a metadata column (scanner, BI-RADS, histology...).

    For external validation this is the interesting analysis, not the headline
    number. BUSI is single-vendor (GE LOGIQ E9). If accuracy holds on BUS-BRA's
    GE scanners but collapses on the Toshiba Aplio 300, that is a scanner-shift
    finding worth reporting, and it is invisible in a pooled average.
    """
    out: Dict[str, Dict] = {}
    for level in np.unique(subgroup):
        m = subgroup == level
        if m.sum() < min_n:
            out[str(level)] = {"n": int(m.sum()), "skipped": f"fewer than {min_n} cases"}
            continue
        out[str(level)] = compute_metrics(
            y_true[m], y_pred[m], proba[m], class_names, malignant_index
        )
    return out


def format_ci(d: Dict[str, float], pct: bool = True) -> str:
    k = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    return f"{d['point']*k:.1f}{unit} [{d['lo']*k:.1f}, {d['hi']*k:.1f}]"


def aggregate_folds(fold_metrics: List[Dict], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Mean +/- SD across outer folds, for the per-fold table."""
    out = {}
    for k in keys:
        vals = []
        for m in fold_metrics:
            v = m
            for part in k.split("."):
                v = v.get(part) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_folds": len(vals),
            }
    return out
