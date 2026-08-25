from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MAL = 1  # class order is ["benign", "malignant"]


def _balanced_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean([(yhat[y == c] == c).mean() for c in np.unique(y)]))


def two_class_argmax_operating_point() -> None:
    ba, sens, spec, ppv = [], [], [], []
    for i in range(1, 6):
        f = np.load(ROOT / f"runs/rebuild_2class/fold_{i}/test_predictions.npz", allow_pickle=True)
        y, proba = f["y_true"], f["proba"]
        yhat = proba.argmax(1)  # plain argmax, no abstention, no tuned threshold
        tp = int(((yhat == MAL) & (y == MAL)).sum())
        fn = int(((yhat != MAL) & (y == MAL)).sum())
        tn = int(((yhat != MAL) & (y != MAL)).sum())
        fp = int(((yhat == MAL) & (y != MAL)).sum())
        ba.append(_balanced_accuracy(y, yhat))
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
        ppv.append(tp / (tp + fp))

    def ms(x):
        return np.mean(x), np.std(x, ddof=1)

    print("Table 1, two-class (plain argmax), mean +/- SD over 5 folds:")
    print("  AUROC                  see runs/rebuild_2class/summary.json (0.887 +/- 0.018)")
    print("  Balanced accuracy      %.3f +/- %.3f" % ms(ba))
    print("  Recall (sens), malig   %.1f%% +/- %.1f" % tuple(v * 100 for v in ms(sens)))
    print("  Specificity, malig     %.1f%% +/- %.1f" % tuple(v * 100 for v in ms(spec)))
    print("  PPV, malig             %.1f%% +/- %.1f" % tuple(v * 100 for v in ms(ppv)))


def _auroc_by_device(report: dict) -> dict:
    """Pull per-device AUROC from a BUS-BRA fold report, whatever the nesting."""
    out = {}

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "auroc" and isinstance(v, (int, float)) and "Device/" in path:
                    out[path.split("Device/")[1].split("/")[0]] = v
                walk(v, path + "/" + k)
        elif isinstance(o, list):
            for x in o:
                walk(x, path)

    walk(report)
    return out


def vendor_stratification() -> None:
    dev: dict[str, list] = {}
    overall = []
    for i in range(1, 6):
        rep = json.load(open(ROOT / f"runs/external/BUS-BRA_fold{i}_report.json"))
        overall.append(rep["frozen_transfer"]["auroc"])
        for name, a in _auroc_by_device(rep).items():
            dev.setdefault(name, []).append(a)

    print("\nSupplementary vendor table, mean +/- SD over 5 folds:")
    for name, vals in dev.items():
        if len(vals) == 5:
            print("  %-30s %.3f +/- %.3f" % (name, np.mean(vals), np.std(vals, ddof=1)))
    print("  %-30s %.3f +/- %.3f" % ("All devices", np.mean(overall), np.std(overall, ddof=1)))


if __name__ == "__main__":
    two_class_argmax_operating_point()
    vendor_stratification()
