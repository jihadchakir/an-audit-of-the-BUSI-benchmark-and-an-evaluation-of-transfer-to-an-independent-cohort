from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats as sps


def old_preprocess(img_u8: np.ndarray, mask_u8, size: int = 128,
                   mode: str = "soft_dim", pad: float = 0.10,
                   normal_fallback: str = "full_image") -> np.ndarray:
    """Reproduce a mask-using preprocessing.

    mode="soft_dim"  the original notebooks' variant, out = img*m + 0.3*img*(1-m)
    mode="roi_crop"  the FIELD'S published practice: crop to the ground-truth
                     lesion bounding box, pad, resize. A `normal` case has an
                     all-black mask and therefore no bounding box, so it takes
                     the fallback path. That asymmetry is the whole point.
    """
    if mode == "roi_crop":
        if mask_u8 is not None and mask_u8.max() > 0:
            if mask_u8.shape[:2] != img_u8.shape[:2]:
                mask_u8 = cv2.resize(mask_u8, (img_u8.shape[1], img_u8.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask_u8 > 0)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            ph, pw = int((y1 - y0 + 1) * pad), int((x1 - x0 + 1) * pad)
            y0 = max(0, y0 - ph); y1 = min(img_u8.shape[0] - 1, y1 + ph)
            x0 = max(0, x0 - pw); x1 = min(img_u8.shape[1] - 1, x1 + pw)
            img_u8 = img_u8[y0:y1 + 1, x0:x1 + 1]
        elif normal_fallback == "center_crop":
            h, w = img_u8.shape[:2]
            side = int(min(h, w) * 0.7)
            img_u8 = img_u8[(h - side) // 2:(h - side) // 2 + side,
                            (w - side) // 2:(w - side) // 2 + side]
        return cv2.resize(img_u8, (size, size)).astype(np.float32) / 255.0

    img = cv2.resize(img_u8, (size, size))
    if mask_u8 is not None:
        mask = cv2.resize(mask_u8, (size, size))
        m3 = np.stack([mask] * 3, axis=-1) / 255.0
        img = img * m3 + img * 0.3 * (1 - m3)
    return np.asarray(img, dtype=np.float32) / 255.0


def new_preprocess(img_u8: np.ndarray, size: int = 128) -> np.ndarray:
    """The rebuild's input: no mask, ever."""
    return cv2.resize(img_u8, (size, size)).astype(np.float32) / 255.0


def summary_features(x: np.ndarray) -> np.ndarray:
    """Global statistics only. No spatial structure, no texture model.

    A model built on these cannot read an ultrasound. It can only notice that
    the pixel distribution of one class looks different from another.
    """
    g = x.mean(axis=2) if x.ndim == 3 else x
    q = np.percentile(g, [1, 5, 25, 50, 75, 95, 99])
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return np.array([
        g.mean(), g.std(), g.min(), g.max(),
        *q,
        float(sps.skew(g.ravel())), float(sps.kurtosis(g.ravel())),
        mag.mean(), mag.std(), np.percentile(mag, 99),
        np.abs(lap).mean(), np.abs(lap).std(),
        float((g > 0.3).mean()),      # cf. the discarded identity, kept as a feature
        float((g < 0.05).mean()),
    ], dtype=np.float64)


def build(df: pd.DataFrame, size: int, mode: str = "soft_dim",
          normal_fallback: str = "full_image") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    F_old, F_new, y, groups = [], [], [], []
    mask_stats = {"n_with_mask": 0, "n_all_black": 0, "per_class_all_black": {}}

    for _, r in df.iterrows():
        img = cv2.imread(r["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp = [p for p in str(r.get("mask_paths", "")).split("|") if p]
        mask = cv2.imread(mp[0], cv2.IMREAD_GRAYSCALE) if mp else None

        if mask is not None:
            mask_stats["n_with_mask"] += 1
            if mask.max() == 0:
                mask_stats["n_all_black"] += 1
                c = r["class_name"]
                mask_stats["per_class_all_black"][c] = mask_stats["per_class_all_black"].get(c, 0) + 1

        F_old.append(summary_features(old_preprocess(img, mask, size, mode,
                                                     normal_fallback=normal_fallback)))
        F_new.append(summary_features(new_preprocess(img, size)))
        y.append(int(r["label"]))
        groups.append(int(r["group"]))

    return (np.stack(F_old), np.stack(F_new), np.array(y), np.array(groups), mask_stats)


def probe(F: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> Dict:
    """Group-aware CV balanced accuracy of a trivial model."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(y)
    for tr, te in cv.split(F, y, groups):
        pipe = Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        pipe.fit(F[tr], y[tr])
        y_pred[te] = pipe.predict(F[te])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "accuracy": float((y_pred == y).mean()),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "per_class_recall": {
            int(c): float((y_pred[y == c] == c).mean()) for c in np.unique(y)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./runs/audit/index_with_groups.csv")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="./runs/audit/mask_leak_report.json")
    ap.add_argument("--mask-mode", default="soft_dim", choices=["soft_dim", "roi_crop"],
                    help="'roi_crop' = the field's published practice")
    ap.add_argument("--roi-normal-fallback", default="full_image",
                    choices=["full_image", "center_crop"])
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    classes = sorted(df["class_name"].unique(), key=lambda c: df[df.class_name == c].label.iloc[0])
    print(f"Loaded {len(df)} images, classes {classes}\n")

    print(f"Mask mode under test: {args.mask_mode}"
          + (f" (normals -> {args.roi_normal_fallback})" if args.mask_mode == "roi_crop" else ""))
    F_old, F_new, y, groups, mstat = build(df, args.size, args.mask_mode, args.roi_normal_fallback)

    print("=" * 74)
    print("1. FACTS ABOUT THE MASKS")
    print("=" * 74)
    print(f"  images with a mask file      : {mstat['n_with_mask']}/{len(df)}")
    print(f"  masks that are entirely black: {mstat['n_all_black']}")
    print(f"  all-black masks by class     : {mstat['per_class_all_black']}")
    print("\n  For an all-black mask the old formula reduces exactly to out = 0.3*img,")
    print("  a uniform darkening. For a lesion case, the region inside the expert's")
    print("  outline keeps full intensity. The two classes were preprocessed by")
    print("  different functions, selected by the label.")

    print("\n" + "=" * 74)
    print("2. TRIVIAL-MODEL PROBE  (logistic regression on 20 global statistics)")
    print("=" * 74)
    print("  This model has no spatial reasoning. It cannot read an ultrasound.")
    print("  Anything above chance is information leaking through image statistics.")
    print("  CV is group-aware, so duplicates do not contaminate the probe itself.\n")

    r_old = probe(F_old, y, groups)
    r_new = probe(F_new, y, groups)
    chance = 1.0 / len(classes)

    lbl = "ROI CROP" if args.mask_mode == "roi_crop" else "SOFT-DIM"
    print(f"  {'':28s} {lbl:>12s} {'NEW (no mask)':>14s}")
    print(f"  {'balanced accuracy':28s} {r_old['balanced_accuracy']*100:11.2f}% "
          f"{r_new['balanced_accuracy']*100:13.2f}%")
    for i, c in enumerate(classes):
        print(f"  {'recall: ' + c:28s} {r_old['per_class_recall'].get(i, 0)*100:11.2f}% "
              f"{r_new['per_class_recall'].get(i, 0)*100:13.2f}%")
    print(f"\n  chance balanced accuracy: {chance*100:.2f}%")

    gap = r_old["balanced_accuracy"] - r_new["balanced_accuracy"]
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"\n  gap attributable to the mask: {gap*100:+.2f} balanced-accuracy points\n")
    if gap > 0.10:
        print("  The mask injects substantial class information recoverable by a model")
        print("  that cannot interpret ultrasound. Part of the published accuracy was")
        print("  this artefact. Quantify the rest with the full ablation.")
    elif gap > 0.03:
        print("  The mask injects a modest but real amount of class information.")
        print("  Report this number; let the full ablation settle the magnitude.")
    else:
        print("  Low-order statistics carry little extra information from the mask.")
        print("  That does NOT rescue the old preprocessing: a CNN sees the contour,")
        print("  which this probe cannot. And the fatal objection is untouched, since")
        print("  the mask is an expert annotation unavailable at inference time.")
    print("\n  Either way, the decisive experiment is the ablation: train the real")
    print("  model with use_gt_mask_oracle on and off, and compare. This probe is a")
    print("  lower bound on the artefact, not a measurement of it.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"mask_facts": mstat, "old_preprocessing": r_old, "new_preprocessing": r_new,
         "gap_balanced_accuracy": float(gap), "chance": chance}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
