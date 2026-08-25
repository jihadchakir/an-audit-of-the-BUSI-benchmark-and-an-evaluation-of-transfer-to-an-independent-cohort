from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

CLASSES = ["benign", "malignant", "normal"]


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------
def scan_busi(root: str | Path) -> pd.DataFrame:
    """Index the BUSI directory.

    Fixes vs the original loader:
      * sorted() everywhere, so the file order is deterministic. The old code
        relied on os.listdir() order, which is filesystem dependent, so a
        `train_test_split(random_state=42)` in one notebook did NOT reproduce
        the same split in another notebook. See CHANGES.md #4.
      * collects ALL masks (BUSI has `_mask_1.png`, `_mask_2.png` for some
        multi-lesion cases); the old code silently used only `_mask.png`.
    """
    root = Path(root)
    rows: List[Dict] = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*.png")):
            if "_mask" in img_path.name:
                continue
            stem = img_path.stem
            masks = sorted(class_dir.glob(f"{stem}_mask*.png"))
            rows.append(
                {
                    "path": str(img_path),
                    "filename": img_path.name,
                    "class_name": class_name,
                    "label": class_idx,
                    "n_masks": len(masks),
                    "mask_paths": "|".join(str(m) for m in masks),
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No images found under {root}. Expected {root}/benign, {root}/malignant, {root}/normal."
        )
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def content_md5(path: str) -> str:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "UNREADABLE"
    return hashlib.md5(img.tobytes()).hexdigest()


def phash(path: str, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Perceptual hash (DCT based). Returns a boolean vector of length hash_size**2.

    Robust to the rescaling / recompression that produced BUSI's near-duplicates.
    """
    img_size = hash_size * highfreq_factor
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(hash_size * hash_size, dtype=bool)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(img.astype(np.float32))
    low = dct[:hash_size, :hash_size].flatten()
    # exclude the DC term from the median so uniform brightness shifts do not
    # dominate the hash
    med = np.median(low[1:])
    return low > med


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """bits: (N, B) boolean -> (N, N) int hamming distances."""
    b = bits.astype(np.int16)
    # ||a - b||_1 for boolean vectors == hamming distance
    return (b[:, None, :] != b[None, :, :]).sum(-1)


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(df: pd.DataFrame, threshold: int = 6) -> Tuple[pd.DataFrame, List[Dict]]:
    """Assign a `group` id: connected components of the near-duplicate graph.

    BUSI has no patient identifiers, so true patient-wise splitting is
    impossible. Duplicate-group-wise splitting is the strongest available
    proxy and must be declared as a limitation in the paper.
    """
    print(f"[audit] hashing {len(df)} images ...")
    md5s = np.array([content_md5(p) for p in df["path"]])
    bits = np.stack([phash(p) for p in df["path"]])

    dist = hamming_matrix(bits)
    n = len(df)
    uf = _UnionFind(n)
    pairs: List[Dict] = []

    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        exact = md5s[i] == md5s[j] and md5s[i] != "UNREADABLE"
        near = dist[i, j] <= threshold
        if exact or near:
            uf.union(int(i), int(j))
            pairs.append(
                {
                    "file_a": df.at[i, "filename"],
                    "file_b": df.at[j, "filename"],
                    "class_a": df.at[i, "class_name"],
                    "class_b": df.at[j, "class_name"],
                    "phash_hamming": int(dist[i, j]),
                    "exact_pixel_duplicate": bool(exact),
                    "label_conflict": bool(df.at[i, "label"] != df.at[j, "label"]),
                }
            )

    roots = [uf.find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    df = df.copy()
    df["group"] = [remap[r] for r in roots]
    df["content_md5"] = md5s
    return df, pairs


# --------------------------------------------------------------------------
# the mask artefact
# --------------------------------------------------------------------------
def old_apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Verbatim reimplementation of the original notebooks' preprocessing."""
    mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
    return img * mask_3ch + img * 0.3 * (1 - mask_3ch)


def mask_artifact_report(df: pd.DataFrame, img_size: int = 128) -> Dict:
    """Measure how much class information the old mask step injected.

    For a BUSI "normal" image the mask is entirely black, so
        out = img*0 + img*0.3*(1-0) = 0.3 * img
    i.e. the whole image is uniformly darkened. For benign/malignant the lesion
    region keeps full intensity. A model can therefore separate "normal" from
    the rest using global brightness alone, with no anatomy involved. This
    function reports the effect size.
    """
    stats: Dict[str, Dict] = {}
    for class_name in CLASSES:
        sub = df[df["class_name"] == class_name]
        if len(sub) == 0:
            continue
        raw_means, old_means, blank_masks = [], [], 0
        for _, row in sub.iterrows():
            img = cv2.imread(row["path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
            raw_means.append(float(img.mean()))

            mask_paths = [p for p in str(row["mask_paths"]).split("|") if p]
            if mask_paths:
                mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (img_size, img_size))
                    if mask.max() == 0:
                        blank_masks += 1
                    old_means.append(float(old_apply_mask(img, mask).mean()))

        stats[class_name] = {
            "n": int(len(sub)),
            "n_all_black_masks": int(blank_masks),
            "frac_all_black_masks": float(blank_masks / max(len(sub), 1)),
            "mean_intensity_raw": float(np.mean(raw_means)) if raw_means else None,
            "mean_intensity_old_preprocessing": float(np.mean(old_means)) if old_means else None,
            "std_intensity_old_preprocessing": float(np.std(old_means)) if old_means else None,
        }

    # How separable are the classes using ONE scalar (mean brightness) after the
    # old preprocessing? If this is high, the old headline accuracy was partly
    # measuring the annotation pipeline, not the anatomy.
    sep = None
    if all(c in stats and stats[c]["mean_intensity_old_preprocessing"] is not None for c in CLASSES):
        normal_mu = stats["normal"]["mean_intensity_old_preprocessing"]
        lesion_mus = [stats[c]["mean_intensity_old_preprocessing"] for c in ("benign", "malignant")]
        pooled_sd = np.mean([stats[c]["std_intensity_old_preprocessing"] for c in CLASSES])
        sep = {
            "cohens_d_normal_vs_lesion_brightness": float(
                abs(normal_mu - np.mean(lesion_mus)) / (pooled_sd + 1e-8)
            ),
            "WARNING": (
                "DO NOT INTERPRET THIS NUMBER. Mean brightness is a diluted proxy: "
                "lesion masks cover a small fraction of the frame and the between-image "
                "variance in scanner gain swamps the difference. On real BUSI this "
                "returns ~0.68, which badly understates the artefact. A trivial-model "
                "probe on the same images recovers the 'normal' class at 100% recall. "
                "Use diagnose_mask_leak.py, which measures the artefact empirically "
                "instead of assuming which statistic carries it."
            ),
        }
    return {"per_class": stats, "separability": sep}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./Dataset_BUSI")
    ap.add_argument("--out", default="./runs/audit")
    ap.add_argument("--phash-threshold", type=int, default=6)
    ap.add_argument("--skip-mask-report", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = scan_busi(args.root)
    print(f"[audit] indexed {len(df)} images")
    print(df["class_name"].value_counts().to_string())

    df, pairs = build_groups(df, threshold=args.phash_threshold)
    pairs_df = pd.DataFrame(pairs)

    n_groups = df["group"].nunique()
    multi = df.groupby("group").size()
    report = {
        "n_images": int(len(df)),
        "class_counts": df["class_name"].value_counts().to_dict(),
        "n_duplicate_groups": int(n_groups),
        "n_images_in_multi_image_groups": int((multi[multi > 1]).sum()),
        "largest_group_size": int(multi.max()),
        "n_exact_pixel_duplicate_pairs": int(pairs_df["exact_pixel_duplicate"].sum()) if len(pairs_df) else 0,
        "n_near_duplicate_pairs": int(len(pairs_df)),
        "n_pairs_with_label_conflict": int(pairs_df["label_conflict"].sum()) if len(pairs_df) else 0,
        "n_images_with_multiple_masks": int((df["n_masks"] > 1).sum()),
        "phash_threshold": args.phash_threshold,
    }

    if not args.skip_mask_report:
        print("[audit] measuring the ground-truth-mask artefact ...")
        report["mask_artifact"] = mask_artifact_report(df)

    df.to_csv(out / "index_with_groups.csv", index=False)
    pairs_df.to_csv(out / "duplicate_pairs.csv", index=False)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)
    print(json.dumps({k: v for k, v in report.items() if k != "mask_artifact"}, indent=2))
    if "mask_artifact" in report and report["mask_artifact"]["separability"]:
        d = report["mask_artifact"]["separability"]["cohens_d_normal_vs_lesion_brightness"]
        blk = report["mask_artifact"]["per_class"].get("normal", {}).get("n_all_black_masks", 0)
        n_norm = report["mask_artifact"]["per_class"].get("normal", {}).get("n", 0)
        print(f"\nNormal cases with an all-black mask: {blk}/{n_norm}")
        if n_norm and blk == n_norm:
            print("  -> 'mask is empty' is EXACTLY equivalent to 'class is normal' in this dataset.")
            print("     The old preprocessing applied a different function to each class,")
            print("     selected by the label.")
        print(f"\n(Cohen's d on mean brightness = {d:.2f}. Ignore it: mean brightness is a")
        print(" diluted proxy and understates the artefact. Run diagnose_mask_leak.py.)")
    print(f"\nWrote {out}/index_with_groups.csv, duplicate_pairs.csv, audit_report.json")


if __name__ == "__main__":
    main()
