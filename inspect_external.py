from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Published figures for sanity-checking a third-party mirror (Kaggle, etc.)
# against the official Zenodo release.
# Gomez-Flores et al., Med Phys 51:3110-3123, 2024.
KNOWN_COHORTS: Dict[str, Dict] = {
    "BUS-BRA": {
        "n_images": 1875,
        "n_patients": 1064,
        "pathology": {"benign": 1268, "malignant": 607},
        "birads": {"2": 562, "3": 463, "4": 693, "5": 157},
        "citation": "Gomez-Flores W, Gregorio-Calas MJ, Pereira WCA. Med Phys 51:3110-3123, 2024. doi:10.1002/mp.16812",
        "official_source": "https://zenodo.org/records/8231412",
    },
    "BrEaST": {
        "n_images": 256,
        "n_patients": 256,
        "citation": "Pawlowska A, et al. Sci Data, 2024. (TCIA: Breast-Lesions-USG)",
        "official_source": "https://www.cancerimagingarchive.net/",
    },
}


def walk_tree(root: Path, max_depth: int = 3) -> None:
    print(f"\n{'='*72}\nDIRECTORY STRUCTURE\n{'='*72}")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        depth = len(rel.parts) - 1
        if depth > max_depth:
            continue
        if path.is_dir():
            n_files = sum(1 for _ in path.glob("*") if _.is_file())
            exts = Counter(p.suffix.lower() for p in path.glob("*") if p.is_file())
            ext_s = ", ".join(f"{v} {k or '(none)'}" for k, v in exts.most_common(4))
            print(f"{'  '*depth}{rel}/   [{n_files} files: {ext_s}]")


def inspect_csvs(root: Path) -> List[Path]:
    csvs = sorted(root.rglob("*.csv"))
    print(f"\n{'='*72}\nMETADATA FILES ({len(csvs)} found)\n{'='*72}")
    for c in csvs:
        try:
            df = pd.read_csv(c)
        except Exception as exc:
            print(f"\n{c.relative_to(root)}: UNREADABLE ({exc})")
            continue
        print(f"\n--- {c.relative_to(root)}  ({len(df)} rows, {len(df.columns)} cols)")
        print(f"    columns: {list(df.columns)}")
        for col in df.columns:
            nun = df[col].nunique()
            if nun <= 12:
                vals = df[col].value_counts().to_dict()
                print(f"      {col:20s} ({nun:4d} unique)  {vals}")
            else:
                sample = df[col].dropna().astype(str).head(3).tolist()
                print(f"      {col:20s} ({nun:4d} unique)  e.g. {sample}")
    return csvs


def inspect_images(root: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {"images": [], "masks": []}
    for p in root.rglob("*.png"):
        key = "masks" if ("mask" in p.stem.lower() or "mask" in p.parent.name.lower()) else "images"
        groups[key].append(p)
    for p in root.rglob("*.jpg"):
        groups["images"].append(p)

    print(f"\n{'='*72}\nIMAGES\n{'='*72}")
    for k, v in groups.items():
        print(f"  {k:8s}: {len(v)}")
        if v:
            print(f"            e.g. {[p.name for p in sorted(v)[:4]]}")
    if groups["masks"]:
        print("\n  NOTE: this cohort ships segmentation masks. Do NOT use them as model")
        print("  input. That is defect #1 from CHANGES.md and it would invalidate the")
        print("  external validation exactly the way it invalidated the internal result.")
    return groups


def try_match(meta: pd.DataFrame, id_col: str, images: List[Path]) -> None:
    print(f"\n{'='*72}\nID -> FILENAME RESOLUTION  (using --id-col '{id_col}')\n{'='*72}")
    if id_col not in meta.columns:
        print(f"  '{id_col}' is not a column. Available: {list(meta.columns)}")
        return

    by_stem = {p.stem: p for p in images}
    hits, misses = 0, []
    for raw in meta[id_col].astype(str):
        key = raw.strip()
        found = key in by_stem
        if not found:
            for cand in (key.replace(".png", ""), f"bus_{key}", key.zfill(4)):
                if cand in by_stem:
                    found = True
                    break
        hits += found
        if not found and len(misses) < 5:
            misses.append(key)

    pct = 100.0 * hits / max(len(meta), 1)
    print(f"  resolved: {hits}/{len(meta)} ({pct:.1f}%)")
    if misses:
        print(f"  unresolved examples: {misses}")
        print(f"  image stems look like: {[p.stem for p in sorted(images)[:4]]}")
        print("  -> adjust --id-col, or extend the fallback list in "
              "external_eval.build_external_index")
    elif hits == len(meta):
        print("  every metadata row resolves to an image. Plumbing is good.")


def check_against_published(name: str, meta: Optional[pd.DataFrame], images: List[Path],
                            label_col: Optional[str], patient_col: Optional[str]) -> None:
    known = KNOWN_COHORTS.get(name)
    if not known:
        return
    print(f"\n{'='*72}\nMIRROR INTEGRITY CHECK vs PUBLISHED {name}\n{'='*72}")
    print(f"  cite: {known['citation']}")
    print(f"  official source: {known['official_source']}")

    ok = True
    n_img = len(images)
    print(f"\n  images:   found {n_img}, published {known['n_images']}  "
          f"{'OK' if n_img == known['n_images'] else '<-- MISMATCH'}")
    ok &= n_img == known["n_images"]

    if meta is not None and patient_col and patient_col in meta.columns:
        n_pat = meta[patient_col].nunique()
        print(f"  patients: found {n_pat}, published {known['n_patients']}  "
              f"{'OK' if n_pat == known['n_patients'] else '<-- MISMATCH'}")
        ok &= n_pat == known["n_patients"]

    if meta is not None and label_col and label_col in meta.columns and "pathology" in known:
        found = meta[label_col].astype(str).str.strip().str.lower().value_counts().to_dict()
        print(f"  pathology: found {found}")
        print(f"             published {known['pathology']}")
        for k, v in known["pathology"].items():
            if found.get(k) != v:
                ok = False

    print()
    if ok:
        print("  Mirror matches the published release. Safe to use, but cite the paper")
        print("  above, not the Kaggle page. The license requires it.")
    else:
        print("  MISMATCH. A third-party mirror that does not match the published counts")
        print("  has been modified: re-split, deduplicated, subsetted or re-encoded.")
        print("  Do not validate on it. Download from the official source instead.")


def check_lr_pairs(images: List[Path], sample: int = 60) -> Optional[Dict]:

    import cv2
    import numpy as np

    pairs: Dict[str, Dict[str, Path]] = {}
    for p in images:
        if "-" not in p.stem:
            continue
        base, _, side = p.stem.rpartition("-")
        if side in ("l", "r"):
            pairs.setdefault(base, {})[side] = p

    both = {k: v for k, v in pairs.items() if len(v) == 2}
    if not both:
        return None

    print(f"\n{'='*72}\nLEFT/RIGHT PAIR CHECK\n{'='*72}")
    print(f"  ids with both -l and -r: {len(both)} / {len(pairs)}")

    def ph(img: "np.ndarray") -> "np.ndarray":
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        d = cv2.dct(img.astype(np.float32))[:8, :8].flatten()
        return d > np.median(d[1:])

    keys = sorted(both)[:sample]
    d_flip, d_plain, exact_flip = [], [], 0
    for k in keys:
        a = cv2.imread(str(both[k]["l"]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(both[k]["r"]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        bf = np.fliplr(b)
        if a.shape == bf.shape and np.array_equal(a, bf):
            exact_flip += 1
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
            bf = cv2.resize(bf, (a.shape[1], a.shape[0]))
        d_flip.append(int((ph(a) != ph(bf)).sum()))
        d_plain.append(int((ph(a) != ph(b)).sum()))

    if not d_flip:
        return None

    med_flip = float(np.median(d_flip))
    med_plain = float(np.median(d_plain))
    print(f"  checked {len(d_flip)} pairs")
    print(f"  exact pixel matches after flipping -r : {exact_flip}/{len(d_flip)}")
    print(f"  median pHash distance, -l vs flip(-r) : {med_flip:.1f}  (0 = identical)")
    print(f"  median pHash distance, -l vs -r       : {med_plain:.1f}")
    print(f"  (for reference, two unrelated ultrasound images sit around 25-32)")

    verdict: str
    if exact_flip > 0.5 * len(d_flip) or med_flip <= 6:
        verdict = "FLIPPED_DUPLICATES"
        print("\n  VERDICT: the -r images are horizontally flipped copies of the -l images.")
        print("  This mirror applied offline augmentation. Consequences:")
        print("    * your effective sample size is HALF the file count")
        print("    * every -l/-r pair must share a group, or you leak a mirror image")
        print("      of a training case into test")
        print("    * bootstrap CIs computed on the file count are too narrow")
        print("  Do not use this copy. Download the original from Zenodo:")
        print(f"    {KNOWN_COHORTS['BUS-BRA']['official_source']}")
    elif med_flip < med_plain - 6:
        verdict = "SUSPICIOUS"
        print("\n  VERDICT: not identical, but -l matches flip(-r) noticeably better than -r.")
        print("  Could be flipped copies that were re-encoded or re-cropped. Inspect a few")
        print("  pairs by eye before using this copy.")
    else:
        verdict = "DISTINCT_SCANS"
        print("\n  VERDICT: -l and -r are distinct images, consistent with left/right breast")
        print("  of the same patient. Group them by the patient column, not by filename.")

    return {"n_pairs": len(both), "verdict": verdict, "median_phash_vs_flip": med_flip,
            "median_phash_plain": med_plain, "exact_flip_matches": exact_flip}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cohort", default="BUS-BRA", choices=sorted(KNOWN_COHORTS) + ["other"])
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--patient-col", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    print(f"Inspecting {root.resolve()}")
    print("This script reports STRUCTURE ONLY. It loads no model and computes no metric,")
    print("so running it does not spend your one look at the external cohort.")

    walk_tree(root)
    csvs = inspect_csvs(root)
    groups = inspect_images(root)
    lr = check_lr_pairs(groups["images"])

    meta = None
    if csvs:
        # heuristic: the biggest CSV is usually the per-image metadata table
        meta_path = max(csvs, key=lambda c: c.stat().st_size)
        try:
            meta = pd.read_csv(meta_path)
            print(f"\n[assuming {meta_path.name} is the per-image metadata table]")
        except Exception:
            meta = None

    if meta is not None and args.id_col:
        try_match(meta, args.id_col, groups["images"])

    check_against_published(args.cohort, meta, groups["images"], args.label_col, args.patient_col)

    print(f"\n{'='*72}\nNEXT STEP\n{'='*72}")
    print("  1. Do NOT run external_eval.py until run_cv.py is finished and frozen.")
    print("  2. This cohort has no 'normal' class, so train with --task 2class.")
    print("  3. When the BUSI pipeline is locked, run external_eval.py ONCE and report")
    print("     whatever it returns, including if it is bad.")

    if args.out:
        summary = {
            "root": str(root.resolve()),
            "n_images": len(groups["images"]),
            "n_masks": len(groups["masks"]),
            "csv_files": [str(c.relative_to(root)) for c in csvs],
            "metadata_columns": list(meta.columns) if meta is not None else [],
            "lr_pair_check": lr,
        }
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
