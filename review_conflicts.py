from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd


def conflicted_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Groups containing more than one distinct label."""
    n_lab = df.groupby("group")["label"].nunique()
    bad = n_lab[n_lab > 1].index
    return df[df["group"].isin(bad)].sort_values(["group", "class_name", "filename"])


def montage(paths: List[str], labels: List[str], out_path: Path, tile: int = 220) -> None:
    tiles = []
    for p, lab in zip(paths, labels):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (tile, tile))
        banner = np.zeros((28, tile, 3), np.uint8)
        colour = {"benign": (90, 200, 90), "malignant": (60, 60, 230),
                  "normal": (200, 160, 60)}.get(lab, (200, 200, 200))
        banner[:] = colour
        cv2.putText(banner, lab, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        name = Path(p).stem[:26]
        strip = np.zeros((20, tile, 3), np.uint8)
        cv2.putText(strip, name, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([banner, img, strip]))
    if tiles:
        cv2.imwrite(str(out_path), np.hstack(tiles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="./runs/audit")
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    audit = Path(args.audit)
    df = pd.read_csv(audit / "index_with_groups.csv")
    pairs = pd.read_csv(audit / "duplicate_pairs.csv")

    bad = conflicted_groups(df)
    groups = sorted(bad["group"].unique())

    print("=" * 74)
    print("DUPLICATE GROUPS WITH CONTRADICTORY LABELS")
    print("=" * 74)
    print(f"\n  conflicted groups : {len(groups)}")
    print(f"  images involved   : {len(bad)}  ({len(bad)/len(df)*100:.1f}% of the dataset)")
    if len(pairs):
        cf = pairs[pairs.label_conflict]
        print(f"  conflicting pairs : {len(cf)}")
        if len(cf):
            print(f"  pHash distances   : {sorted(cf.phash_hamming.tolist())}")
            print(f"  exact pixel dupes : {int(cf.exact_pixel_duplicate.sum())}")
            print("\n  A tightened threshold does NOT remove these. The distances below ~4 out")
            print("  of 64 bits mean the images are effectively identical.")

    print("\n  breakdown by class pair:")
    combos: Dict[str, int] = {}
    for g in groups:
        sub = bad[bad.group == g]
        key = " + ".join(sorted(sub.class_name.unique()))
        combos[key] = combos.get(key, 0) + 1
    for k, v in sorted(combos.items(), key=lambda x: -x[1]):
        print(f"    {k:26s} {v} group(s)")

    print("\n  per group:")
    for g in groups:
        sub = bad[bad.group == g]
        files = ", ".join(f"{r.filename} [{r.class_name}]" for r in sub.itertuples())
        print(f"    group {g:4d}  n={len(sub)}  {files}")

    out_dir = audit / "conflicts"
    if not args.no_montage:
        out_dir.mkdir(parents=True, exist_ok=True)
        for g in groups:
            sub = bad[bad.group == g]
            montage(sub["path"].tolist(), sub["class_name"].tolist(),
                    out_dir / f"group_{g:04d}.png")
        print(f"\n  Wrote {len(groups)} side-by-side montages to {out_dir}/")
        print("  LOOK AT THEM. If a pair is clearly two different lesions, the hash is")
        print("  wrong and you should say so. If it is the same scan under two labels,")
        print("  that is a labelling error in BUSI and belongs in the paper.")

    excl = audit / "excluded_conflicted.csv"
    bad[["path", "filename", "class_name", "label", "group"]].to_csv(excl, index=False)
    (audit / "conflicts_summary.json").write_text(json.dumps({
        "n_conflicted_groups": len(groups),
        "n_images_excluded": int(len(bad)),
        "fraction_of_dataset": float(len(bad) / len(df)),
        "class_pair_breakdown": combos,
        "groups": [int(g) for g in groups],
    }, indent=2))

    print(f"\n  Wrote exclusion list to {excl}")
    print(f"\n  To use it, set in config.py:")
    print(f"      data.drop_conflicted_groups = True   (this is the default)")
    print(f"  run_cv.py will then drop these {len(bad)} images and record the count in")
    print(f"  the run config, so the exclusion is visible to a reviewer.")


if __name__ == "__main__":
    main()
