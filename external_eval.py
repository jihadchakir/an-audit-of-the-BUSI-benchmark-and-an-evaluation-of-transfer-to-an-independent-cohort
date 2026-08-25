from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import Config
from decision import Decision, apply_decision, fit_decision
from metrics import compute_metrics, format_ci, standard_ci_block, subgroup_report
from model import ResNetFPNEncoder, extract_embeddings


def build_external_index(
    images_dir: str,
    labels_csv: str,
    id_col: str,
    label_col: str,
    benign_values: List[str],
    malignant_values: List[str],
    patient_col: Optional[str] = None,
    ext: str = ".png",
    keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Map an arbitrary external cohort onto our schema."""
    meta = pd.read_csv(labels_csv)
    meta[label_col] = meta[label_col].astype(str).str.strip().str.lower()
    benign = {v.lower() for v in benign_values}
    malignant = {v.lower() for v in malignant_values}

    rows = []
    images = {p.stem: p for p in Path(images_dir).rglob(f"*{ext}") if "_mask" not in p.stem}
    missing = 0
    for _, r in meta.iterrows():
        key = str(r[id_col]).strip()
        path = images.get(key)
        if path is None:  # try a few common variants
            for cand in (key.replace(".png", ""), f"bus_{key}", key.zfill(4)):
                if cand in images:
                    path = images[cand]
                    break
        if path is None:
            missing += 1
            continue
        v = r[label_col]
        if v in benign:
            label, name = 0, "benign"
        elif v in malignant:
            label, name = 1, "malignant"
        else:
            continue
        row = {
            "path": str(path),
            "filename": path.name,
            "label": label,
            "class_name": name,
            "mask_paths": "",
            "group": str(r[patient_col]) if patient_col and patient_col in meta.columns else key,
        }
        for c in (keep_cols or []):
            if c in meta.columns:
                row[c] = r[c]
        rows.append(row)
    if missing:
        print(f"[external] {missing} rows in {labels_csv} had no matching image and were skipped")
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    if df.empty:
        raise SystemExit("No external images matched. Check --id-col and the filename convention.")
    print(f"[external] {len(df)} images, {df['class_name'].value_counts().to_dict()}, "
          f"{df['group'].nunique()} patients/groups")
    return df


def load_fold(fold_dir: str | Path, device: str):
    fold_dir = Path(fold_dir)
    cfg_path = fold_dir.parent / "config.json"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config()

    with open(fold_dir / "ensemble.pkl", "rb") as f:
        blob = pickle.load(f)
    ens, decision, class_names = blob["ensemble"], blob["decision"], blob["class_names"]

    ck = torch.load(fold_dir / "encoder_final.pt", map_location=device, weights_only=False)
    enc_cfg = ck["cfg"]
    model = ResNetFPNEncoder(
        embedding_dim=enc_cfg["embedding_dim"],
        fpn_channels=enc_cfg["fpn_channels"],
        pretrained=False,
        dropout=enc_cfg["dropout"],
        freeze_bn=enc_cfg.get("freeze_bn", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ens, decision, class_names, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--label-col", default="Pathology")
    ap.add_argument("--patient-col", default=None)
    ap.add_argument("--benign-values", default="benign,b,0")
    ap.add_argument("--malignant-values", default="malignant,m,1")
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--out", default="./runs/external")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-recalibrated", action="store_true")
    ap.add_argument("--cohort-name", default="external")
    ap.add_argument("--stratify-cols", default=None,
                    help="comma separated metadata columns to slice metrics by, e.g. Device,BIRADS")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, ens, decision, class_names, cfg = load_fold(args.fold_dir, args.device)
    if len(class_names) != 2:
        print("[external] WARNING: the loaded model is 3-class but external cohorts have no "
              "'normal' category. Malignant-vs-rest metrics remain interpretable; overall "
              "accuracy does not. Prefer training with --task 2class for this comparison.")

    strat_cols = [c.strip() for c in args.stratify_cols.split(",")] if args.stratify_cols else []
    df = build_external_index(
        args.images_dir, args.labels_csv, args.id_col, args.label_col,
        args.benign_values.split(","), args.malignant_values.split(","),
        args.patient_col, args.ext, keep_cols=strat_cols,
    )
    groups = df["group"].to_numpy()
    n_clusters = len(np.unique(groups))
    if n_clusters < len(df):
        print(f"[external] {len(df)} images from {n_clusters} patients/groups. Confidence "
              f"intervals use a CLUSTER bootstrap over patients, not images.")

    from data import make_eval_loader

    cfg.data.use_gt_mask_oracle = False  # never, on external data
    E, y = extract_embeddings(model, make_eval_loader(df, cfg), args.device)
    proba = ens.predict_proba(E)

    mal_idx = class_names.index("malignant")
    y_pred, answered = apply_decision(proba, decision)

    frozen = compute_metrics(y, y_pred, proba, class_names, mal_idx, answered)
    frozen["ci"] = standard_ci_block(y[answered], y_pred[answered], proba[answered],
                                     mal_idx, cfg.eval.n_bootstrap, cfg.eval.ci_alpha,
                                     cfg.eval.seed, groups=groups[answered])
    frozen["decision"] = decision.to_dict()
    frozen["n_clusters"] = int(n_clusters)
    for c in strat_cols:
        if c in df.columns:
            frozen.setdefault("subgroups", {})[c] = subgroup_report(
                y, y_pred, proba, df[c].to_numpy(), class_names, mal_idx)

    report = {"cohort": args.cohort_name, "n": int(len(df)),
              "source_fold": str(args.fold_dir), "frozen_transfer": frozen}

    if args.report_recalibrated:
        # honest labelling: this uses the external labels to pick tau, so it is
        # NOT a transfer result. It is the ceiling after site-specific calibration.
        d2 = fit_decision(proba, y, mal_idx, cfg.decision)
        yp2, an2 = apply_decision(proba, d2)
        rc = compute_metrics(y, yp2, proba, class_names, mal_idx, an2)
        rc["ci"] = standard_ci_block(y[an2], yp2[an2], proba[an2], mal_idx,
                                     cfg.eval.n_bootstrap, cfg.eval.ci_alpha, cfg.eval.seed,
                                     groups=groups[an2])
        rc["decision"] = d2.to_dict()
        rc["caveat"] = ("Threshold re-selected on this cohort's labels. Upper bound under "
                        "site-specific calibration, NOT an out-of-the-box transfer result.")
        report["recalibrated_upper_bound"] = rc

    (out / f"{args.cohort_name}_report.json").write_text(json.dumps(report, indent=2, default=float))
    np.savez(out / f"{args.cohort_name}_predictions.npz",
             y_true=y, y_pred=y_pred, proba=proba, answered=answered, paths=df["path"].to_numpy())

    print("\n" + "=" * 72)
    print(f"EXTERNAL VALIDATION: {args.cohort_name}  (frozen encoder, frozen heads, frozen tau)")
    print("=" * 72)
    for k in ("accuracy", "balanced_accuracy", "malignant_sensitivity",
              "malignant_specificity", "malignant_ppv"):
        print(f"  {k:24s} {format_ci(frozen['ci'][k])}")
    print(f"  {'coverage':24s} {frozen['coverage']*100:.1f}%")
    if "malignant_auroc" in frozen:
        print(f"  {'malignant AUROC':24s} {frozen['malignant_auroc']:.3f}")
    for col, block in (frozen.get("subgroups") or {}).items():
        print(f"\n  by {col}:")
        for level, m in block.items():
            if "skipped" in m:
                print(f"    {level:32s} n={m['n']:4d}  (skipped: too few)")
            else:
                print(f"    {level:32s} n={m['n']:4d}  bacc={m['balanced_accuracy']:.3f}  "
                      f"sens={m['malignant']['sensitivity']:.3f}  spec={m['malignant']['specificity']:.3f}")
    print(f"\nWrote {out}/{args.cohort_name}_report.json")


if __name__ == "__main__":
    main()
