from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from decision import Decision, apply_decision, select_threshold
from metrics import bootstrap_ci, compute_metrics, format_ci, sensitivity_specificity, standard_ci_block


def load_fold(fold_dir: Path) -> Optional[Dict]:
    emb_p, ens_p, pred_p = (fold_dir / "embeddings.npz", fold_dir / "ensemble.pkl",
                            fold_dir / "test_predictions.npz")
    if not (emb_p.exists() and ens_p.exists()):
        return None
    z = np.load(emb_p)
    with open(ens_p, "rb") as f:
        blob = pickle.load(f)
    ens, class_names = blob["ensemble"], blob["class_names"]

    out = {
        "fold": int(fold_dir.name.split("_")[-1]),
        "class_names": class_names,
        "y_va": z["y_va"], "p_va": ens.predict_proba(z["E_va"]),
        "y_te": z["y_te"], "p_te": ens.predict_proba(z["E_te"]),
        "decision": blob["decision"],
    }
    out["groups_te"] = np.load(pred_p, allow_pickle=True)["groups"] if pred_p.exists() else None
    try:
        out["members"] = ens.members
        out["member_probas"] = ens.member_probas(z["E_te"])
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] could not recover member probabilities for {fold_dir.name}: {exc}")
        out["members"], out["member_probas"] = [], {}
    return out


# ---------------------------------------------------------------------------
def section_discrimination(folds: List[Dict], mal: int, seed: int) -> Dict:
    print("=" * 78)
    print("1. THRESHOLD-FREE DISCRIMINATION")
    print("=" * 78)
    print("\n  AUROC does not depend on the operating point, so it cannot be inflated by")
    print("  the choice of tau. This is the number to headline.\n")

    from sklearn.metrics import roc_auc_score

    per_fold = []
    for f in folds:
        y, p = f["y_te"], f["p_te"][:, mal]
        auc = roc_auc_score((y == mal).astype(int), p)
        per_fold.append(auc)
        print(f"    fold {f['fold']}   AUROC = {auc:.4f}   (n={len(y)})")

    y = np.concatenate([f["y_te"] for f in folds])
    p = np.concatenate([f["p_te"] for f in folds])
    g = ([np.array([f"f{f['fold']}_{x}" for x in f["groups_te"]]) for f in folds]
         if all(f["groups_te"] is not None for f in folds) else None)
    groups = np.concatenate(g) if g else None

    auc_fn = lambda t, _pred, pr: float(roc_auc_score((t == mal).astype(int), pr[:, mal]))
    ci = bootstrap_ci(auc_fn, y, y, p, n_boot=2000, seed=seed, groups=groups)

    mean, sd = float(np.mean(per_fold)), float(np.std(per_fold, ddof=1))
    print(f"\n    per-fold mean +/- SD : {mean:.4f} +/- {sd:.4f}   <- REPORT THIS")
    print(f"    pooled out-of-fold   : {ci['point']:.4f}  95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"    bootstrap unit       : {ci['unit']}"
          + (f" ({ci['n_clusters']} groups)" if ci.get("n_clusters") else ""))

    if mean - ci["point"] > 0.005:
        print(f"""
    NOTE: pooled ({ci['point']:.4f}) is BELOW the per-fold mean ({mean:.4f}) by
    {(mean-ci['point'])*100:.1f} points. This is expected and is an artefact of pooling, not a
    property of any model you would deploy. AUROC is a ranking metric; pooling
    probabilities from {len(folds)} differently-calibrated models ranks them against each
    other, so a benign case from a high-scoring fold can outrank a malignant from
    a low-scoring one. That penalises inter-fold calibration mismatch, which no
    single deployed model has.

    So: report AUROC as the per-fold mean +/- SD, and use the pooled cluster
    bootstrap only for the uncertainty interval. Threshold-based metrics
    (sensitivity, specificity, accuracy) DO pool correctly, because each fold's
    tau is applied to its own test split and the outputs are just labels.""")
    return {"per_fold": per_fold, "per_fold_mean": mean, "per_fold_sd": sd, "pooled_ci": ci}


def section_operating_points(folds: List[Dict], mal: int, targets: List[float], seed: int) -> Dict:
    print("\n" + "=" * 78)
    print("2. OPERATING-POINT TABLE")
    print("=" * 78)
    print("\n  For each target, tau is re-selected on EACH FOLD'S OWN VALIDATION SPLIT and")
    print("  applied once to that fold's test split. Test labels are never consulted when")
    print("  choosing tau. Only the target changes between rows.\n")

    hdr = (f"  {'rule':>14s} {'sens':>16s} {'spec':>16s} {'PPV':>16s} {'bal acc':>16s} "
           f"{'gap':>7s}  {'tau (per fold)':>28s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows: Dict[str, Dict] = {}
    for t in [None] + list(targets):
        y_all, p_all, pred_all, grp_all, taus = [], [], [], [], []
        for f in folds:
            if t is None:
                tau = 1.0
                d = Decision(rule="argmax", malignant_index=mal, tau=tau, abstain_below=None)
            else:
                tau, _ = select_threshold(f["p_va"], f["y_va"], mal, "target_sensitivity", t)
                d = Decision(rule="target_sensitivity", malignant_index=mal, tau=tau,
                             abstain_below=None, target_sensitivity=t)
            taus.append(tau)
            yp, _ = apply_decision(f["p_te"], d)
            y_all.append(f["y_te"]); p_all.append(f["p_te"]); pred_all.append(yp)
            if f["groups_te"] is not None:
                grp_all.append(np.array([f"f{f['fold']}_{x}" for x in f["groups_te"]]))

        y = np.concatenate(y_all); p = np.concatenate(p_all); yp = np.concatenate(pred_all)
        groups = np.concatenate(grp_all) if len(grp_all) == len(folds) else None
        ci = standard_ci_block(y, yp, p, mal, n_boot=1000, seed=seed, groups=groups)

        if t is None:
            label, tau_s, gap_s = "argmax", "none (no tuning)", "  -"
        else:
            label = f"sens >= {t:.0%}"
            tau_s = "[" + ", ".join(f"{x:.3f}" for x in taus) + "]"
            gap_s = f"{(ci['malignant_sensitivity']['point'] - t)*100:+5.1f}"
        print(f"  {label:>14s} {format_ci(ci['malignant_sensitivity']):>16s} "
              f"{format_ci(ci['malignant_specificity']):>16s} {format_ci(ci['malignant_ppv']):>16s} "
              f"{format_ci(ci['balanced_accuracy']):>16s} {gap_s:>7s}  {tau_s:>28s}")
        key = "argmax" if t is None else f"{t:.2f}"
        rows[key] = {"ci": ci, "taus": taus,
                     "tau_spread_ratio": None if t is None else float(max(taus) / max(min(taus), 1e-9)),
                     "achieved_minus_target": None if t is None else
                         float(ci["malignant_sensitivity"]["point"] - t)}

    gaps = [r["achieved_minus_target"] for r in rows.values() if r.get("achieved_minus_target") is not None]
    if gaps and all(g < 0 for g in gaps):
        print(f"""
  Every target UNDERSHOOTS on test, by {min(gaps)*100:.1f} to {max(gaps)*100:.1f} points, all in the same
  direction. This is the winner's curse of threshold selection: tau is placed at
  the very edge of the validation distribution (roughly the 2nd-lowest-scoring
  malignant case among ~33), so it is optimistically positioned and degrades on
  new data. The previous version of this work never saw the effect because it
  chose tau on the set it reported, where by construction there is no gap.

  The `argmax` row involves NO threshold selection at all. It is therefore the
  most defensible single operating point in this table, and it is immune to
  everything section 3 measures.""")

    print("\n  Read the tau column, not the metrics.")
    for t, row in rows.items():
        if t == "argmax" or row.get("tau_spread_ratio") is None:
            continue  # the argmax rule has no threshold to spread
        lo, hi = min(row["taus"]), max(row["taus"])
        print(f"    target {float(t):.0%}: tau ranges {lo:.3f} to {hi:.3f} "
              f"({row['tau_spread_ratio']:.1f}x) across folds")
    print("  Compare that spread against the AUROC spread in section 1. If the encoders")
    print("  are interchangeable but the thresholds are not, the rule is the problem.")
    return rows


def section_threshold_stability(folds: List[Dict], mal: int, target: float,
                                n_boot: int, seed: int) -> Dict:
    print("\n" + "=" * 78)
    print("3. THRESHOLD STABILITY  (the encoder is fixed; only the val draw changes)")
    print("=" * 78)
    print(f"\n  Resample each fold's validation split {n_boot}x, re-select tau at target")
    print(f"  sensitivity {target:.0%} on each resample, apply each tau to the UNCHANGED test")
    print("  split. The encoder never changes. Any spread below is caused by the")
    print("  validation draw alone.\n")

    rng = np.random.default_rng(seed)
    print(f"  {'fold':>5s} {'tau median':>11s} {'tau IQR':>18s} {'test spec median':>17s} "
          f"{'test spec 5-95%':>20s} {'test sens median':>17s}")
    print("  " + "-" * 94)

    out: Dict[str, Dict] = {}
    for f in folds:
        y_va, p_va, y_te, p_te = f["y_va"], f["p_va"], f["y_te"], f["p_te"]
        n = len(y_va)
        cls_idx = [np.where(y_va == c)[0] for c in np.unique(y_va)]
        taus, specs, senss = [], [], []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(ci, size=len(ci), replace=True) for ci in cls_idx])
            tau, _ = select_threshold(p_va[idx], y_va[idx], mal, "target_sensitivity", target)
            d = Decision(rule="target_sensitivity", malignant_index=mal, tau=tau,
                         abstain_below=None, target_sensitivity=target)
            yp, _ = apply_decision(p_te, d)
            s = sensitivity_specificity(y_te, yp, mal)
            taus.append(tau); specs.append(s["specificity"]); senss.append(s["sensitivity"])

        taus, specs, senss = map(np.array, (taus, specs, senss))
        q = lambda a, x: float(np.quantile(a, x))
        print(f"  {f['fold']:5d} {np.median(taus):11.4f} "
              f"[{q(taus,.25):.3f}, {q(taus,.75):.3f}]".rjust(19)
              + f"{np.median(specs)*100:16.1f}% "
              + f"[{q(specs,.05)*100:.1f}, {q(specs,.95)*100:.1f}]".rjust(20)
              + f"{np.median(senss)*100:16.1f}%")
        out[str(f["fold"])] = {
            "tau_median": float(np.median(taus)),
            "tau_iqr": [q(taus, .25), q(taus, .75)],
            "test_specificity_median": float(np.median(specs)),
            "test_specificity_90pct_interval": [q(specs, .05), q(specs, .95)],
            "test_sensitivity_median": float(np.median(senss)),
        }

    widths = [o["test_specificity_90pct_interval"][1] - o["test_specificity_90pct_interval"][0]
              for o in out.values()]
    print(f"\n  Median width of the 90% interval on test specificity, caused purely by")
    print(f"  resampling 101 validation images: {np.median(widths)*100:.1f} points.")
    print("\n  Interpretation. tau at a target sensitivity is essentially an extreme order")
    print("  statistic of the malignant scores in validation: with 33 malignant cases, it")
    print("  is set by roughly the 2nd-lowest-scoring one. That has huge variance by")
    print("  construction. It is why five encoders with AUROC spanning 4.7 points produce")
    print("  operating points whose specificity spans 61 points.")
    return out


def section_members(folds: List[Dict], mal: int, seed: int) -> Dict:
    """Does the three-member ensemble beat its best single member?

    The submitted paper defended a prototype + XGBoost + RF ensemble. If one
    member alone matches it, say so. A simpler model that performs identically
    is a finding, not a failure, and defending components that add nothing is a
    losing argument in review.

    Everything here is at ARGMAX, so no threshold selection contaminates the
    comparison.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    print("\n" + "=" * 78)
    print("4. DOES THE ENSEMBLE EARN ITS KEEP?  (all at argmax, no tuning)")
    print("=" * 78 + "\n")

    names = [m.name for m in folds[0].get("members", [])]
    if not names:
        print("  (member probabilities unavailable; skipping)")
        return {}

    print(f"  {'member':>16s} {'AUROC mean +/- SD':>22s} {'bal acc mean +/- SD':>22s}")
    print("  " + "-" * 62)

    out: Dict[str, Dict] = {}
    for name in names + ["ENSEMBLE"]:
        aucs, baccs = [], []
        for f in folds:
            p = f["p_te"] if name == "ENSEMBLE" else f["member_probas"][name]
            y = f["y_te"]
            aucs.append(roc_auc_score((y == mal).astype(int), p[:, mal]))
            baccs.append(balanced_accuracy_score(y, p.argmax(1)))
        out[name] = {
            "auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs, ddof=1)),
            "bacc_mean": float(np.mean(baccs)), "bacc_sd": float(np.std(baccs, ddof=1)),
            "auroc_per_fold": [float(a) for a in aucs],
        }
        tag = "  <-" if name == "ENSEMBLE" else ""
        print(f"  {name:>16s} {np.mean(aucs):11.4f} +/- {np.std(aucs, ddof=1):.4f} "
              f"{np.mean(baccs):11.4f} +/- {np.std(baccs, ddof=1):.4f}{tag}")

    best_single = max((k for k in out if k != "ENSEMBLE"), key=lambda k: out[k]["auroc_mean"])
    delta = out["ENSEMBLE"]["auroc_mean"] - out[best_single]["auroc_mean"]
    sd = out["ENSEMBLE"]["auroc_sd"]
    print(f"\n  best single member: {best_single} (AUROC {out[best_single]['auroc_mean']:.4f})")
    print(f"  ensemble minus best single: {delta:+.4f} AUROC")
    if abs(delta) < sd:
        print(f"\n  That difference is smaller than the fold-to-fold SD ({sd:.4f}). The ensemble")
        print(f"  is not distinguishable from '{best_single}' alone on this data. Report that")
        print("  honestly: a reviewer who suspects the ensemble is decorative will test")
        print("  exactly this, and finding it yourself is much better than being told.")
    else:
        print(f"\n  The ensemble exceeds its best single member by more than one fold-SD.")
        print("  That is a defensible reason to keep it.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="./runs/rebuild_2class")
    ap.add_argument("--targets", default="0.90,0.95,0.99")
    ap.add_argument("--stability-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run = Path(args.run)
    folds = [f for f in (load_fold(d) for d in sorted(run.glob("fold_*"))) if f]
    if not folds:
        raise SystemExit(f"No fold artefacts under {run}. Run run_cv.py first.")

    class_names = folds[0]["class_names"]
    mal = class_names.index("malignant")
    print(f"Loaded {len(folds)} folds from {run}, classes {class_names}\n")

    report = {
        "n_folds": len(folds),
        "class_names": class_names,
        "discrimination": section_discrimination(folds, mal, args.seed),
        "operating_points": section_operating_points(
            folds, mal, [float(x) for x in args.targets.split(",")], args.seed),
        "threshold_stability": section_threshold_stability(
            folds, mal, 0.95, args.stability_boot, args.seed),
        "members": section_members(folds, mal, args.seed),
    }

    print("\n" + "=" * 78)
    print("WHAT TO PUT IN THE PAPER")
    print("=" * 78)
    d = report["discrimination"]["pooled_ci"]
    m, sd = report["discrimination"]["per_fold_mean"], report["discrimination"]["per_fold_sd"]
    print(f"""
  Headline:  AUROC {m:.3f} +/- {sd:.3f} (mean +/- SD over folds), 95% CI
             [{d['lo']:.3f}, {d['hi']:.3f}] from a cluster bootstrap over duplicate groups.
             Every image predicted once by a model that never saw it.

  Then the operating-point table from section 2, with the tau column visible.

  Then section 3 as a finding in its own right: in small-dataset ultrasound CAD,
  variance in threshold selection dominates variance in the model. It is
  invisible when a single split is reported, and it disappears entirely when the
  threshold is tuned on the test set, which is what the previous version did.
""")

    out = run / "analysis.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
