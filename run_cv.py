from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from config import Config
from decision import apply_decision, fit_decision
from heads import ConservativeEnsemble, FeatureSpec
from metrics import aggregate_folds, compute_metrics, format_ci, standard_ci_block
from model import extract_embeddings
from splits import (apply_duplicate_policy, describe_splits, drop_conflicted_groups,
                    filter_task, make_nested_splits)
from train_encoder import train_encoder


def run_fold(df: pd.DataFrame, spec, cfg: Config, device: str, out_dir: Path) -> Dict:
    fold_dir = out_dir / f"fold_{spec.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    df_train = df.iloc[spec.train_idx].reset_index(drop=True)
    df_val = df.iloc[spec.val_idx].reset_index(drop=True)
    df_test = df.iloc[spec.test_idx].reset_index(drop=True)

    print(f"\n{'='*72}\nFOLD {spec.fold}  {spec.sizes()}\n{'='*72}")

    # 1. encoder ------------------------------------------------------------
    model, history = train_encoder(df_train, df_val, cfg, device, fold_dir, seed=cfg.seed + spec.fold)

    # 2. embeddings ---------------------------------------------------------
    from data import make_eval_loader

    E_tr, y_tr = extract_embeddings(model, make_eval_loader(df_train, cfg), device)
    E_va, y_va = extract_embeddings(model, make_eval_loader(df_val, cfg), device)
    E_te, y_te = extract_embeddings(model, make_eval_loader(df_test, cfg), device)
    np.savez_compressed(fold_dir / "embeddings.npz",
                        E_tr=E_tr, y_tr=y_tr, E_va=E_va, y_va=y_va, E_te=E_te, y_te=y_te)

    # 3. heads, fit on TRAIN only -------------------------------------------
    ens = ConservativeEnsemble(len(cfg.class_names), cfg.head).fit(E_tr, y_tr)
    spec_f = FeatureSpec(E_tr.shape[1], len(cfg.class_names))
    print(f"  feature width: {spec_f.total}  {spec_f.breakdown()}")

    p_va = ens.predict_proba(E_va)
    ens.set_weights_from_val(E_va, y_va)  # no-op unless ensemble='weighted_vote'
    p_te = ens.predict_proba(E_te)

    # 4. operating point, chosen on VAL -------------------------------------
    if cfg.decision.select_on == "test":
        # ABLATION ONLY: reproduces the original protocol of choosing the
        # threshold on the set that is then reported. See CHANGES.md #2.
        decision = fit_decision(p_te, y_te, cfg.malignant_index, cfg.decision,
                                chosen_on="OUTER_TEST_ABLATION_NOT_A_RESULT")
    else:
        decision = fit_decision(p_va, y_va, cfg.malignant_index, cfg.decision)
    # NB: Decision's fields are named achieved_val_* but hold whatever split the
    # threshold was actually fitted on. Under the --threshold-on test ablation
    # that is TEST, so label the line by decision.chosen_on rather than assuming.
    src = "val" if decision.chosen_on == "inner_val" else "TEST (ablation)"
    sens_s = ("n/a" if decision.achieved_val_sensitivity is None
              else f"{decision.achieved_val_sensitivity:.4f}")
    spec_s = ("n/a" if decision.achieved_val_specificity is None
              else f"{decision.achieved_val_specificity:.4f}")
    print(f"  decision: tau={decision.tau:.4f}  fitted on {src}: "
          f"sens={sens_s}, spec={spec_s}, abstain_below={decision.abstain_below}")

    # 5. the one and only test evaluation -----------------------------------
    yhat_te, answered_te = apply_decision(p_te, decision)
    m_test = compute_metrics(y_te, yhat_te, p_te, cfg.class_names, cfg.malignant_index, answered_te)
    groups_te = df_test["group"].to_numpy()
    m_test["ci"] = standard_ci_block(
        y_te[answered_te], yhat_te[answered_te], p_te[answered_te],
        cfg.malignant_index, cfg.eval.n_bootstrap, cfg.eval.ci_alpha, cfg.eval.seed + spec.fold,
        groups=groups_te[answered_te],
    )

    yhat_va, answered_va = apply_decision(p_va, decision)
    m_val = compute_metrics(y_va, yhat_va, p_va, cfg.class_names, cfg.malignant_index, answered_va)

    per_member = {}
    for name, pm in ens.member_probas(E_te).items():
        per_member[name] = compute_metrics(
            y_te, pm.argmax(1), pm, cfg.class_names, cfg.malignant_index
        )

    result = {
        "fold": spec.fold,
        "sizes": spec.sizes(),
        "encoder": {"best_epoch": history["best_epoch"],
                    "best_val_balanced_accuracy": history["best_val_balanced_accuracy"],
                    "saturation_epoch": history.get("saturation_epoch"),
                    "final_frac_active_triplets": history.get("final_frac_active_triplets"),
                    "min_frac_active_triplets": history.get("min_frac_active_triplets")},
        "decision": decision.to_dict(),
        "feature_width": spec_f.total,
        "val": m_val,
        "test": m_test,
        "test_per_member": per_member,
    }

    with open(fold_dir / "ensemble.pkl", "wb") as f:
        pickle.dump({"ensemble": ens, "decision": decision, "class_names": cfg.class_names}, f)
    torch.save({"model": model.state_dict(), "cfg": cfg.encoder.__dict__}, fold_dir / "encoder_final.pt")
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    np.savez(fold_dir / "test_predictions.npz",
             y_true=y_te, y_pred=yhat_te, proba=p_te, answered=answered_te,
             paths=df_test["path"].to_numpy(), groups=groups_te)

    print(f"  TEST  bacc={m_test['balanced_accuracy']:.4f}  "
          f"malignant sens={m_test['malignant']['sensitivity']:.4f}  "
          f"spec={m_test['malignant']['specificity']:.4f}  coverage={m_test['coverage']:.3f}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="./runs/audit/index_with_groups.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default=None, choices=["3class", "2class"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--folds", default=None, help="comma separated subset, e.g. 1,2")
    ap.add_argument("--rule", default=None, choices=["argmax", "target_sensitivity", "youden"])
    # --- ablation switches. Each one makes the run NOT a result. ---
    ap.add_argument("--mask-oracle", action="store_true",
                    help="ABLATION: shorthand for --mask-mode soft_dim")
    ap.add_argument("--mask-mode", default=None, choices=["none", "soft_dim", "roi_crop"],
                    help="ABLATION: 'soft_dim' = the original notebook's variant; "
                         "'roi_crop' = the field's published practice (CHANGES.md #1)")
    ap.add_argument("--roi-normal-fallback", default=None,
                    choices=["full_image", "center_crop", "matched_random"],
                    help="what to do for 3-class 'normal' cases, which have no bounding box")
    ap.add_argument("--splitting", default=None, choices=["group", "random"],
                    help="ABLATION: 'random' splits per image, leaking near-duplicates (#5)")
    ap.add_argument("--threshold-on", default=None, choices=["val", "test"],
                    help="ABLATION: 'test' tunes the threshold on the reported set (#2)")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.out:
        cfg.out_dir = args.out
    if args.task:
        cfg.data.task = args.task
    if args.rule:
        cfg.decision.rule = args.rule
    if args.mask_oracle:
        cfg.data.use_gt_mask_oracle = True
        cfg.data.mask_mode = "soft_dim"
    if args.mask_mode:
        cfg.data.mask_mode = args.mask_mode
        cfg.data.use_gt_mask_oracle = args.mask_mode != "none"
    if args.roi_normal_fallback:
        cfg.data.roi_normal_fallback = args.roi_normal_fallback
    if args.splitting:
        cfg.splitting = args.splitting
    if args.threshold_on:
        cfg.decision.select_on = args.threshold_on

    ablations = []
    if cfg.data.mask_mode == "soft_dim":
        ablations.append("ground-truth mask, soft-dim variant: out = img*m + 0.3*img*(1-m) "
                         "(the original notebook's choice, NOT the field's) (CHANGES.md #1)")
    elif cfg.data.mask_mode == "roi_crop":
        ablations.append("ground-truth mask, ROI crop: the field's published practice "
                         f"(pad={cfg.data.roi_pad_ratio}, normals -> {cfg.data.roi_normal_fallback}) "
                         "(CHANGES.md #1)")
    if cfg.splitting == "random":
        ablations.append("per-image random splitting, duplicates leak (CHANGES.md #5)")
    if cfg.decision.select_on == "test":
        ablations.append("threshold tuned on the reported test set (CHANGES.md #2)")
    if ablations:
        print("\n" + "#" * 74)
        print("#  ABLATION RUN. THE OUTPUT OF THIS RUN IS NOT A RESULT.")
        print("#  Active defects, deliberately reintroduced:")
        for a in ablations:
            print(f"#    - {a}")
        print("#  Report these rows only inside the ablation table, labelled as such.")
        print("#" * 74 + "\n")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")

    df = pd.read_csv(args.index)
    if "group" not in df.columns:
        raise SystemExit("Run audit_dataset.py first: the index needs a 'group' column.")
    n_start = len(df)
    dropped = None
    if cfg.data.drop_conflicted_groups:
        df, dropped = drop_conflicted_groups(df)
        if dropped is not None and len(dropped):
            dropped.to_csv(out_dir / "excluded_conflicted.csv", index=False)
    df = filter_task(df, cfg.data.task)
    df = apply_duplicate_policy(df, cfg.data.duplicate_policy, cfg.seed)
    print(f"[run] {n_start} indexed -> {len(df)} used "
          f"({0 if dropped is None else len(dropped)} label-conflicted excluded), "
          f"{df['group'].nunique()} duplicate-groups, task={cfg.data.task}")

    specs = make_nested_splits(df, cfg.n_outer_folds, cfg.inner_val_fraction, cfg.seed,
                               grouping=cfg.splitting)
    table = describe_splits(df, specs, cfg.class_names)
    table.to_csv(out_dir / "splits.csv", index=False)
    print(table.to_string(index=False))

    wanted = {int(x) for x in args.folds.split(",")} if args.folds else None
    results: List[Dict] = []
    for spec in specs:
        if wanted and spec.fold not in wanted:
            continue
        results.append(run_fold(df, spec, cfg, args.device, out_dir))
        (out_dir / "all_folds.json").write_text(json.dumps(results, indent=2, default=float))

    # ---- pooled reporting -------------------------------------------------
    keys = ["test.balanced_accuracy", "test.accuracy", "test.macro_f1",
            "test.malignant.sensitivity", "test.malignant.specificity",
            "test.malignant.ppv", "test.coverage", "test.ece"]
    if len(cfg.class_names) == 2:
        keys.append("test.auroc")
    else:
        keys.append("test.auroc_ovr_macro")
    summary = aggregate_folds(results, keys)

    # pooled out-of-fold predictions: every image is predicted exactly once, by
    # a model that never saw it. This is the estimate to headline.
    pooled_y, pooled_p, pooled_pred, pooled_ans, pooled_grp = [], [], [], [], []
    for spec in specs:
        f = out_dir / f"fold_{spec.fold}" / "test_predictions.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        pooled_y.append(z["y_true"]); pooled_p.append(z["proba"])
        pooled_pred.append(z["y_pred"]); pooled_ans.append(z["answered"])
        pooled_grp.append(z["groups"])

    pooled = {}
    if pooled_y:
        y = np.concatenate(pooled_y); p = np.concatenate(pooled_p)
        yh = np.concatenate(pooled_pred); an = np.concatenate(pooled_ans)
        gr = np.concatenate(pooled_grp)
        pooled = compute_metrics(y, yh, p, cfg.class_names, cfg.malignant_index, an)
        pooled["ci"] = standard_ci_block(y[an], yh[an], p[an], cfg.malignant_index,
                                         cfg.eval.n_bootstrap, cfg.eval.ci_alpha, cfg.eval.seed,
                                         groups=gr[an])

    (out_dir / "summary.json").write_text(
        json.dumps({
            "is_ablation": bool(ablations),
            "active_defects": ablations,
            "protocol": {
                "task": cfg.data.task,
                "splitting": cfg.splitting,
                "threshold_selected_on": cfg.decision.select_on,
                "gt_mask_in_input": cfg.data.use_gt_mask_oracle,
                "mask_mode": cfg.data.mask_mode,
                "roi_normal_fallback": cfg.data.roi_normal_fallback,
                "decision_rule": cfg.decision.rule,
                "drop_conflicted_groups": cfg.data.drop_conflicted_groups,
            },
            "per_fold_mean_sd": summary,
            "pooled_out_of_fold": pooled,
        }, indent=2, default=float)
    )

    print("\n" + "=" * 72)
    print("POOLED OUT-OF-FOLD RESULTS  (every image predicted by a model that never saw it)")
    print("=" * 72)
    if pooled:
        for k in ("accuracy", "balanced_accuracy", "malignant_sensitivity",
                  "malignant_specificity", "malignant_ppv"):
            print(f"  {k:24s} {format_ci(pooled['ci'][k])}")
        print(f"  {'coverage':24s} {pooled['coverage']*100:.1f}%")
        print(f"  {'ECE':24s} {pooled['ece']:.4f}")
    print("\nPer-fold mean +/- SD:")
    for k, v in summary.items():
        print(f"  {k:34s} {v['mean']:.4f} +/- {v['sd']:.4f}  (range {v['min']:.4f}-{v['max']:.4f})")
    print(f"\nWrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
