from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import HeadConfig, DecisionConfig
from decision import apply_decision, fit_decision, select_threshold
from heads import ConservativeEnsemble, FeatureSpec, class_prototypes, loo_prototypes, make_features
from metrics import bootstrap_ci, compute_metrics, sensitivity_specificity
from splits import FoldSpec, assert_no_leakage, make_nested_splits


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def toy_df(n_groups: int = 60, copies: int = 2, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset with duplicate groups: `copies` near-identical images
    per group, mimicking BUSI's duplicated scans."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = g % 3
        for c in range(copies):
            rows.append({
                "path": f"/fake/{label}/img_{g}_{c}.png",
                "filename": f"img_{g}_{c}.png",
                "label": label,
                "class_name": ["benign", "malignant", "normal"][label],
                "group": g,
            })
    return pd.DataFrame(rows)


def toy_embeddings(df: pd.DataFrame, dim: int = 16, group_noise: float = 0.05, seed: int = 0):
    """Embeddings that carry a strong per-group signature on top of a weak
    per-class signal. This is exactly the structure that makes duplicate leakage
    inflate accuracy: memorising the group is enough to get the label right."""
    rng = np.random.default_rng(seed)
    class_centres = rng.normal(size=(3, dim)) * 0.35
    group_centres = {g: rng.normal(size=dim) for g in df["group"].unique()}
    E = np.stack([
        class_centres[r.label] + group_centres[r.group] + rng.normal(0, group_noise, dim)
        for r in df.itertuples()
    ])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E, df["label"].to_numpy()


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def test_nested_splits_are_group_disjoint():
    df = toy_df()
    specs = make_nested_splits(df, n_outer=5, inner_val_fraction=0.2, seed=42)
    assert len(specs) == 5
    for s in specs:
        assert_no_leakage(df, s)  # raises if any group straddles
        assert len(s.test_idx) > 0 and len(s.val_idx) > 0


def test_every_image_is_tested_exactly_once():
    """The old script trained 5 folds and evaluated 1. All 5 are used here."""
    df = toy_df()
    specs = make_nested_splits(df, n_outer=5, seed=42)
    test_counts = np.zeros(len(df), dtype=int)
    for s in specs:
        test_counts[s.test_idx] += 1
    assert (test_counts == 1).all()


def test_leakage_detector_catches_a_straddling_group():
    df = toy_df()
    g0 = np.where(df["group"] == 0)[0]
    rest = np.setdiff1d(np.arange(len(df)), g0)
    # put copy 0 in train and copy 1 in test: the classic duplicate leak
    bad = FoldSpec(fold=1,
                   train_idx=np.concatenate([[g0[0]], rest[:40]]),
                   val_idx=rest[40:70],
                   test_idx=np.concatenate([[g0[1]], rest[70:]]))
    with pytest.raises(AssertionError, match="straddle"):
        assert_no_leakage(df, bad)


def test_random_split_inflates_accuracy_versus_group_split():
    """Quantifies the bug. Random per-image splitting lets a 1-NN classifier
    look up the duplicate of the test image in the training set."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold

    df = toy_df(n_groups=90, copies=2, seed=3)
    E, y = toy_embeddings(df, seed=3)

    # naive: what the original notebooks did
    naive = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(E, y):
        knn = KNeighborsClassifier(1).fit(E[tr], y[tr])
        naive.append((knn.predict(E[te]) == y[te]).mean())

    # honest: duplicate groups kept together
    honest = []
    for s in make_nested_splits(df, n_outer=5, seed=0):
        idx = np.concatenate([s.train_idx, s.val_idx])
        knn = KNeighborsClassifier(1).fit(E[idx], y[idx])
        honest.append((knn.predict(E[s.test_idx]) == y[s.test_idx]).mean())

    print(f"\n  random-split accuracy: {np.mean(naive):.3f}   "
          f"group-split accuracy: {np.mean(honest):.3f}   "
          f"inflation: {(np.mean(naive)-np.mean(honest))*100:.1f} points")
    assert np.mean(naive) > np.mean(honest) + 0.15


# --------------------------------------------------------------------------
# features / prototypes
# --------------------------------------------------------------------------
def test_feature_width_is_275_not_278():
    """The old notebooks' docstring claimed 278 features; the code built 275.
    Whatever the paper says, it must match this number. Both the two-class (270)
    and three-class (275) widths reported in the manuscript are pinned here so
    the paper and the pipeline cannot silently drift apart."""
    spec = FeatureSpec(embedding_dim=256, n_classes=3)
    assert spec.total == 275
    assert spec.breakdown()["TOTAL"] == 275
    E = np.random.default_rng(0).normal(size=(10, 256))
    protos = np.random.default_rng(1).normal(size=(3, 256))
    assert make_features(E, protos).shape == (10, 275)

    # two-class width, stated as 270 in the manuscript
    spec2 = FeatureSpec(embedding_dim=256, n_classes=2)
    assert spec2.total == 270
    assert spec2.breakdown()["TOTAL"] == 270
    protos2 = np.random.default_rng(2).normal(size=(2, 256))
    assert make_features(E, protos2).shape == (10, 270)


def test_loo_prototypes_match_brute_force():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(30, 8))
    y = rng.integers(0, 3, 30)
    while len(np.unique(y)) < 3 or np.bincount(y).min() < 2:
        y = rng.integers(0, 3, 30)

    P = loo_prototypes(E, y, 3, renormalize=False)
    for i in range(len(E)):
        for c in range(3):
            m = y == c
            if c == y[i]:
                m = m.copy()
                m[i] = False
            expected = E[m].mean(0)
            np.testing.assert_allclose(P[i, c], expected, rtol=1e-9)


def test_loo_prototype_distances_exceed_naive_ones():
    """The point of LOO: without it, a training point is artificially close to
    its own class prototype, so the meta-classifier is fit on distances it will
    never see at test time."""
    rng = np.random.default_rng(0)
    E = rng.normal(size=(60, 8))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    y = np.repeat([0, 1, 2], 20)

    shared = class_prototypes(E, y, 3, renormalize=False)
    loo = loo_prototypes(E, y, 3, renormalize=False)
    d_naive = np.linalg.norm(E - shared[y], axis=1)
    d_loo = np.linalg.norm(E - loo[np.arange(len(y)), y], axis=1)
    assert (d_loo > d_naive - 1e-12).all()
    assert d_loo.mean() > d_naive.mean()


# --------------------------------------------------------------------------
# heads / probabilities
# --------------------------------------------------------------------------
def test_ensemble_outputs_are_probabilities():
    """The old ensemble mixed a rescaled distance with two probability outputs
    and thresholded the max. Every member must now live on one scale."""
    cfg = HeadConfig(members=["prototype", "random_forest", "logreg"], calibration="sigmoid",
                     calibration_cv=3, rf_n_estimators=30)
    df = toy_df(n_groups=45, copies=1, seed=1)
    E, y = toy_embeddings(df, seed=1)

    ens = ConservativeEnsemble(3, cfg).fit(E, y)
    for name, p in ens.member_probas(E).items():
        assert p.shape == (len(E), 3), name
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6, err_msg=name)
        assert (p >= 0).all() and (p <= 1).all(), name

    P = ens.predict_proba(E)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-6)


# --------------------------------------------------------------------------
# decision rule
# --------------------------------------------------------------------------
def test_threshold_hits_target_sensitivity_on_validation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    proba = rng.dirichlet([1, 1, 1], 300)
    # make malignant score informative
    proba[y == 1, 1] += 0.3
    proba /= proba.sum(1, keepdims=True)

    tau, info = select_threshold(proba, y, 1, rule="target_sensitivity", target_sensitivity=0.95)
    sens = (proba[y == 1, 1] >= tau).mean()
    assert sens >= 0.95 - 1e-9


def test_decision_is_frozen_and_reused_verbatim():
    """A Decision chosen on val must be applied to test unchanged. This test
    fails if anyone reintroduces a threshold sweep on the test set."""
    rng = np.random.default_rng(0)
    y_val = rng.integers(0, 3, 200)
    p_val = rng.dirichlet([2, 2, 2], 200)
    d = fit_decision(p_val, y_val, 1, DecisionConfig(target_malignant_sensitivity=0.9))

    assert d.chosen_on == "inner_val"
    with pytest.raises(Exception):  # frozen dataclass
        d.tau = 0.1  # type: ignore[misc]

    p_test = rng.dirichlet([2, 2, 2], 120)
    yhat_a, _ = apply_decision(p_test, d)
    yhat_b, _ = apply_decision(p_test, d)
    np.testing.assert_array_equal(yhat_a, yhat_b)


def test_tuning_threshold_on_the_test_set_inflates_sensitivity():
    """Demonstrates the size of the original error using pure noise.

    The submitted notebook swept thresholds on the set it reported, kept the
    smallest one reaching 100% malignant recall, and in the follow-up cell went
    further, setting the threshold directly from the confidence of the single
    malignant case it had missed. Below: with a classifier that has ZERO signal,
    that procedure still returns 100% malignant recall every time, while a
    threshold chosen on an independent split does not. A headline number that a
    random classifier also achieves is not evidence about the model.
    """
    oracle_sens, honest_sens = [], []
    for rep in range(25):
        rng = np.random.default_rng(rep)
        y = rng.integers(0, 3, 300)
        proba = rng.dirichlet([1, 1, 1], 300)  # no signal whatsoever
        val, test = np.arange(150), np.arange(150, 300)

        # (a) the original procedure: threshold read off the reported set itself
        tau_oracle = proba[test][y[test] == 1, 1].min() - 1e-12
        pred = np.where(proba[test][:, 1] >= tau_oracle, 1, proba[test].argmax(1))
        oracle_sens.append((pred[y[test] == 1] == 1).mean())

        # (b) the honest procedure: threshold fixed on validation, applied once
        tau, _ = select_threshold(proba[val], y[val], 1, "target_sensitivity", 0.95)
        pred = np.where(proba[test][:, 1] >= tau, 1, proba[test].argmax(1))
        honest_sens.append((pred[y[test] == 1] == 1).mean())

    print(f"\n  threshold tuned on the reported set: sensitivity "
          f"{np.mean(oracle_sens)*100:.1f}% (a random classifier!)"
          f"\n  threshold fixed on validation:       sensitivity "
          f"{np.mean(honest_sens)*100:.1f}%")
    assert np.mean(oracle_sens) == 1.0
    assert np.mean(honest_sens) < np.mean(oracle_sens)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_sensitivity_specificity_matches_hand_count():
    y = np.array([1, 1, 1, 0, 0, 2])
    p = np.array([1, 1, 0, 1, 0, 2])
    s = sensitivity_specificity(y, p, 1)
    assert s["tp"] == 2 and s["fn"] == 1 and s["fp"] == 1 and s["tn"] == 2
    assert s["sensitivity"] == pytest.approx(2 / 3)
    assert s["specificity"] == pytest.approx(2 / 3)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_wide_at_n42():
    """With ~42 malignant test cases the CI on sensitivity is roughly +/-10
    points. A bare '100%' without this interval overstates the evidence."""
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(42, int), np.zeros(114, int)])
    pred = y.copy()
    pred[:2] = 0  # 40/42
    proba = np.zeros((len(y), 2))
    proba[np.arange(len(y)), pred] = 1.0

    ci = bootstrap_ci(lambda t, p, _: sensitivity_specificity(t, p, 1)["sensitivity"],
                      y, pred, proba, n_boot=500, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] > 0.05


def test_coverage_is_reported_with_accuracy():
    """Abstention must never be reported as a free accuracy gain."""
    y = np.array([0, 1, 2, 1])
    pred = np.array([0, 1, 2, 0])
    proba = np.eye(3)[pred] * 0.9 + 0.05
    answered = np.array([True, True, True, False])
    m = compute_metrics(y, pred, proba, ["benign", "malignant", "normal"], 1, answered)
    assert m["coverage"] == 0.75
    assert m["n_answered"] == 3
    assert m["accuracy"] == 1.0  # perfect, but only on 75% of cases


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))


def test_cluster_bootstrap_is_wider_than_image_bootstrap():
    """BUS-BRA has 1875 images from 1064 patients: 811 patients contribute two
    views of the same lesion. Treating those as independent draws shrinks every
    interval. Here, two perfectly correlated copies per patient must not buy any
    extra precision."""
    rng = np.random.default_rng(0)
    n_pat = 200
    y_pat = rng.integers(0, 2, n_pat)
    correct = rng.random(n_pat) < 0.8

    # each patient contributes two identical (perfectly correlated) views
    y = np.repeat(y_pat, 2)
    groups = np.repeat(np.arange(n_pat), 2)
    pred = np.where(np.repeat(correct, 2), y, 1 - y)
    proba = np.eye(2)[pred] * 0.9 + 0.05

    acc = lambda t, p, _: float((t == p).mean())
    naive = bootstrap_ci(acc, y, pred, proba, n_boot=800, seed=0)
    clust = bootstrap_ci(acc, y, pred, proba, n_boot=800, seed=0, groups=groups)

    w_naive = naive["hi"] - naive["lo"]
    w_clust = clust["hi"] - clust["lo"]
    print(f"\n  image-level CI width:   {w_naive:.4f}  (claims n=400)"
          f"\n  patient-level CI width: {w_clust:.4f}  (honest n=200)"
          f"\n  understatement: {(1 - w_naive/w_clust)*100:.0f}%")
    assert naive["unit"] == "image" and clust["unit"] == "cluster"
    assert clust["n_clusters"] == n_pat
    assert w_clust > w_naive * 1.2


def test_label_conflicted_groups_are_dropped():
    """BUSI has near-duplicate pairs labelled benign in one folder and malignant
    in the other, including one byte-identical pixel duplicate. Their ground
    truth is unknowable, so they must leave the dataset entirely rather than be
    grouped (contradictory supervision) or randomly deduplicated (coin flip on
    which label is right)."""
    from splits import drop_conflicted_groups

    df = toy_df(n_groups=30, copies=2, seed=0)
    # forge a conflict: group 0's two copies get different labels
    idx = np.where(df["group"] == 0)[0]
    df.loc[idx[0], ["label", "class_name"]] = [0, "benign"]
    df.loc[idx[1], ["label", "class_name"]] = [1, "malignant"]

    kept, dropped = drop_conflicted_groups(df, verbose=False)
    assert len(dropped) == 2
    assert set(dropped["group"]) == {0}
    assert 0 not in set(kept["group"])
    assert len(kept) == len(df) - 2
    # every surviving group has exactly one label
    assert kept.groupby("group")["label"].nunique().eq(1).all()


def test_binary_task_end_to_end():
    """The 2-class path has to work, not just the 3-class one the original
    notebooks assumed. This caught a hardcoded multiclass XGBoost objective that
    made num_class=0 and crashed the whole run at fold 1.

    NOTE: xgboost is exercised here only if installed. The prototype and tree
    heads cover the rest of the binary code path either way."""
    members = ["prototype", "random_forest", "logreg"]
    try:
        import xgboost  # noqa: F401
        members.append("xgboost")
    except ImportError:
        pass

    cfg = HeadConfig(members=members, calibration="sigmoid", calibration_cv=3,
                     rf_n_estimators=30, xgb_n_estimators=30)
    df = toy_df(n_groups=60, copies=1, seed=2)
    df = df[df.label < 2].reset_index(drop=True)      # benign / malignant only
    E, y = toy_embeddings(df, seed=2)
    y = df.label.to_numpy()

    ens = ConservativeEnsemble(2, cfg).fit(E, y)
    P = ens.predict_proba(E)
    assert P.shape == (len(E), 2)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-6)

    # feature width must track the class count, not assume 3
    assert FeatureSpec(E.shape[1], 2).total == E.shape[1] + 6 + 4 + 1 + 3

    d = fit_decision(P, y, 1, DecisionConfig(target_malignant_sensitivity=0.9))
    yp, ans = apply_decision(P, d)
    m = compute_metrics(y, yp, P, ["benign", "malignant"], 1, ans)
    assert "auroc" in m or "auroc_error" in m
    assert 0.0 <= m["balanced_accuracy"] <= 1.0


def test_random_splitting_ablation_actually_leaks():
    """The `splitting='random'` ablation must genuinely reintroduce the defect,
    otherwise the ablation table understates what the defect was worth. Groups
    are expected to straddle here; that is the entire point."""
    df = toy_df(n_groups=40, copies=2, seed=1)

    honest = make_nested_splits(df, n_outer=5, seed=0, grouping="group")
    for s in honest:
        assert_no_leakage(df, s)  # must not raise

    leaky = make_nested_splits(df, n_outer=5, seed=0, grouping="random")
    straddles = 0
    for s in leaky:
        tr = set(df.iloc[np.concatenate([s.train_idx, s.val_idx])]["group"])
        te = set(df.iloc[s.test_idx]["group"])
        straddles += len(tr & te)
    assert straddles > 0, "random ablation failed to leak; it is not reproducing the defect"

    # and the leakage detector must still catch it when asked
    with pytest.raises(AssertionError, match="straddle"):
        assert_no_leakage(df, leaky[0])


def test_ablation_flags_are_recorded_in_the_decision():
    """A threshold chosen on test must be self-identifying in the saved artefact,
    so an ablation row can never be mistaken for a result later."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.dirichlet([2, 2], 200)

    honest = fit_decision(p, y, 1, DecisionConfig(rule="target_sensitivity"))
    assert honest.chosen_on == "inner_val"

    ablation = fit_decision(p, y, 1, DecisionConfig(rule="target_sensitivity"),
                            chosen_on="OUTER_TEST_ABLATION_NOT_A_RESULT")
    assert "ABLATION" in ablation.chosen_on
    assert "NOT_A_RESULT" in ablation.chosen_on
