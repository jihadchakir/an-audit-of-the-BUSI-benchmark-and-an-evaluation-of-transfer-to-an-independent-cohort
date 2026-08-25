from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass
class FoldSpec:
    fold: int
    train_idx: np.ndarray  # inner train
    val_idx: np.ndarray    # inner val
    test_idx: np.ndarray   # outer test, held out

    def sizes(self) -> dict:
        return {
            "train": int(len(self.train_idx)),
            "val": int(len(self.val_idx)),
            "test": int(len(self.test_idx)),
        }


def assert_no_leakage(
    df: pd.DataFrame,
    spec: FoldSpec,
    group_col: str = "group",
    path_col: str = "path",
    check_groups: bool = True,
) -> None:
    """Hard guarantees. Raises rather than warns, on purpose.

    check_groups=False is used ONLY by the `splitting="random"` ablation, which
    deliberately reproduces the leaky per-image split. Path-level checks still
    apply even there.
    """
    parts = {"train": spec.train_idx, "val": spec.val_idx, "test": spec.test_idx}

    # 1. no index appears twice
    all_idx = np.concatenate(list(parts.values()))
    if len(all_idx) != len(np.unique(all_idx)):
        raise AssertionError("An index appears in more than one split.")

    # 2. no file path appears in two splits
    for a in parts:
        for b in parts:
            if a >= b:
                continue
            pa = set(df.iloc[parts[a]][path_col])
            pb = set(df.iloc[parts[b]][path_col])
            overlap = pa & pb
            if overlap:
                raise AssertionError(f"{len(overlap)} paths shared between {a} and {b}: {list(overlap)[:3]}")

    # 3. no duplicate group straddles a split
    for a in (parts if check_groups else {}):
        for b in parts:
            if a >= b:
                continue
            ga = set(df.iloc[parts[a]][group_col])
            gb = set(df.iloc[parts[b]][group_col])
            overlap = ga & gb
            if overlap:
                raise AssertionError(
                    f"{len(overlap)} duplicate-groups straddle {a}/{b}. "
                    f"Near-duplicate images would leak. Groups: {sorted(overlap)[:5]}"
                )

    # 4. every class present in every split (otherwise metrics are undefined)
    for name, idx in parts.items():
        present = set(df.iloc[idx]["label"])
        if len(present) < df["label"].nunique():
            missing = set(df["label"].unique()) - present
            raise AssertionError(f"Split '{name}' is missing class(es) {missing}.")


def make_nested_splits(
    df: pd.DataFrame,
    n_outer: int = 5,
    inner_val_fraction: float = 0.2,
    seed: int = 42,
    group_col: str = "group",
    label_col: str = "label",
    grouping: str = "group",
) -> List[FoldSpec]:
    """Build all outer folds. Every fold serves as test exactly once.

    The inner train/val split is itself group-aware: it is one fold of a
    StratifiedGroupKFold over the outer-train portion, with
    n_splits = round(1 / inner_val_fraction).
    """
    if group_col not in df.columns:
        raise KeyError(
            f"'{group_col}' column missing. Run audit_dataset.py first and load "
            "index_with_groups.csv."
        )

    if grouping not in ("group", "random"):
        raise ValueError(f"grouping must be 'group' or 'random', got '{grouping}'")
    if grouping == "random":
        print("\n" + "!" * 74)
        print("!! ABLATION: splitting='random'. Near-duplicate copies of the same scan")
        print("!! WILL land on both sides of the split. This reproduces the original")
        print("!! notebooks' protocol. The resulting numbers are NOT a result.")
        print("!" * 74 + "\n")

    y = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()
    X = np.zeros(len(df))

    inner_k = max(2, int(round(1.0 / inner_val_fraction)))
    if grouping == "group":
        outer_iter = StratifiedGroupKFold(n_splits=n_outer, shuffle=True,
                                          random_state=seed).split(X, y, groups)
    else:
        outer_iter = StratifiedKFold(n_splits=n_outer, shuffle=True,
                                     random_state=seed).split(X, y)

    specs: List[FoldSpec] = []
    for fold, (outer_train_idx, test_idx) in enumerate(outer_iter):
        sub_y = y[outer_train_idx]
        if grouping == "group":
            inner = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=seed + fold)
            tr_rel, va_rel = next(inner.split(np.zeros(len(outer_train_idx)), sub_y,
                                             groups[outer_train_idx]))
        else:
            inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed + fold)
            tr_rel, va_rel = next(inner.split(np.zeros(len(outer_train_idx)), sub_y))

        spec = FoldSpec(
            fold=fold + 1,
            train_idx=outer_train_idx[tr_rel],
            val_idx=outer_train_idx[va_rel],
            test_idx=test_idx,
        )
        assert_no_leakage(df, spec, group_col=group_col, check_groups=(grouping == "group"))
        specs.append(spec)

    return specs


def describe_splits(df: pd.DataFrame, specs: List[FoldSpec], class_names: List[str]) -> pd.DataFrame:
    """Table for the supplementary material."""
    rows = []
    for spec in specs:
        for part, idx in (("train", spec.train_idx), ("val", spec.val_idx), ("test", spec.test_idx)):
            counts = df.iloc[idx]["label"].value_counts()
            row = {"fold": spec.fold, "split": part, "n": len(idx),
                   "n_groups": df.iloc[idx]["group"].nunique()}
            for i, name in enumerate(class_names):
                row[name] = int(counts.get(i, 0))
            rows.append(row)
    return pd.DataFrame(rows)


def filter_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Restrict to the requested label space and re-index labels contiguously."""
    if task == "3class":
        return df.reset_index(drop=True)
    if task == "2class":
        out = df[df["class_name"].isin(["benign", "malignant"])].copy()
        mapping = {"benign": 0, "malignant": 1}
        out["label"] = out["class_name"].map(mapping)
        return out.reset_index(drop=True)
    raise ValueError(f"Unknown task '{task}'. Use '3class' or '2class'.")


def drop_conflicted_groups(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove every image in a duplicate group that contains more than one label.

    Returns (kept, dropped). The dropped frame is not discarded silently: it is
    written to the run directory so the exclusion is auditable.

    Rationale: a near-duplicate pair with conflicting labels means at least one
    of the two is wrong, and nothing in the data says which. Keeping them either
    trains the model on a contradiction or scores it on a case no model could
    get right. Both corrupt the estimate. Excluding them is the only defensible
    option, and it must be reported, not buried.
    """
    n_lab = df.groupby("group")["label"].nunique()
    bad_groups = set(n_lab[n_lab > 1].index)
    mask = df["group"].isin(bad_groups)
    kept, dropped = df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
    if verbose and len(dropped):
        combos = (dropped.groupby("group")["class_name"]
                  .apply(lambda s: " + ".join(sorted(set(s)))).value_counts().to_dict())
        print(f"[splits] dropped {len(dropped)} images in {len(bad_groups)} label-conflicted "
              f"duplicate groups ({len(dropped)/len(df)*100:.1f}% of the dataset)")
        for k, v in combos.items():
            print(f"           {k}: {v} group(s)")
    return kept, dropped


def apply_duplicate_policy(df: pd.DataFrame, policy: str, seed: int = 42) -> pd.DataFrame:
    """'group' keeps every image (grouped splitting handles the leak).
    'drop' keeps one representative per duplicate group, which shrinks the
    dataset but removes the memorisation advantage entirely. Report both."""
    if policy == "group":
        return df
    if policy == "drop":
        rng = np.random.default_rng(seed)
        keep_idx = []
        for _, idx in df.groupby("group").indices.items():
            keep_idx.append(int(idx[rng.integers(len(idx))]))
        return df.iloc[sorted(keep_idx)].sort_values("path").reset_index(drop=True)
    raise ValueError(f"Unknown duplicate_policy '{policy}'.")
