from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar


# --------------------------------------------------------------------------
# prototypes
# --------------------------------------------------------------------------
def class_prototypes(E: np.ndarray, y: np.ndarray, n_classes: int, renormalize: bool = True) -> np.ndarray:
    """(C, D) class means of the embeddings."""
    D = E.shape[1]
    protos = np.zeros((n_classes, D), dtype=np.float64)
    for c in range(n_classes):
        m = y == c
        if m.sum() == 0:
            raise ValueError(f"No training samples for class {c}; cannot build a prototype.")
        protos[c] = E[m].mean(axis=0)
    if renormalize:
        protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12
    return protos


def loo_prototypes(
    E: np.ndarray, y: np.ndarray, n_classes: int, renormalize: bool = True
) -> np.ndarray:
    """(N, C, D) prototypes with row i excluded from its own class mean.

    Exact and cheap: mean_{-i} = (sum_c - E_i) / (n_c - 1).
    Use for TRAIN rows only. Val/test rows use the shared prototypes.
    """
    N, D = E.shape
    sums = np.zeros((n_classes, D), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        m = y == c
        counts[c] = int(m.sum())
        sums[c] = E[m].sum(axis=0)

    base = sums / np.maximum(counts, 1)[:, None]
    P = np.repeat(base[None, :, :], N, axis=0)
    for i in range(N):
        c = int(y[i])
        if counts[c] > 1:
            P[i, c] = (sums[c] - E[i]) / (counts[c] - 1)
    if renormalize:
        P /= np.linalg.norm(P, axis=2, keepdims=True) + 1e-12
    return P


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    embedding_dim: int
    n_classes: int

    @property
    def n_pair_ratios(self) -> int:
        return self.n_classes * (self.n_classes - 1) // 2

    @property
    def total(self) -> int:
        return (
            self.embedding_dim          # raw embedding
            + 3 * self.n_classes        # euclidean, cosine, manhattan per class
            + 4                         # mean, std, min, max of euclidean dists
            + self.n_pair_ratios        # pairwise distance ratios
            + 3                         # top-1 dist, top-2 dist, margin
        )

    def breakdown(self) -> Dict[str, int]:
        return {
            "embedding": self.embedding_dim,
            "per_class_distances(euclid,cos,manhattan)": 3 * self.n_classes,
            "distance_summary(mean,std,min,max)": 4,
            "pairwise_distance_ratios": self.n_pair_ratios,
            "nearest_two(d1,d2,margin)": 3,
            "TOTAL": self.total,
        }


def make_features(E: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Build the meta-feature matrix.

    E:      (N, D) embeddings
    protos: (C, D) shared prototypes, or (N, C, D) per-row (leave-one-out).
    """
    E = np.asarray(E, dtype=np.float64)
    N, D = E.shape
    if protos.ndim == 2:
        P = np.repeat(protos[None, :, :], N, axis=0)
    elif protos.ndim == 3:
        P = protos
        if P.shape[0] != N:
            raise ValueError(f"per-row prototypes have {P.shape[0]} rows, expected {N}")
    else:
        raise ValueError("protos must be (C, D) or (N, C, D)")

    C = P.shape[1]
    diff = E[:, None, :] - P                                    # (N, C, D)
    euclid = np.linalg.norm(diff, axis=2)                       # (N, C)
    manhattan = np.abs(diff).sum(axis=2)                        # (N, C)
    e_norm = np.linalg.norm(E, axis=1, keepdims=True)           # (N, 1)
    p_norm = np.linalg.norm(P, axis=2)                          # (N, C)
    cosine = np.einsum("nd,ncd->nc", E, P) / (e_norm * p_norm + 1e-8)

    blocks: List[np.ndarray] = [E]
    # interleave per class to keep the original feature ordering semantics
    per_class = np.empty((N, 3 * C))
    for c in range(C):
        per_class[:, 3 * c + 0] = euclid[:, c]
        per_class[:, 3 * c + 1] = cosine[:, c]
        per_class[:, 3 * c + 2] = manhattan[:, c]
    blocks.append(per_class)

    blocks.append(
        np.stack(
            [euclid.mean(1), euclid.std(1), euclid.min(1), euclid.max(1)], axis=1
        )
    )

    ratios = np.stack(
        [euclid[:, a] / (euclid[:, b] + 1e-8) for a, b in combinations(range(C), 2)], axis=1
    )
    blocks.append(ratios)

    order = np.sort(euclid, axis=1)
    blocks.append(np.stack([order[:, 0], order[:, 1], order[:, 1] - order[:, 0]], axis=1))

    F = np.concatenate(blocks, axis=1)
    expected = FeatureSpec(D, C).total
    if F.shape[1] != expected:
        raise AssertionError(f"feature width {F.shape[1]} != spec {expected}")
    return F


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------
class PrototypeHead:
    """Nearest-prototype classifier with a calibrated softmax over -distance.

    The temperature is fit by minimising NLL on leave-one-out training
    distances, so no validation data is consumed here (validation is reserved
    for the operating point).
    """

    name = "prototype"

    def __init__(self, n_classes: int, renormalize: bool = True):
        self.n_classes = n_classes
        self.renormalize = renormalize
        self.protos_: Optional[np.ndarray] = None
        self.temperature_: float = 1.0

    def fit(self, E: np.ndarray, y: np.ndarray) -> "PrototypeHead":
        self.protos_ = class_prototypes(E, y, self.n_classes, self.renormalize)
        loo = loo_prototypes(E, y, self.n_classes, self.renormalize)
        d = np.linalg.norm(E[:, None, :] - loo, axis=2)  # (N, C)

        def nll(log_t: float) -> float:
            t = np.exp(log_t)
            logits = -d / t
            logits -= logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            return float(-logp[np.arange(len(y)), y].mean())

        res = minimize_scalar(nll, bounds=(-6.0, 3.0), method="bounded")
        self.temperature_ = float(np.exp(res.x))
        return self

    def predict_proba(self, E: np.ndarray) -> np.ndarray:
        if self.protos_ is None:
            raise RuntimeError("PrototypeHead is not fitted")
        d = np.linalg.norm(E[:, None, :] - self.protos_[None, :, :], axis=2)
        logits = -d / self.temperature_
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


def _make_base(kind: str, cfg) -> BaseEstimator:
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "xgboost is required for the 'xgboost' head. "
                "pip install xgboost, or drop it from head.members in config.py"
            ) from exc
        return XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_lr,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            # `objective` is deliberately NOT set. XGBClassifier infers it from
            # the number of classes at fit time: binary:logistic for 2,
            # multi:softprob for more. Hardcoding "multi:softprob" (as the
            # original notebook did, being 3-class only) breaks the 2-class task:
            # the wrapper only populates `num_class` when it detects >2 classes,
            # so num_class stays 0 and XGBoost raises
            #   "value 0 for Parameter num_class should be greater equal to 1".
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
    if kind == "logreg":  # used by the test suite, cheap and dependency free
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, class_weight="balanced")
    raise ValueError(f"Unknown head '{kind}'")


class MetaHead:
    """Scaler -> base classifier -> probability calibration, all fit on train only.

    Calibration uses internal K-fold on the training set (CalibratedClassifierCV),
    NOT the validation split, so that validation stays clean for choosing the
    operating point.
    """

    def __init__(self, kind: str, cfg):
        self.name = kind
        self.kind = kind
        self.cfg = cfg
        self.pipe_: Optional[Pipeline] = None

    def fit(self, F: np.ndarray, y: np.ndarray) -> "MetaHead":
        base = _make_base(self.kind, self.cfg)
        if self.cfg.calibration == "none":
            clf = base
        else:
            n_min = int(np.bincount(y).min())
            cv = max(2, min(self.cfg.calibration_cv, n_min))
            clf = CalibratedClassifierCV(estimator=base, method=self.cfg.calibration, cv=cv)
        self.pipe_ = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.pipe_.fit(F, y)
        return self

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        if self.pipe_ is None:
            raise RuntimeError(f"MetaHead({self.kind}) is not fitted")
        return self.pipe_.predict_proba(F)


class ConservativeEnsemble:
    """Soft-voting ensemble over calibrated members.

    The word "conservative" now refers to the *operating point* (chosen on
    validation to hit a target malignant sensitivity, see decision.py), not to
    an ad-hoc max-of-scores rule. The ensemble itself is just an average of
    posteriors, which keeps the output a probability that can be calibrated,
    thresholded, and reported with a proper scoring rule.
    """

    def __init__(self, n_classes: int, cfg):
        self.n_classes = n_classes
        self.cfg = cfg
        self.members: List = []
        self.weights_: Optional[np.ndarray] = None
        self.feature_spec_: Optional[FeatureSpec] = None

    def fit(self, E_train: np.ndarray, y_train: np.ndarray) -> "ConservativeEnsemble":
        E_train = np.asarray(E_train, dtype=np.float64)
        self.feature_spec_ = FeatureSpec(E_train.shape[1], self.n_classes)

        proto = PrototypeHead(self.n_classes, self.cfg.renormalize_prototypes).fit(E_train, y_train)
        self.proto_ = proto

        # leave-one-out prototypes for the TRAIN features, shared prototypes at
        # inference. This is the fix for (b) at the top of this file.
        loo = loo_prototypes(E_train, y_train, self.n_classes, self.cfg.renormalize_prototypes)
        F_train = make_features(E_train, loo)

        self.members = []
        for kind in self.cfg.members:
            if kind == "prototype":
                self.members.append(proto)
            else:
                self.members.append(MetaHead(kind, self.cfg).fit(F_train, y_train))
        self.weights_ = np.ones(len(self.members)) / len(self.members)
        return self

    def _features(self, E: np.ndarray) -> np.ndarray:
        return make_features(np.asarray(E, dtype=np.float64), self.proto_.protos_)

    def member_probas(self, E: np.ndarray) -> Dict[str, np.ndarray]:
        F = self._features(E)
        out: Dict[str, np.ndarray] = {}
        for m in self.members:
            out[m.name] = m.predict_proba(E) if isinstance(m, PrototypeHead) else m.predict_proba(F)
        return out

    def predict_proba(self, E: np.ndarray) -> np.ndarray:
        probas = self.member_probas(E)
        stack = np.stack([probas[m.name] for m in self.members])  # (M, N, C)
        w = self.weights_[:, None, None]
        return (stack * w).sum(0)

    def set_weights_from_val(self, E_val: np.ndarray, y_val: np.ndarray) -> None:
        """Optional: weight members by validation log-loss. Uses validation, so if
        you call this you must NOT also use validation for the threshold without
        splitting it. Off by default; ensemble='soft_vote' keeps uniform weights."""
        if self.cfg.ensemble != "weighted_vote":
            return
        probas = self.member_probas(E_val)
        scores = []
        for m in self.members:
            p = np.clip(probas[m.name][np.arange(len(y_val)), y_val], 1e-12, 1.0)
            scores.append(-np.log(p).mean())
        inv = 1.0 / (np.array(scores) + 1e-8)
        self.weights_ = inv / inv.sum()
