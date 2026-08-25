from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Decision:
    """A frozen operating point. Chosen on validation, applied to test as-is."""

    rule: str
    malignant_index: int
    tau: float
    abstain_below: Optional[float]
    # provenance, for the audit trail
    chosen_on: str = "inner_val"
    target_sensitivity: Optional[float] = None
    achieved_val_sensitivity: Optional[float] = None
    achieved_val_specificity: Optional[float] = None
    achieved_val_coverage: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _binary_scores(proba: np.ndarray, malignant_index: int) -> np.ndarray:
    return proba[:, malignant_index]


def select_threshold(
    val_proba: np.ndarray,
    val_y: np.ndarray,
    malignant_index: int,
    rule: str = "target_sensitivity",
    target_sensitivity: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """Choose tau on the validation split.

    rule:
      argmax             -> tau = None-equivalent (0.5 on the malignant score is
                            not used; prediction is plain argmax). Returned tau
                            is 1.0 as a sentinel meaning "never override argmax".
      target_sensitivity -> smallest tau whose validation malignant sensitivity
                            is >= target. Ties broken by best specificity.
      youden             -> maximise sensitivity + specificity - 1.
    """
    s = _binary_scores(val_proba, malignant_index)
    is_mal = val_y == malignant_index

    if rule == "argmax":
        return 1.0, {"note": "argmax rule, no override"}

    # candidate thresholds: midpoints between observed scores, so the choice is
    # not tied to an arbitrary grid like [0.10, 0.15, ...]
    uniq = np.unique(s)
    cands = np.concatenate([[0.0], (uniq[:-1] + uniq[1:]) / 2.0, [1.0]]) if len(uniq) > 1 else np.array([0.0, 1.0])

    sens = np.array([(s[is_mal] >= t).mean() if is_mal.any() else 0.0 for t in cands])
    spec = np.array([(s[~is_mal] < t).mean() if (~is_mal).any() else 0.0 for t in cands])

    if rule == "target_sensitivity":
        ok = np.where(sens >= target_sensitivity)[0]
        if len(ok) == 0:
            # cannot reach the target on validation; take the most sensitive point
            # and say so loudly rather than silently lowering the bar
            best = int(np.argmax(sens))
            print(
                f"[decision] WARNING: target sensitivity {target_sensitivity:.2f} unreachable "
                f"on validation (max {sens.max():.3f}). Using the most sensitive threshold."
            )
        else:
            best = int(ok[np.argmax(spec[ok])])
    elif rule == "youden":
        best = int(np.argmax(sens + spec - 1.0))
    else:
        raise ValueError(f"Unknown rule '{rule}'")

    return float(cands[best]), {
        "val_sensitivity": float(sens[best]),
        "val_specificity": float(spec[best]),
    }


def select_abstention(
    val_proba: np.ndarray,
    target_coverage: float = 0.90,
) -> float:
    """Confidence floor giving the requested coverage on validation.

    Cases below the floor are referred to a radiologist rather than scored.
    Coverage is reported next to every accuracy figure; an abstaining system
    that is accurate on 60% of cases is not comparable to one that answers
    everything, and the paper must not present it as if it were.
    """
    conf = val_proba.max(axis=1)
    q = float(np.quantile(conf, 1.0 - target_coverage))
    return q


def apply_decision(proba: np.ndarray, decision: Decision) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (y_pred, answered_mask).

    Rule: if the malignant posterior clears tau, call it malignant (this is the
    'conservative' part, and it is intentionally allowed to override argmax).
    Otherwise take argmax. Then, if abstention is enabled and the top posterior
    is below the floor, mark the case as not answered.
    """
    y_pred = proba.argmax(axis=1)
    if decision.rule != "argmax":
        override = proba[:, decision.malignant_index] >= decision.tau
        y_pred = np.where(override, decision.malignant_index, y_pred)

    answered = np.ones(len(proba), dtype=bool)
    if decision.abstain_below is not None:
        answered = proba.max(axis=1) >= decision.abstain_below
        # never abstain on a case the conservative rule flagged as malignant:
        # flagging for biopsy IS the answer in that case
        if decision.rule != "argmax":
            answered = answered | (proba[:, decision.malignant_index] >= decision.tau)

    return y_pred, answered


def fit_decision(
    val_proba: np.ndarray,
    val_y: np.ndarray,
    malignant_index: int,
    cfg,
    chosen_on: str = "inner_val",
) -> Decision:
    """Everything about the operating point, decided here, on validation.

    `chosen_on` is recorded in the returned Decision and serialised with the
    fold. If it says anything other than 'inner_val', the row is an ablation.
    """
    tau, info = select_threshold(
        val_proba,
        val_y,
        malignant_index,
        rule=cfg.rule,
        target_sensitivity=cfg.target_malignant_sensitivity,
    )
    floor = select_abstention(val_proba, cfg.target_coverage) if cfg.enable_abstention else None

    d = Decision(
        rule=cfg.rule,
        malignant_index=malignant_index,
        tau=tau,
        abstain_below=floor,
        chosen_on=chosen_on,
        target_sensitivity=cfg.target_malignant_sensitivity if cfg.rule == "target_sensitivity" else None,
        achieved_val_sensitivity=info.get("val_sensitivity"),
        achieved_val_specificity=info.get("val_specificity"),
    )
    _, answered = apply_decision(val_proba, d)
    return Decision(
        **{**d.to_dict(), "achieved_val_coverage": float(answered.mean())}
    )
