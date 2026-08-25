
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    root: str = "./Dataset_BUSI"
    # 3-class ("benign", "malignant", "normal") or 2-class ("benign", "malignant").
    # 2-class is the clinically meaningful task and the only one that transfers to
    # external sets (BUS-BRA / UDIAT have no "normal" category).
    task: str = "3class"
    img_size: int = 256

    # Ground-truth masks are NEVER used to build model input. See CHANGES.md #1.
    # These flags exist only for the ablation table.
    #
    #   "none"      the rebuild. No mask, ever.
    #   "soft_dim"  the original notebooks' variant:
    #                   out = img*m + 0.3*img*(1-m)
    #               Note this is NOT what the BUSI literature does. It is one
    #               person's choice and ablating it measures that choice.
    #   "roi_crop"  THE FIELD'S ACTUAL PRACTICE: crop to the bounding box of the
    #               ground-truth mask, with padding, then resize. This is what
    #               published BUSI classification papers mean by "ROI-focused
    #               preprocessing", and it is advocated as an improvement rather
    #               than recognised as leakage. Ablating THIS is what makes a
    #               finding about the literature instead of about one notebook.
    use_gt_mask_oracle: bool = False   # back-compat: True == mask_mode "soft_dim"
    mask_mode: str = "none"            # none | soft_dim | roi_crop
    roi_pad_ratio: float = 0.10        # bbox padding, as a fraction of w/h

    # In 3-class BUSI a `normal` case has an all-black mask, so there IS no
    # bounding box to crop. Every paper doing 3-class with ROI cropping must do
    # *something* here, and whatever it does, the normal class gets preprocessed
    # by a different function than the lesion classes, selected by the label.
    # That is the structural problem. This flag makes the choice explicit.
    #   full_image      normals stay whole frames (~1x zoom vs the lesions' ~3.5x)
    #   center_crop     normals get a fixed central crop (~1.8x)
    #   matched_random  THE CONTROL: crop normals to a box drawn from the LESION
    #                   bounding-box size distribution, placed at random. Same
    #                   field of view, same zoom, same resampling scale as a
    #                   lesion crop, with no lesion in it. If normal recall
    #                   collapses here, the artifact was field-of-view rather
    #                   than anatomy, and the structural claim is proved.
    roi_normal_fallback: str = "full_image"  # full_image | center_crop | matched_random

    # Near-duplicate handling. BUSI is known to contain duplicated and
    # near-duplicated images (Aumente-Maestro et al., CMPB 2025).
    # "group"  -> keep all copies but force them into the same CV fold
    # "drop"   -> keep one representative per duplicate group
    duplicate_policy: str = "group"
    phash_hamming_threshold: int = 6

    # Images whose duplicate carries a DIFFERENT label have no usable ground
    # truth: you cannot train on contradictory supervision and you cannot be
    # scored on a case whose label contradicts itself. BUSI has 10 such pairs,
    # one of them a byte-identical pixel duplicate filed under two classes.
    # Excluded by default; the count is recorded in the run config.
    # See review_conflicts.py and CHANGES.md #5.
    drop_conflicted_groups: bool = True

    num_workers: int = 4


@dataclass
class EncoderConfig:
    backbone: str = "resnet50"
    fpn_channels: int = 256
    embedding_dim: int = 256
    pretrained: bool = True
    freeze_bn: bool = False
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 60
    # PK sampler: p_classes * k_per_class = effective batch size
    p_classes: int = 3
    k_per_class: int = 8
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    margin: float = 0.3
    mining: str = "batch_hard"  # batch_hard | batch_semihard | soft_margin
    warmup_epochs: int = 3
    patience: int = 12
    # Early stopping / model selection metric computed on the inner validation
    # split. NOT the triplet loss on random triplets. See CHANGES.md #6.
    select_metric: str = "val_balanced_accuracy"
    amp_dtype: str = "bf16"  # bf16 | fp16 | fp32
    grad_clip: float = 1.0


@dataclass
class HeadConfig:
    members: List[str] = field(default_factory=lambda: ["prototype", "xgboost", "random_forest"])
    renormalize_prototypes: bool = True
    calibration: str = "sigmoid"  # sigmoid | isotonic | none
    calibration_cv: int = 5
    ensemble: str = "soft_vote"  # soft_vote | weighted_vote
    rf_n_estimators: int = 400
    rf_max_depth: Optional[int] = None
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 4
    xgb_lr: float = 0.05


@dataclass
class DecisionConfig:
    # The operating point is chosen on the inner validation split ONLY and then
    # applied to the outer test fold exactly once. See CHANGES.md #3.
    rule: str = "argmax"  # argmax | target_sensitivity | youden

    # ABLATION ONLY. "test" reproduces the original notebooks' protocol: choose
    # the threshold on the set you then report. It exists so the ablation table
    # can put a number on what that was worth. Never report a "test" row as a
    # result. See CHANGES.md #2.
    select_on: str = "val"  # val | test
    target_malignant_sensitivity: float = 0.95
    # Referral / abstention: if the top probability is below this, the case is
    # sent to a human instead of being scored. Chosen on validation, coverage
    # is always reported alongside accuracy.
    enable_abstention: bool = True
    target_coverage: float = 0.90


@dataclass
class EvalConfig:
    n_bootstrap: int = 2000
    ci_alpha: float = 0.05
    seed: int = 1337


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ABLATION ONLY. "random" reproduces the original notebooks' per-image
    # train_test_split, which put near-duplicate copies of the same scan on both
    # sides of the split. See CHANGES.md #5.
    splitting: str = "group"  # group | random

    n_outer_folds: int = 5
    inner_val_fraction: float = 0.2
    seed: int = 42
    out_dir: str = "./runs/rebuild"

    @property
    def class_names(self) -> List[str]:
        if self.data.task == "2class":
            return ["benign", "malignant"]
        return ["benign", "malignant", "normal"]

    @property
    def malignant_index(self) -> int:
        return self.class_names.index("malignant")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            data=DataConfig(**raw["data"]),
            encoder=EncoderConfig(**raw["encoder"]),
            train=TrainConfig(**raw["train"]),
            head=HeadConfig(**raw["head"]),
            decision=DecisionConfig(**raw["decision"]),
            eval=EvalConfig(**raw["eval"]),
            n_outer_folds=raw["n_outer_folds"],
            inner_val_fraction=raw["inner_val_fraction"],
            seed=raw["seed"],
            out_dir=raw["out_dir"],
        )
