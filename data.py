from __future__ import annotations

import zlib
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_gray(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


class BUSIDataset(Dataset):
    """Grayscale ultrasound -> 3-channel ImageNet-normalised tensor.

    Note on normalisation: the original code fed raw [0, 1] RGB into a
    ResNet50 initialised with ImageNet weights, which expects mean/std
    normalised input. That mismatch alone costs accuracy and makes the
    'pretrained' claim shaky. Fixed here.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 256,
        train: bool = False,
        use_gt_mask_oracle: bool = False,
        mask_mode: str = "none",
        roi_pad_ratio: float = 0.10,
        roi_normal_fallback: str = "full_image",
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train
        # back-compat: the old boolean means the soft-dim variant
        if use_gt_mask_oracle and mask_mode == "none":
            mask_mode = "soft_dim"
        self.mask_mode = mask_mode
        self.use_gt_mask_oracle = mask_mode != "none"
        self.roi_pad_ratio = roi_pad_ratio
        self.roi_normal_fallback = roi_normal_fallback
        self.labels = self.df["label"].to_numpy()
        self._lesion_box_shapes = None
        if self.mask_mode == "roi_crop" and self.roi_normal_fallback == "matched_random":
            self._lesion_box_shapes = self._collect_lesion_box_shapes()

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape
        # random resized crop, mild: ultrasound geometry is meaningful
        scale = rng.uniform(0.85, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = rng.integers(0, h - ch + 1)
        x0 = rng.integers(0, w - cw + 1)
        img = img[y0:y0 + ch, x0:x0 + cw]

        if rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])

        angle = rng.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)

        # gain / contrast jitter mimics scanner TGC settings, not free lunch
        alpha = rng.uniform(0.9, 1.1)
        beta = rng.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # speckle-ish noise
        if rng.random() < 0.3:
            noise = rng.normal(0, 6, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def _collect_lesion_box_shapes(self) -> np.ndarray:
        """(N, 2) array of lesion bbox sizes as FRACTIONS of the frame.

        Fractions, not pixels, so the box transfers across BUSI's very variable
        image sizes. Collected once, from this split's own rows.
        """
        shapes = []
        for _, row in self.df.iterrows():
            mps = [p for p in str(row.get("mask_paths", "")).split("|") if p]
            if not mps:
                continue
            m = cv2.imread(mps[0], cv2.IMREAD_GRAYSCALE)
            if m is None or m.max() == 0:
                continue
            ys, xs = np.where(m > 0)
            h = (ys.max() - ys.min() + 1) * (1 + 2 * self.roi_pad_ratio)
            w = (xs.max() - xs.min() + 1) * (1 + 2 * self.roi_pad_ratio)
            shapes.append([min(h / m.shape[0], 1.0), min(w / m.shape[1], 1.0)])
        if not shapes:
            raise RuntimeError("matched_random needs lesion masks to sample box sizes from")
        return np.asarray(shapes, dtype=np.float64)

    def _apply_mask_oracle(self, img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if self.mask_mode == "soft_dim":
            if mask is None:
                return img
            m = mask.astype(np.float32) / 255.0
            return np.clip(img * m + img * 0.3 * (1 - m), 0, 255).astype(np.uint8)

        if self.mask_mode == "roi_crop":
            # The field's practice: crop to the ground-truth lesion bounding box.
            if mask is None or mask.max() == 0:
                # No lesion outlined => nothing to crop. This is the `normal`
                # class in 3-class BUSI, and it is exactly where the structural
                # problem lives: the normal class necessarily gets a DIFFERENT
                # preprocessing function, chosen by its label.
                if self.roi_normal_fallback == "center_crop":
                    h, w = img.shape
                    side = int(min(h, w) * 0.7)
                    y0, x0 = (h - side) // 2, (w - side) // 2
                    return img[y0:y0 + side, x0:x0 + side]
                if self.roi_normal_fallback == "matched_random":
                    # THE CONTROL: give the normal case the same field of view a
                    # lesion crop would have had, at a random location.
                    h, w = img.shape
                    fh, fw = self._lesion_box_shapes[
                        np.random.default_rng(zlib.crc32(str(id(img)).encode())
                                              % (2**32)).integers(len(self._lesion_box_shapes))]
                    ch, cw = max(8, int(h * fh)), max(8, int(w * fw))
                    ch, cw = min(ch, h), min(cw, w)
                    rng = np.random.default_rng(zlib.crc32(img.tobytes()[:512]) % (2**32))
                    y0 = int(rng.integers(0, h - ch + 1))
                    x0 = int(rng.integers(0, w - cw + 1))
                    return img[y0:y0 + ch, x0:x0 + cw]
                return img  # full_image
            ys, xs = np.where(mask > 0)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            ph = int((y1 - y0 + 1) * self.roi_pad_ratio)
            pw = int((x1 - x0 + 1) * self.roi_pad_ratio)
            y0 = max(0, y0 - ph); y1 = min(img.shape[0] - 1, y1 + ph)
            x0 = max(0, x0 - pw); x1 = min(img.shape[1] - 1, x1 + pw)
            return img[y0:y1 + 1, x0:x1 + 1]

        raise ValueError(f"Unknown mask_mode '{self.mask_mode}'")

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[i]
        img = _load_gray(row["path"])
        if img is None:
            raise RuntimeError(f"Unreadable image: {row['path']}")

        if self.mask_mode != "none":
            # ORACLE ABLATION ONLY. Reintroduces the ground-truth mask.
            mask_paths = [p for p in str(row.get("mask_paths", "")).split("|") if p]
            mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE) if mask_paths else None
            if mask is not None and mask.shape != img.shape:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            img = self._apply_mask_oracle(img, mask)

        if self.train:
            # crc32, not hash(): Python's str hash is salted per process, so
            # hash()-derived seeds are not reproducible across runs
            path_seed = zlib.crc32(str(row["path"]).encode()) & 0xFFFFFFFF
            rng = np.random.default_rng((path_seed ^ (torch.initial_seed() & 0xFFFFFFFF)))
            img = self._augment(img, rng)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        x = np.stack([x, x, x], axis=0)  # 3 channels for the ImageNet backbone
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
        x = (x - mean) / std
        return torch.from_numpy(x), int(row["label"])


class PKSampler(Sampler[List[int]]):
    """Yields batches of P classes x K images.

    This is what makes batch-hard triplet mining possible: every batch is
    guaranteed to contain positives and negatives for every anchor. The old
    generator drew triplets uniformly at random from the whole training set,
    so after a few epochs almost every triplet was already satisfied, the loss
    collapsed towards zero, and `val_loss` stopped carrying information about
    embedding quality. That is why "best val_loss" selected a poor model.
    """

    def __init__(self, labels: Sequence[int], p_classes: int, k_per_class: int, seed: int = 42,
                 length: Optional[int] = None):
        self.labels = np.asarray(labels)
        self.p = p_classes
        self.k = k_per_class
        self.rng = np.random.default_rng(seed)
        self.classes = np.unique(self.labels)
        if self.p > len(self.classes):
            self.p = len(self.classes)
        self.index_by_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.batch_size = self.p * self.k
        self.length = length or max(1, len(self.labels) // self.batch_size)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.length):
            chosen = self.rng.choice(self.classes, size=self.p, replace=False)
            batch: List[int] = []
            for c in chosen:
                pool = self.index_by_class[c]
                replace = len(pool) < self.k
                batch.extend(self.rng.choice(pool, size=self.k, replace=replace).tolist())
            yield batch


def make_loaders(df_train, df_val, cfg, seed: int = 42):
    from torch.utils.data import DataLoader

    kw = dict(mask_mode=cfg.data.mask_mode, roi_pad_ratio=cfg.data.roi_pad_ratio,
              roi_normal_fallback=cfg.data.roi_normal_fallback,
              use_gt_mask_oracle=cfg.data.use_gt_mask_oracle)
    ds_train = BUSIDataset(df_train, cfg.data.img_size, train=True, **kw)
    ds_val = BUSIDataset(df_val, cfg.data.img_size, train=False, **kw)

    sampler = PKSampler(ds_train.labels, cfg.train.p_classes, cfg.train.k_per_class, seed=seed)
    train_loader = DataLoader(
        ds_train, batch_sampler=sampler, num_workers=cfg.data.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=32, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True
    )
    return train_loader, val_loader


def make_eval_loader(df, cfg):
    from torch.utils.data import DataLoader

    ds = BUSIDataset(df, cfg.data.img_size, train=False,
                     mask_mode=cfg.data.mask_mode, roi_pad_ratio=cfg.data.roi_pad_ratio,
                     roi_normal_fallback=cfg.data.roi_normal_fallback,
                     use_gt_mask_oracle=cfg.data.use_gt_mask_oracle)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=cfg.data.num_workers)
