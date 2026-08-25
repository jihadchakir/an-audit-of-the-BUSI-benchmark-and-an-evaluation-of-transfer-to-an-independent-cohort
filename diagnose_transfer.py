from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import Config
from model import ResNetFPNEncoder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


# ---------------------------------------------------------------------------
# preprocessing variants, each maps a uint8 grayscale image -> normalised CHW
# ---------------------------------------------------------------------------
def _to_tensor(gray_u8: np.ndarray) -> np.ndarray:
    x = gray_u8.astype(np.float32) / 255.0
    x = np.stack([x, x, x], 0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def pp_plain(gray: np.ndarray, size: int) -> np.ndarray:
    """Exactly what training used: squash-resize to a square, ImageNet norm."""
    return _to_tensor(cv2.resize(gray, (size, size)))


def pp_letterbox(gray: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize with padding. Fixes the portrait/landscape squash."""
    h, w = gray.shape
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(gray, (nw, nh))
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return _to_tensor(canvas)


def pp_clahe(gray: np.ndarray, size: int) -> np.ndarray:
    """Squash-resize, then CLAHE. Fixes brightness/contrast mismatch."""
    g = cv2.resize(gray, (size, size))
    return _to_tensor(cv2.createCLAHE(2.0, (8, 8)).apply(g))


def pp_clahe_letterbox(gray: np.ndarray, size: int) -> np.ndarray:
    """Both fixes together."""
    h, w = gray.shape
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(gray, (nw, nh))
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return _to_tensor(cv2.createCLAHE(2.0, (8, 8)).apply(canvas))


VARIANTS: Dict[str, Callable] = {
    "plain_resize (as trained)": pp_plain,
    "letterbox (fix aspect)": pp_letterbox,
    "clahe (fix contrast)": pp_clahe,
    "clahe+letterbox (both)": pp_clahe_letterbox,
}


# ---------------------------------------------------------------------------
def load_encoder(fold_dir: Path, device: str) -> Tuple[torch.nn.Module, Config]:
    cfg_path = fold_dir.parent / "config.json"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config()
    ck = torch.load(fold_dir / "encoder_final.pt", map_location=device, weights_only=False)
    ec = ck["cfg"]
    model = ResNetFPNEncoder(
        embedding_dim=ec["embedding_dim"], fpn_channels=ec["fpn_channels"],
        pretrained=False, dropout=ec["dropout"], freeze_bn=ec.get("freeze_bn", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def embed(model, paths: List[str], pp: Callable, size: int, device: str, bs: int = 32) -> np.ndarray:
    out = []
    batch = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            out.append(None); continue
        batch.append((len(out), pp(g, size)))
        out.append("pending")
        if len(batch) == bs:
            xs = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
            z = model(xs).float().cpu().numpy()
            for (idx, _), e in zip(batch, z):
                out[idx] = e
            batch = []
    if batch:
        xs = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        z = model(xs).float().cpu().numpy()
        for (idx, _), e in zip(batch, z):
            out[idx] = e
    return np.stack([o for o in out if o is not None and not isinstance(o, str)])


def busbra_paths_labels(images_dir: str, csv: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    meta = pd.read_csv(csv)
    imgs = {p.stem: str(p) for p in Path(images_dir).rglob("*.png") if "_mask" not in p.stem}
    paths, y, dev = [], [], []
    for _, r in meta.iterrows():
        key = str(r["ID"]).strip()
        if key in imgs:
            paths.append(imgs[key])
            y.append(1 if str(r["Pathology"]).strip().lower() == "malignant" else 0)
            dev.append(str(r.get("Device", "?")))
    return paths, np.array(y), np.array(dev)


def prototype_auroc(E_ref: np.ndarray, y_ref: np.ndarray,
                    E_ext: np.ndarray, y_ext: np.ndarray) -> float:
    """Malignant AUROC using a nearest-prototype score, prototypes from the
    reference (BUSI train) embeddings. Encoder-only, no fitted head, so this
    isolates the embedding geometry from any head-transfer effect."""
    protos = np.stack([E_ref[y_ref == c].mean(0) for c in (0, 1)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12
    d = np.linalg.norm(E_ext[:, None, :] - protos[None, :, :], axis=2)
    score = d[:, 0] - d[:, 1]  # higher => closer to malignant prototype
    return roc_auc_score(y_ext, score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--busi-index", required=True)
    ap.add_argument("--busbra-images", required=True)
    ap.add_argument("--busbra-csv", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="./runs/external/transfer_diagnosis.json")
    args = ap.parse_args()

    model, cfg = load_encoder(Path(args.fold_dir), args.device)
    size = cfg.data.img_size

    busi = pd.read_csv(args.busi_index)
    busi = busi[busi["class_name"].isin(["benign", "malignant"])].reset_index(drop=True)
    busi_paths = busi["path"].tolist()
    busi_y = (busi["class_name"] == "malignant").astype(int).to_numpy()

    ext_paths, ext_y, ext_dev = busbra_paths_labels(args.busbra_images, args.busbra_csv)
    print(f"BUSI (ref): {len(busi_paths)} images | BUS-BRA (ext): {len(ext_paths)} images")
    print(f"Encoder: {args.fold_dir}, img_size={size}\n")
    print("Malignant AUROC (threshold-free). BUSI-internal is the ceiling for this")
    print("encoder; BUS-BRA columns show what each preprocessing fix recovers.\n")

    print(f"  {'preprocessing':28s} {'BUSI (ref)':>12s} {'BUS-BRA':>10s} {'GE only':>10s}")
    print("  " + "-" * 64)

    results: Dict[str, Dict] = {}
    for name, pp in VARIANTS.items():
        E_busi = embed(model, busi_paths, pp, size, args.device)
        E_ext = embed(model, ext_paths, pp, size, args.device)

        # BUSI-internal AUROC via 5-fold prototype (rough ceiling for this pp)
        from sklearn.model_selection import StratifiedKFold
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(E_busi, busi_y):
            aucs.append(prototype_auroc(E_busi[tr], busi_y[tr], E_busi[te], busi_y[te]))
        busi_auc = float(np.mean(aucs))

        ext_auc = prototype_auroc(E_busi, busi_y, E_ext, ext_y)
        ge = ext_dev == ext_dev[[i for i, d in enumerate(ext_dev) if "GE" in d or "Logiq" in d][0]] \
            if any("GE" in d or "Logiq" in d for d in ext_dev) else None
        ge_mask = np.array(["GE" in d or "Logiq" in d for d in ext_dev])
        ge_auc = (prototype_auroc(E_busi, busi_y, E_ext[ge_mask], ext_y[ge_mask])
                  if ge_mask.sum() > 20 else float("nan"))

        results[name] = {"busi": busi_auc, "busbra": ext_auc, "ge_only": ge_auc}
        print(f"  {name:28s} {busi_auc:11.3f} {ext_auc:10.3f} {ge_auc:10.3f}")

    base = results["plain_resize (as trained)"]["busbra"]
    best = max(results, key=lambda k: results[k]["busbra"])
    gain = results[best]["busbra"] - base

    print("\n" + "=" * 66)
    print("READING")
    print("=" * 66)
    print(f"""
  plain-resize BUS-BRA AUROC : {base:.3f}   (what the full run reported)
  best variant               : {best}
  best BUS-BRA AUROC         : {results[best]['busbra']:.3f}   ({gain:+.3f})

  This is a LOWER BOUND on what a matched-preprocessing retrain recovers, because
  the encoder was trained on plain-resize BUSI and these variants are themselves
  slightly out-of-distribution for it.""")
    if gain > 0.06:
        print(f"""
  A large inference-time gain ({gain:+.3f}) means low-level preprocessing, not
  anatomy, drove most of the drop. Retrain all five encoders with '{best}'
  applied identically to BUSI and BUS-BRA, then re-run external_eval. Expect the
  honest transfer number to land meaningfully above {base:.3f}.""")
    else:
        print(f"""
  Small inference-time gain ({gain:+.3f}). Preprocessing is not the main driver;
  the drop is more likely genuine domain shift (scanner, population, acquisition).
  A retrain may help a little but will not rescue this. Report the transfer
  number honestly as a limitation, and lean on the per-Device breakdown.""")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
