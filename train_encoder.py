from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from data import make_eval_loader, make_loaders
from losses import BatchHardTripletLoss, batch_hard_stats
from model import ResNetFPNEncoder, extract_embeddings


def _amp_dtype(name: str) -> Optional[torch.dtype]:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[name]


def _cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def prototype_probe(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: str,
) -> Tuple[float, float]:
    """Nearest-prototype balanced accuracy on the validation split."""
    E_tr, y_tr = extract_embeddings(model, train_loader, device)
    E_va, y_va = extract_embeddings(model, val_loader, device)

    n_classes = int(max(y_tr.max(), y_va.max())) + 1
    protos = np.stack([E_tr[y_tr == c].mean(0) for c in range(n_classes)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12
    d = np.linalg.norm(E_va[:, None, :] - protos[None, :, :], axis=2)
    y_pred = d.argmin(1)
    return float(balanced_accuracy_score(y_va, y_pred)), float((y_pred == y_va).mean())


def train_encoder(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cfg,
    device: str = "cuda",
    out_dir: str | Path = "./runs/fold",
    seed: int = 42,
) -> Tuple[torch.nn.Module, Dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader = make_loaders(df_train, df_val, cfg, seed=seed)
    probe_train_loader = make_eval_loader(df_train, cfg)  # no augmentation

    model = ResNetFPNEncoder(
        embedding_dim=cfg.encoder.embedding_dim,
        fpn_channels=cfg.encoder.fpn_channels,
        pretrained=cfg.encoder.pretrained,
        dropout=cfg.encoder.dropout,
        freeze_bn=cfg.encoder.freeze_bn,
    ).to(device)

    criterion = BatchHardTripletLoss(cfg.train.margin, cfg.train.mining)
    opt = torch.optim.AdamW(
        model.param_groups(cfg.train.lr, cfg.train.backbone_lr_mult),
        weight_decay=cfg.train.weight_decay,
    )
    amp = _amp_dtype(cfg.train.amp_dtype)
    scaler = torch.amp.GradScaler(device, enabled=(amp == torch.float16))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = steps_per_epoch * cfg.train.warmup_epochs
    base_lrs = [g["lr"] for g in opt.param_groups]

    # An ablation run deliberately reintroduces the defects, so a dead loss there
    # is the finding rather than a fault. Detect it so the log does not hand out
    # advice that would destroy the measurement.
    is_ablation = bool(getattr(cfg.data, "use_gt_mask_oracle", False)
                       or getattr(cfg, "splitting", "group") == "random"
                       or getattr(cfg.decision, "select_on", "val") == "test")

    history = {"train_loss": [], "val_balanced_accuracy": [], "val_accuracy": [],
               "frac_active_triplets": [], "lr": [], "saturation_epoch": None}
    best_score, best_epoch, bad_epochs = -1.0, -1, 0
    ckpt = out_dir / "encoder_best.pt"
    step = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        losses, actives = [], []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            f = _cosine_warmup(step, total_steps, warmup_steps)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * f

            opt.zero_grad(set_to_none=True)
            if amp is not None:
                with torch.autocast(device_type=device, dtype=amp):
                    z = model(x)
                    loss = criterion(z, y)
            else:
                z = model(x)
                loss = criterion(z, y)

            if amp == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()

            losses.append(loss.item())
            s = batch_hard_stats(z.detach().float(), y)
            if s:
                actives.append(s["frac_active_triplets"])
            step += 1

        val_bacc, val_acc = prototype_probe(model, probe_train_loader, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_balanced_accuracy"].append(val_bacc)
        history["val_accuracy"].append(val_acc)
        history["frac_active_triplets"].append(float(np.mean(actives)) if actives else 0.0)
        history["lr"].append(opt.param_groups[-1]["lr"])

        improved = val_bacc > best_score + 1e-4
        if improved:
            best_score, best_epoch, bad_epochs = val_bacc, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_bacc": val_bacc}, ckpt)
        else:
            bad_epochs += 1

        print(
            f"  epoch {epoch+1:3d}/{cfg.train.epochs}  loss={np.mean(losses):.4f}  "
            f"val_bacc={val_bacc:.4f}  active={history['frac_active_triplets'][-1]:.2f}  "
            f"{'*' if improved else ' '}  ({time.time()-t0:.0f}s)"
        )

        if history["frac_active_triplets"][-1] < 0.1 and history["saturation_epoch"] is None:
            history["saturation_epoch"] = epoch + 1

        if history["frac_active_triplets"][-1] < 0.01 and epoch > cfg.train.warmup_epochs:
            if is_ablation:
                print(f"  [ablation] triplet loss saturated at epoch "
                      f"{history['saturation_epoch']}: the reintroduced defects made the task "
                      f"trivial, so almost no triplet violates the margin. This is the "
                      f"measurement, NOT a problem. Do not tune it away.")
            else:
                print("  [warn] no active triplets left: the loss has saturated. "
                      "Raise the margin or the batch size if this happens early.")

        if bad_epochs >= cfg.train.patience:
            print(f"  early stop at epoch {epoch+1} (best epoch {best_epoch+1}, val_bacc {best_score:.4f})")
            break

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    history["best_epoch"] = int(best_epoch + 1)
    history["best_val_balanced_accuracy"] = float(best_score)
    history["final_frac_active_triplets"] = float(history["frac_active_triplets"][-1])
    history["min_frac_active_triplets"] = float(min(history["frac_active_triplets"]))
    history["is_ablation"] = is_ablation
    (out_dir / "encoder_history.json").write_text(json.dumps(history, indent=2))
    return model, history
