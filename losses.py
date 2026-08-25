from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distances(x: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Numerically safe Euclidean distance matrix."""
    dot = x @ x.t()
    sq = torch.diagonal(dot)
    d2 = sq.unsqueeze(0) - 2.0 * dot + sq.unsqueeze(1)
    d2 = d2.clamp(min=0.0)
    if squared:
        return d2
    # subgradient of sqrt at 0 is undefined; mask, sqrt, restore
    mask = (d2 == 0).float()
    d = torch.sqrt(d2 + mask * 1e-16)
    return d * (1.0 - mask)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3, mining: str = "batch_hard"):
        super().__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        d = pairwise_distances(embeddings)
        n = labels.size(0)

        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=labels.device)
        pos_mask = same & ~eye
        neg_mask = ~same

        if not pos_mask.any() or not neg_mask.any():
            return embeddings.sum() * 0.0

        if self.mining == "batch_hard":
            hardest_pos = (d * pos_mask.float()).max(dim=1).values
            d_neg = d.clone()
            d_neg[~neg_mask] = float("inf")
            hardest_neg = d_neg.min(dim=1).values
            valid = torch.isfinite(hardest_neg) & pos_mask.any(dim=1)
            loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
            return loss.mean() if loss.numel() else embeddings.sum() * 0.0

        if self.mining == "soft_margin":
            hardest_pos = (d * pos_mask.float()).max(dim=1).values
            d_neg = d.clone()
            d_neg[~neg_mask] = float("inf")
            hardest_neg = d_neg.min(dim=1).values
            valid = torch.isfinite(hardest_neg)
            return F.softplus(hardest_pos[valid] - hardest_neg[valid]).mean()

        if self.mining == "batch_semihard":
            # semi-hard: negatives further than the positive but within margin
            loss_terms = []
            for i in range(n):
                pos = d[i][pos_mask[i]]
                neg = d[i][neg_mask[i]]
                if pos.numel() == 0 or neg.numel() == 0:
                    continue
                ap = pos.max()
                semi = neg[(neg > ap) & (neg < ap + self.margin)]
                an = semi.min() if semi.numel() else neg.min()
                loss_terms.append(F.relu(ap - an + self.margin))
            if not loss_terms:
                return embeddings.sum() * 0.0
            return torch.stack(loss_terms).mean()

        raise ValueError(f"Unknown mining strategy '{self.mining}'")


@torch.no_grad()
def batch_hard_stats(embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
    """Diagnostics worth logging: if `frac_active` goes to 0 the loss is dead
    and any model selection based on it is meaningless."""
    d = pairwise_distances(embeddings)
    n = labels.size(0)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    pos_mask, neg_mask = same & ~eye, ~same
    if not pos_mask.any() or not neg_mask.any():
        return {}
    hp = (d * pos_mask.float()).max(dim=1).values
    dn = d.clone()
    dn[~neg_mask] = float("inf")
    hn = dn.min(dim=1).values
    active = (hp - hn + 0.3 > 0).float().mean().item()
    return {
        "mean_hardest_pos": hp.mean().item(),
        "mean_hardest_neg": hn.mean().item(),
        "frac_active_triplets": active,
    }
