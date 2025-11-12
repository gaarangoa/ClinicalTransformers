import torch


def value_mask_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes MSE restricted to masked positions and averages over the batch.
    """

    diff = (preds - targets) ** 2
    masked = diff * mask
    denom = mask.sum(dim=1) + 1e-8
    per_sample = masked.sum(dim=1) / denom
    return per_sample.mean()
