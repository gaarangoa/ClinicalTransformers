from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

from ctv2.tokenizer import FeatureTokenizer


@dataclass
class ValueMaskedSample:
    values: torch.Tensor
    token_ids: torch.Tensor
    targets: torch.Tensor
    value_mask: torch.Tensor
    padding_mask: torch.Tensor


class ValueMaskedAnnDataDataset(Dataset):
    """
    Efficient dataset backed by AnnData that materializes only the rows requested
    by the PyTorch DataLoader.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        tokenizer: FeatureTokenizer,
        *,
        max_features: Optional[int] = None,
        mask_fraction: float = 0.2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.adata = adata
        self.tokenizer = tokenizer
        self.mask_fraction = float(mask_fraction)
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()

        self._feature_tokens = tokenizer.encode_many(adata.var_names.tolist())
        self._cls_token = tokenizer.cls_id
        self._pad_token = tokenizer.pad_id

        if self.max_features is None:
            self.seq_len = len(self._feature_tokens) + 1
        else:
            self.seq_len = self.max_features + 1

    def __len__(self) -> int:
        return self.adata.n_obs

    def _dense_row(self, idx: int) -> np.ndarray:
        row = self.adata.X[idx]
        if hasattr(row, "toarray"):
            row = row.toarray()
        return np.asarray(row).reshape(-1)

    def __getitem__(self, idx: int) -> ValueMaskedSample:
        row = self._dense_row(idx)
        feature_tokens = np.array(self._feature_tokens, dtype=np.int64)
        feature_values = np.array(row, dtype=np.float32)

        finite_mask = np.isfinite(feature_values)
        non_zero = np.abs(feature_values) > 0
        candidates = np.where(finite_mask & non_zero)[0]
        if candidates.size == 0:
            candidates = np.where(finite_mask)[0]

        if self.max_features is not None and candidates.size > self.max_features:
            candidates = self.rng.choice(
                candidates, size=self.max_features, replace=False
            )

        tokens = [self._cls_token]
        values = [1.0]
        for feat_idx in candidates:
            tokens.append(int(feature_tokens[feat_idx]))
            values.append(float(feature_values[feat_idx]))

        pad_len = self.seq_len - len(tokens)
        tokens.extend([self._pad_token] * pad_len)
        values.extend([0.0] * pad_len)

        tokens = np.array(tokens, dtype=np.int64)
        values = np.array(values, dtype=np.float32)
        targets = values.copy()

        mask = np.zeros_like(values)
        valid_positions = np.arange(1, len(tokens) - pad_len)
        if valid_positions.size > 0 and self.mask_fraction > 0:
            n_mask = max(1, int(valid_positions.size * self.mask_fraction))
            masked_idx = self.rng.choice(valid_positions, size=n_mask, replace=False)
            mask[masked_idx] = 1.0
            values[masked_idx] = 0.0

        padding_mask = (tokens == self._pad_token).astype(np.float32)

        return ValueMaskedSample(
            values=torch.from_numpy(values),
            token_ids=torch.from_numpy(tokens),
            targets=torch.from_numpy(targets),
            value_mask=torch.from_numpy(mask.astype(np.float32)),
            padding_mask=torch.from_numpy(padding_mask),
        )


def value_mask_collate(batch: list[ValueMaskedSample]) -> dict:
    stacked = {
        "values": torch.stack([sample.values for sample in batch]),
        "token_ids": torch.stack([sample.token_ids for sample in batch]),
        "targets": torch.stack([sample.targets for sample in batch]),
        "value_mask": torch.stack([sample.value_mask for sample in batch]),
        "padding_mask": torch.stack([sample.padding_mask for sample in batch]),
    }
    return stacked
