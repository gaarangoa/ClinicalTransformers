from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import anndata as ad


@dataclass
class TabularTokenizer:
    """
    Preprocesses mixed tabular data (categorical / binary / continuous) into
    numeric matrices suitable for the ctv2 pipeline. Works similarly to
    scikit-learn transformers but stores feature metadata for reconstruction.
    """

    categorical_features: Sequence[str] = ()
    binary_features: Sequence[str] = ()
    continuous_features: Sequence[str] = ()
    drop_na: bool = True

    _feature_names: List[str] = field(default_factory=list, init=False)
    _cat_levels: Dict[str, List[str]] = field(default_factory=dict, init=False)
    _continuous_stats: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)

    def fit(self, df: pd.DataFrame) -> "TabularTokenizer":
        df = df.copy()
        if self.drop_na:
            df = df.dropna(subset=self.continuous_features + self.categorical_features)

        for col in self.categorical_features:
            levels = df[col].astype("category").cat.categories.tolist()
            self._cat_levels[col] = levels
            for level in levels:
                self._feature_names.append(f"{col}::{level}")

        for col in self.binary_features:
            self._feature_names.append(col)

        for col in self.continuous_features:
            series = df[col].astype(float)
            stats = {
                "mean": float(series.mean()),
                "std": float(series.std() or 1.0),
            }
            self._continuous_stats[col] = stats
            self._feature_names.append(col)

        return self

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def transform(self, df: pd.DataFrame) -> ad.AnnData:
        if not self._feature_names:
            raise RuntimeError("Tokenizer must be fitted before calling transform()")

        columns = []

        for col, levels in self._cat_levels.items():
            cat = pd.Categorical(df[col], categories=levels)
            encoded = pd.get_dummies(cat, prefix=col)
            columns.append(encoded)

        if self.binary_features:
            columns.append(df[self.binary_features].astype(float))

        for col, stats in self._continuous_stats.items():
            normalized = (df[col].astype(float) - stats["mean"]) / stats["std"]
            columns.append(normalized.to_frame())

        matrix = pd.concat(columns, axis=1)
        matrix = matrix.reindex(columns=self._feature_names, fill_value=0.0)

        adata = ad.AnnData(X=matrix.values.astype(np.float32))
        adata.var = pd.DataFrame(index=self._feature_names)
        adata.obs = df.index.to_frame()
        return adata
