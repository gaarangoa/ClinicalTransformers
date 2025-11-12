from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import anndata as ad

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


@dataclass
class FeatureTokenizer(PreTrainedTokenizerFast):
    """
    HuggingFace-compatible tokenizer that treats feature names as individual tokens.
    """

    features: Iterable[str]
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"
    mask_token: str = "<mask>"
    unk_token: str = "<unk>"
    _feature_ids: OrderedDict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        uniques = list(dict.fromkeys(self.features))
        vocab = OrderedDict()
        for special in [self.pad_token, self.cls_token, self.mask_token, self.unk_token]:
            if special not in vocab:
                vocab[special] = len(vocab)
        for feat in uniques:
            if feat not in vocab:
                vocab[feat] = len(vocab)

        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=self.unk_token))

        super().__init__(
            tokenizer_object=tokenizer,
            pad_token=self.pad_token,
            cls_token=self.cls_token,
            mask_token=self.mask_token,
            unk_token=self.unk_token,
        )

        self._feature_ids = vocab
        self._id_to_feature = {idx: tok for tok, idx in vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._feature_ids)

    @property
    def pad_id(self) -> int:
        return int(self.convert_tokens_to_ids(self.pad_token))

    @property
    def cls_id(self) -> int:
        return int(self.convert_tokens_to_ids(self.cls_token))

    def encode_feature(self, token: str) -> int:
        try:
            return self._feature_ids[token]
        except KeyError:
            raise KeyError(f"Unknown feature '{token}'")

    def encode_many(self, tokens: Iterable[str]) -> List[int]:
        return [self.encode_feature(tok) for tok in tokens]

    def decode_id(self, idx: int) -> str:
        return self._id_to_feature[int(idx)]

    def decode_many(self, ids: Iterable[int]) -> List[str]:
        return [self.decode_id(i) for i in ids]


def _ensure_dataframe(data: Union[str, pd.DataFrame, Dict[str, Iterable]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, str):
        return pd.read_csv(data)
    if isinstance(data, dict):
        return pd.DataFrame(data)
    raise TypeError(f"Unsupported data type {type(data)}")


@dataclass
class FeatureValueTokenizer:
    categorical_features: Sequence[str]
    continuous_features: Sequence[str]
    binary_features: Sequence[str] = ()
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"
    mask_token: str = "<mask>"
    unk_token: str = "<unk>"
    max_seq_len: Optional[int] = None

    cat_maps: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False)
    cat_denoms: Dict[str, float] = field(default_factory=dict, init=False)
    cont_stats: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    feature_order: List[str] = field(default_factory=list, init=False)
    tokenizer: Optional[FeatureTokenizer] = field(default=None, init=False)

    def fit(self, data: Union[str, pd.DataFrame, Dict[str, Iterable]]) -> "FeatureValueTokenizer":
        df = _ensure_dataframe(data)
        self.feature_order = list(self.categorical_features) + list(self.binary_features) + list(self.continuous_features)

        for col in self.categorical_features:
            series = df[col].dropna().astype(str)
            levels = sorted(series.unique())
            mapping = {lvl: idx for idx, lvl in enumerate(levels)}
            mapping["__UNK__"] = len(mapping)
            self.cat_maps[col] = mapping
            self.cat_denoms[col] = max(1.0, float(len(mapping) - 1)) or 1.0

        for col in self.continuous_features:
            series = df[col].astype(float)
            min_v = float(series.min())
            max_v = float(series.max())
            if min_v == max_v:
                max_v = min_v + 1.0
            self.cont_stats[col] = {"min": min_v, "max": max_v, "span": max_v - min_v}

        self.tokenizer = FeatureTokenizer(
            features=self.feature_order,
            pad_token=self.pad_token,
            cls_token=self.cls_token,
            mask_token=self.mask_token,
            unk_token=self.unk_token,
        )
        if self.max_seq_len is None:
            self.max_seq_len = len(self.feature_order) + 1
        return self

    def _encode_categorical(self, col: str, value) -> float:
        if pd.isna(value):
            raise ValueError
        mapping = self.cat_maps[col]
        idx = mapping.get(str(value), mapping["__UNK__"])
        denom = self.cat_denoms[col]
        return float(idx / denom if denom else 0.0)

    def _encode_binary(self, value) -> float:
        if pd.isna(value):
            raise ValueError
        return float(value)

    def _encode_continuous(self, col: str, value) -> float:
        if pd.isna(value):
            raise ValueError
        stats = self.cont_stats[col]
        span = stats["span"]
        scaled = (float(value) - stats["min"]) / span
        return float(np.clip(scaled, 0.0, 1.0))

    def encode(
        self,
        data: Union[str, pd.DataFrame, Dict[str, Iterable]],
        *,
        return_tensors: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be fitted before calling encode().")

        df = _ensure_dataframe(data)
        pad_id = self.tokenizer.pad_id
        cls_id = self.tokenizer.cls_id
        seq_len = self.max_seq_len or (len(self.feature_order) + 1)

        token_rows: List[List[int]] = []
        value_rows: List[List[float]] = []

        for _, row in df.iterrows():
            row_tokens = [cls_id]
            row_values = [1.0]
            for feat in self.feature_order:
                try:
                    if feat in self.categorical_features:
                        val = self._encode_categorical(feat, row[feat])
                    elif feat in self.binary_features:
                        val = self._encode_binary(row[feat])
                    else:
                        val = self._encode_continuous(feat, row[feat])
                except ValueError:
                    continue

                row_tokens.append(self.tokenizer.encode_feature(feat))
                row_values.append(val)

                if len(row_tokens) >= seq_len:
                    break

            pad_needed = seq_len - len(row_tokens)
            if pad_needed > 0:
                row_tokens.extend([pad_id] * pad_needed)
                row_values.extend([0.0] * pad_needed)

            token_rows.append(row_tokens[:seq_len])
            value_rows.append(row_values[:seq_len])

        token_tensor = torch.tensor(token_rows, dtype=torch.long) if return_tensors else np.array(token_rows, dtype=np.int64)
        value_tensor = torch.tensor(value_rows, dtype=torch.float32) if return_tensors else np.array(value_rows, dtype=np.float32)
        pad_mask = (token_tensor == pad_id).float() if return_tensors else (token_tensor == pad_id).astype(np.float32)

        return token_tensor, value_tensor, pad_mask

    def to_anndata(self, data: Union[str, pd.DataFrame, Dict[str, Iterable]]) -> ad.AnnData:
        df = _ensure_dataframe(data)
        columns = []
        for feat in self.feature_order:
            col = df[feat]
            encoded = np.zeros(len(df), dtype=np.float32)
            for i, val in enumerate(col.values):
                try:
                    if feat in self.categorical_features:
                        encoded[i] = self._encode_categorical(feat, val)
                    elif feat in self.binary_features:
                        encoded[i] = self._encode_binary(val)
                    else:
                        encoded[i] = self._encode_continuous(feat, val)
                except ValueError:
                    encoded[i] = 0.0
            columns.append(encoded)
        matrix = np.stack(columns, axis=1)
        adata = ad.AnnData(X=matrix)
        adata.var = pd.DataFrame(index=self.feature_order)
        adata.obs = df.index.to_frame()
        return adata

    def save_pretrained(self, save_directory: Union[str, os.PathLike]) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be fitted before saving.")

        os.makedirs(save_directory, exist_ok=True)
        config = {
            "categorical_features": list(self.categorical_features),
            "binary_features": list(self.binary_features),
            "continuous_features": list(self.continuous_features),
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "mask_token": self.mask_token,
            "unk_token": self.unk_token,
            "max_seq_len": self.max_seq_len,
            "cat_maps": self.cat_maps,
            "cat_denoms": self.cat_denoms,
            "cont_stats": self.cont_stats,
            "feature_order": self.feature_order,
        }
        config_path = os.path.join(save_directory, "feature_value_tokenizer.json")
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(config, fp)

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, os.PathLike]) -> "FeatureValueTokenizer":
        config_path = os.path.join(load_directory, "feature_value_tokenizer.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No saved tokenizer found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as fp:
            config = json.load(fp)

        tokenizer = cls(
            categorical_features=config["categorical_features"],
            binary_features=config["binary_features"],
            continuous_features=config["continuous_features"],
            pad_token=config["pad_token"],
            cls_token=config["cls_token"],
            mask_token=config["mask_token"],
            unk_token=config["unk_token"],
            max_seq_len=config.get("max_seq_len"),
        )
        tokenizer.cat_maps = config["cat_maps"]
        tokenizer.cat_denoms = config["cat_denoms"]
        tokenizer.cont_stats = config["cont_stats"]
        tokenizer.feature_order = config["feature_order"]
        tokenizer.tokenizer = FeatureTokenizer(
            features=tokenizer.feature_order,
            pad_token=tokenizer.pad_token,
            cls_token=tokenizer.cls_token,
            mask_token=tokenizer.mask_token,
            unk_token=tokenizer.unk_token,
        )
        return tokenizer
