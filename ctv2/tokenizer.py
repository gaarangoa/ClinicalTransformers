from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class FeatureTokenizer:
    """
    Simple feature-name tokenizer that mirrors the functionality of the
    TensorFlow implementation but is framework agnostic.
    """

    features: Iterable[str]
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"

    def __post_init__(self) -> None:
        unique_features = list(dict.fromkeys(self.features))
        self._vocabulary: List[str] = [self.pad_token, self.cls_token] + unique_features
        self._encoder: Dict[str, int] = {tok: idx for idx, tok in enumerate(self._vocabulary)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def pad_id(self) -> int:
        return self._encoder[self.pad_token]

    @property
    def cls_id(self) -> int:
        return self._encoder[self.cls_token]

    def encode(self, token: str) -> int:
        if token not in self._encoder:
            raise KeyError(f"Unknown token '{token}'")
        return self._encoder[token]

    def decode(self, idx: int) -> str:
        return self._vocabulary[idx]

    def encode_many(self, tokens: Iterable[str]) -> List[int]:
        return [self.encode(tok) for tok in tokens]
