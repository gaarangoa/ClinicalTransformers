from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Iterable, List

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
