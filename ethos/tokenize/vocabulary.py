import pickle
from typing import Iterable, Optional, Hashable, Any

import numpy as np
import pandas as pd
from torch import Tensor

from .constants import TOKEN_DTYPE
from .special_tokens import SpecialToken


class QStorage:
    """Class to store quantiles and stokens for further processing using an existing vocab."""

    def __init__(self, storage: Optional[dict] = None):
        self._storage: dict[Hashable, Optional[np.ndarray]] = storage or {}

    def to_dict(self) -> dict:
        return self._storage

    def __bool__(self):
        return bool(self._storage)

    def values(self) -> set:
        # backward compatibility, the check is to be removed in the future
        return set(s for s in self._storage.keys() if isinstance(s, str))

    def register_values(self, records: Iterable[Hashable]):
        unique_values = np.unique(list(records))
        # nan is always put at the end, so we only check the last element
        if isinstance(unique_values[-1], float):
            unique_values = unique_values[:-1]
        self._storage.update(
            {record: None for record in unique_values if record not in self._storage}
        )

    def values_to_deciles(
        self, values: pd.Series, record_name: Hashable, scheme: str = "quantiles"
    ) -> pd.Series:
        bins = self._storage.get(record_name, None)
        if bins is not None:
            values.clip(bins[0], bins[-1], inplace=True)
        q_values, bins = SpecialToken.convert_to_deciles(values, bins, scheme, ret_bins=True)
        if record_name not in self._storage:
            self._storage[record_name] = bins
        return q_values


class Vocabulary:
    _stoi: dict[str, int]
    _itos: Optional[dict[int, str]]
    _q_storages: dict[Hashable, dict]
    _meta: dict[str, Any]

    def __init__(self, path=None):
        self._itos = None
        self._q_storages = {}
        self._meta = {}
        if path is not None:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._stoi = data["stoi"]
            self._itos = data["itos"]
            self._q_storages = data["q_storage"]
            self._meta = data["meta"]
        else:
            self._stoi = {t: i for i, t in enumerate(SpecialToken.ALL)}

    def tokenize(self, words: Iterable) -> np.ndarray[TOKEN_DTYPE]:
        new_words = np.asarray(np.copy(words))
        new_words = set(new_words[~pd.isna(new_words)])
        new_words = new_words.difference(self._stoi.keys())
        token_offset = len(self._stoi)
        new_stoi = dict(zip(new_words, np.arange(token_offset, token_offset + len(new_words))))
        self._stoi.update(new_stoi)
        if self._itos is not None:
            self._itos.update({v: k for k, v in new_stoi.items()})
        return self.encode(words)

    def encode(self, words: Iterable[str] | str) -> np.ndarray[np.float64] | np.float64:
        if isinstance(words, str):
            return self.stoi.get(words, np.nan)
        return np.fromiter(
            (self.stoi.get(w, np.nan) for w in words), count=len(words), dtype=np.float64
        )

    def decode(
        self, tokens: Iterable[TOKEN_DTYPE] | TOKEN_DTYPE
    ) -> np.ndarray[np.object_] | np.object_:
        if isinstance(tokens, Tensor):
            tokens = tokens.tolist()
        if not isinstance(tokens, Iterable):
            return self.itos.get(tokens, np.nan)
        return np.fromiter(
            (self.itos.get(t, np.nan) for t in tokens), count=len(tokens), dtype=np.object_
        )

    def __len__(self):
        return len(self.stoi)

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        if self._itos is None:
            self._itos = {v: k for k, v in self._stoi.items()}
        return self._itos

    @property
    def meta(self):
        return self._meta

    def add_meta(self, key, value):
        self._meta[key] = value

    def get_q_storage(self, name: Hashable) -> QStorage:
        return QStorage(self._q_storages.get(name))

    def add_q_storage(self, name: Hashable, q_storage: QStorage):
        self._q_storages[name] = q_storage.to_dict()

    def to_pickle(self, path):
        data = {
            "vocab_size": len(self._stoi),
            "stoi": self.stoi,
            "itos": self.itos,
            "q_storage": self._q_storages,
            "meta": self._meta,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def get_timeline_total_time(self, timeline: Iterable, decode=False) -> np.float64:
        sep_estimates = self.meta["separator_estimates"]["mean"]
        name, size = SpecialToken.get_longest_separator()
        sep_estimates[name] = size

        def get_sep_time(t):
            # not sure whether the second check should be implicit like this
            if decode and t < len(self):
                t = self.decode(t)
            return sep_estimates.get(t, np.nan)

        sep_times = np.vectorize(get_sep_time, otypes=[np.float64])(timeline)
        return np.nansum(sep_times)


class QStorageContext:
    def __init__(self, name: str, vocab: Optional[Vocabulary] = None):
        self.name = name
        self.vocab = vocab
        self.q_storage = None

    def __enter__(self) -> QStorage:
        if self.vocab is not None:
            self.q_storage = self.vocab.get_q_storage(self.name)
            return self.q_storage
        # if vocab is None, then create a dummy vocab, that will be discarded later
        return QStorage()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.vocab is not None:
            self.vocab.add_q_storage(self.name, self.q_storage)
