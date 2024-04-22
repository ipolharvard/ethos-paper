import abc
from bisect import bisect
from typing import Sequence, Callable

import numpy as np
import torch as th

from ..tokenize import SpecialToken


class TimelineDataset(th.utils.data.Dataset):
    BASE_YEAR = 1970

    def __init__(self, data: dict, encode: Callable, block_size: int = 2048):
        self.times: th.Tensor = data["times"]
        self.tokens: th.Tensor = data["tokens"]
        self.patient_context: th.Tensor = data["patient_context"]
        self.age_reference: Sequence[np.ndarray[np.float64]] = data["age_reference"]
        self.patient_offsets: np.ndarray[np.int64] = data["patient_data_offsets"]
        self.patient_ids: np.ndarray = data["patient_ids"]
        # if ids are bytes, convert them to string to be able to use .item() later
        if isinstance(self.patient_ids[0], bytes):
            self.patient_ids = self.patient_ids.astype("U")
        # block size is the max length of the full timeline (context + events)
        self.block_size: int = block_size
        # +1 - token of age, +1 - token of datetime, both computed in runtime
        self.context_len: int = self.patient_context.shape[1] + 1 + 1
        self.timeline_len: int = self.block_size - self.context_len
        # vocab encode function that translates strings to integers
        self.encode: Callable = encode

    def __len__(self) -> int:
        return len(self.times) - self.timeline_len

    def __getitem__(self, idx: int) -> tuple[th.Tensor, th.Tensor]:
        patient_context = self._get_patient_context(idx)
        timeline = self.tokens[idx : idx + self.timeline_len + 1]
        x = th.cat((patient_context, timeline[:-1]))
        y = th.cat((patient_context, timeline[1:]))
        return x, y

    def _get_patient_context(self, idx: int) -> th.Tensor:
        """Returns the patient context at the time given by the index."""
        patient_idx = self._get_patient_idx(idx)
        patient_context = self.patient_context[patient_idx]

        patient_age_at_timeline_start = self.times[idx]
        patient_age_token = self._years_to_token(patient_age_at_timeline_start)
        year = self._get_year_at_timeline_start(patient_idx, patient_age_at_timeline_start)
        anchor_year_token = self._years_to_token(year - self.BASE_YEAR)
        return th.cat((patient_context, th.tensor([patient_age_token, anchor_year_token])))

    def _get_patient_idx(self, idx: int) -> int:
        """Given the index in data, returns the patient's index (no.) in the patient data."""
        patient_idx = bisect(self.patient_offsets, idx)
        return patient_idx - 1

    def _years_to_token(self, years: float) -> int:
        return self.encode(SpecialToken.age_to_year_token(years))

    def _get_year_at_timeline_start(self, patient_idx: int, patient_age: float) -> float:
        year_of_birth = self.age_reference[patient_idx]
        year = year_of_birth + patient_age
        return year.item()


class InferenceDataset(TimelineDataset, abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        pass

    def _get_indices_of_stokens(self, stokens: str | Sequence[str]) -> np.ndarray[np.int64]:
        tokens = self.encode(stokens)
        if np.isnan(tokens).any():
            raise ValueError(f"Tokens for {stokens} could not be found in the vocabulary.")
        tokens = th.tensor(tokens)
        return th.nonzero(th.isin(self.tokens, tokens)).view(-1).numpy()

    @staticmethod
    def _match_next_value(
        to_match: Sequence, match_with: Sequence, always_match: bool = True
    ) -> np.ndarray[int | float]:
        """
        Return the next closest values in `match_with` for every corresponding value in `to_match`.
        Both sequences must be sorted in the ascending order.

        If `always_match` is True, the function will always try to assign a value in `match_with`,
        if it does not find it, it will raise out-of-bounds error.
        If `always_match` is False, the function will return `np.nan` for every value without the
        match.
        """
        match_with_indices = np.searchsorted(match_with, to_match)
        if always_match:
            return match_with[match_with_indices]
        else:
            matched_values = np.fromiter(
                (
                    match_with[match_with_idx] if match_with_idx < len(match_with) else np.nan
                    for match_with_idx in match_with_indices
                ),
                dtype=float,
                count=len(to_match),
            )
            if not np.isnan(matched_values[-1]):
                return matched_values.astype(int)
            return matched_values
