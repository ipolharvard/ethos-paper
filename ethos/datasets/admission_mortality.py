import abc
from typing import Iterable

import numpy as np
import torch as th

from .base import InferenceDataset
from ..constants import ADMISSION_STOKEN, DISCHARGE_STOKEN
from ..tokenize import SpecialToken


class _AdmissionMortalityBase(InferenceDataset, abc.ABC):
    admission_indices: np.ndarray[np.int64]
    admission_toi_indices: np.ndarray[np.int64]
    toi_indices: np.ndarray[np.int64]

    def __init__(
        self, data, encode, block_size: int, admission_stoken: str, toi_stokens: Iterable[str]
    ):
        super().__init__(data, encode, block_size)
        admission_token = self.encode(admission_stoken)
        self.admission_indices = th.nonzero(self.tokens == admission_token).view(-1).numpy()
        tokens_of_interest = th.tensor(self.encode(toi_stokens).astype(int))
        self.toi_indices = th.nonzero(th.isin(self.tokens, tokens_of_interest)).view(-1).numpy()
        # when a subset of data is used, the cut-off is done arbitrarily, so we need to make sure
        # that the last admission token is not included if it doesn't have a corresponding toi
        if self.admission_indices[-1] > self.toi_indices[-1]:
            self.admission_indices = self.admission_indices[:-1]
        # precompute the corresponding toi indices for each admission
        self.admission_toi_indices = np.searchsorted(self.toi_indices, self.admission_indices)

    def __len__(self) -> int:
        return len(self.admission_indices)

    def _get_toi_idx(self, admission_idx):
        return self.toi_indices[self.admission_toi_indices[admission_idx]]

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        admission_idx = self.admission_indices[idx]
        patient_idx = self._get_patient_idx(admission_idx)
        data_start_idx = self.patient_offsets[patient_idx]

        if admission_idx - data_start_idx - 1 > self.timeline_len:
            data_start_idx = admission_idx + 1 - self.timeline_len

        patient_context = self._get_patient_context(data_start_idx)
        timeline = self.tokens[data_start_idx : admission_idx + 1]
        x = th.cat((patient_context, timeline))
        # get the idx of toi that corresponds to the admission
        toi_idx = self._get_toi_idx(idx)
        year = self._get_year_at_timeline_start(patient_idx, self.times[data_start_idx])
        return x, {
            "expected": self.tokens[toi_idx].item(),
            "true_token_dist": (toi_idx - admission_idx).item(),
            "true_token_time": (self.times[toi_idx] - self.times[admission_idx]).item(),
            "patient_id": self.patient_ids[patient_idx].item(),
            "patient_age": self.times[data_start_idx].item(),
            "admission_token_idx": admission_idx.item(),
            "year": year,
        }


class AdmissionMortalityDataset(_AdmissionMortalityBase):
    """Produces timelines that end on inpatient_admission_start token and go back in patients'
    history. The target is the inpatient_admission_end (discharge) or death token."""

    def __init__(self, data, encode, block_size: int):
        stokens_of_interest = [SpecialToken.DEATH, DISCHARGE_STOKEN]
        super().__init__(data, encode, block_size, ADMISSION_STOKEN, stokens_of_interest)


class AdmissionMortalityNextTokenDataset(_AdmissionMortalityBase):
    def __init__(self, data, encode, block_size: int):
        stokens_of_interest = [SpecialToken.DEATH, DISCHARGE_STOKEN]
        super().__init__(data, encode, block_size, ADMISSION_STOKEN, stokens_of_interest)

    def __getitem__(self, idx):
        admission_idx = self.admission_indices[idx]
        patient_idx = self._get_patient_idx(admission_idx)
        data_start_idx = self.patient_offsets[patient_idx]

        toi_idx = self._get_toi_idx(idx)
        if toi_idx - data_start_idx > self.timeline_len:
            data_start_idx = toi_idx - self.timeline_len

        patient_context = self._get_patient_context(data_start_idx)
        timeline = self.tokens[data_start_idx:toi_idx]
        x = th.cat((patient_context, timeline))
        return x, {
            "expected": self.tokens[toi_idx].item(),
            "patient_id": self.patient_ids[patient_idx].item(),
            "patient_age": self.times[data_start_idx].item(),
        }
