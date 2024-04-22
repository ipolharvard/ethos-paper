from bisect import bisect_left
from typing import Callable

import numpy as np
import torch as th

from .admission_mortality import _AdmissionMortalityBase
from .base import InferenceDataset
from ..constants import (
    ICU_ADMISSION_STOKEN,
    ICU_DISCHARGE_STOKEN,
    DISCHARGE_STOKEN,
    ADMISSION_STOKEN,
)
from ..tokenize import SpecialToken
from ..tokenize.mimic.special import ICUStayIdMixin


class DrgPredictionDataset(InferenceDataset):
    def __init__(self, data, encode: Callable, block_size: int):
        super().__init__(data, encode, block_size)
        discharge_token = self.encode(DISCHARGE_STOKEN)
        self.discharge_indices = th.nonzero(self.tokens == discharge_token).view(-1).numpy()
        # when a subset of data is used, the cut-off is done arbitrarily, so we need to make sure
        # that the last discharge token is not included if it doesn't have a corresponding dgr
        if self.discharge_indices[-1] + 4 > len(self.tokens):
            self.discharge_indices = self.discharge_indices[:-1]

    def __len__(self):
        return len(self.discharge_indices)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, dict]:
        drg_idx = self.discharge_indices[idx] + 3
        patient_idx = self._get_patient_idx(drg_idx)
        data_start_idx = self.patient_offsets[patient_idx]
        if drg_idx - data_start_idx > self.timeline_len:
            data_start_idx = drg_idx - self.timeline_len
        patient_context = self._get_patient_context(data_start_idx)
        timeline = self.tokens[data_start_idx:drg_idx]
        x = th.cat((patient_context, timeline))
        return x, {
            "expected": self.tokens[drg_idx].item(),
            "patient_id": self.patient_ids[patient_idx].item(),
            "patient_age": self.times[data_start_idx].item(),
            "drg_idx": drg_idx.item(),
        }


class SofaPredictionDataset(ICUStayIdMixin, InferenceDataset):
    def __init__(self, data, encode: Callable, block_size: int):
        super().__init__(
            data["patient_ids"],
            data=data,
            encode=encode,
            block_size=block_size,
        )
        admission_token = self.encode(ICU_ADMISSION_STOKEN)
        self.admission_indices = th.nonzero(self.tokens == admission_token).view(-1).numpy()
        # when a subset of data is used, the cut-off is done arbitrarily, so we need to make sure
        # that the last admission token is not included if it doesn't have a corresponding toi
        if self.admission_indices[-1] + 3 > len(self.tokens):
            self.admission_indices = self.admission_indices[:-1]

    def __len__(self):
        return len(self.admission_indices)

    def __getitem__(self, idx: int) -> tuple[th.Tensor, dict]:
        admission_idx = self.admission_indices[idx]
        patient_idx = self._get_patient_idx(admission_idx)
        data_start_idx = self.patient_offsets[patient_idx]
        # shorten the input timeline if the patient history is too long, -1 because we include
        # the admission token
        if admission_idx - data_start_idx - 1 > self.timeline_len:
            data_start_idx = admission_idx + 1 - self.timeline_len
        patient_context = self._get_patient_context(data_start_idx)
        timeline = self.tokens[data_start_idx : admission_idx + 1]
        x = th.cat((patient_context, timeline))
        return x, {
            "expected": self.tokens[admission_idx + 3].item(),
            "patient_id": self.patient_ids[patient_idx].item(),
            "patient_age": self.times[data_start_idx].item(),
            "admission_idx": admission_idx.item(),
            "stay_id": self._get_stay_id(idx).item(),
        }


class ICUMortalityDataset(ICUStayIdMixin, _AdmissionMortalityBase):
    TIME_OFFSET = 1 / 365.25  # one day (24h) in years

    def __init__(self, data, encode, block_size: int, use_time_offset: bool = True):
        stokens_of_interest = [SpecialToken.DEATH, ICU_DISCHARGE_STOKEN]
        super().__init__(
            data["patient_ids"],
            data=data,
            encode=encode,
            block_size=block_size,
            admission_stoken=ICU_ADMISSION_STOKEN,
            toi_stokens=stokens_of_interest,
        )
        if use_time_offset:
            # exclude admissions that are shorter than `TIME_OFFSET` years
            self._exclude_too_short_stays()

    def _exclude_too_short_stays(self):
        is_long_enough = np.fromiter(
            (
                (self.times[self._get_toi_idx(i)] - self.times[adm_idx]) > self.TIME_OFFSET
                for i, adm_idx in enumerate(self.admission_indices)
            ),
            dtype=bool,
            count=len(self.admission_indices),
        )
        self.admission_indices = self.admission_indices[is_long_enough]
        self.admission_toi_indices = self.admission_toi_indices[is_long_enough]
        self._stay_ids = self._stay_ids[is_long_enough]
        times = self.times.numpy()
        for i, adm_idx in enumerate(self.admission_indices):
            offset = bisect_left(
                times[adm_idx : self._get_next_timeline_start(adm_idx)],
                times[adm_idx] + self.TIME_OFFSET,
            )
            self.admission_indices[i] += offset - 1

    def _get_next_timeline_start(self, idx: int):
        return self.patient_offsets[self._get_patient_idx(idx) + 1]

    def __getitem__(self, idx: int):
        x, y = super().__getitem__(idx)
        y["stay_id"] = self._get_stay_id(idx).item()
        return x, y


class ICUReadmissionDataset(InferenceDataset):
    """
    To talk about ICU readmission, there has to be at least one ICU stay within an inpatient stay.
    Readmission:
     - Discard cases where patients are <18 years old at the time of the inpatient admission.
     - Discard cases where patients die during the first ICU stay.
     - Only ICU readmission after the first ICU within the same hospital stay are considered.
     - Positive cases are those who were admitted to the ICU again within the same inpatient stay.
    """

    def __init__(self, data, encode, block_size: int):
        super().__init__(data, encode, block_size)
        adm_indices = self._get_indices_of_stokens(ADMISSION_STOKEN)

        # discard cases where patients are <18 years old at the time of the inpatient admission
        at_least_18_year_old = (self.times[adm_indices] >= 18).numpy()

        # choose inpatient stays with at least one ICU stay
        icu_adm_indices = self._get_indices_of_stokens(ICU_ADMISSION_STOKEN)
        adm_icu_adm_indices = self._match_next_value(
            adm_indices, icu_adm_indices, always_match=False
        )
        icu_dc_indices = self._get_indices_of_stokens(ICU_DISCHARGE_STOKEN)
        adm_icu_dc_indices = self._match_next_value(adm_indices, icu_dc_indices, always_match=False)
        dc_or_end_indices = self._get_indices_of_stokens(
            [DISCHARGE_STOKEN, SpecialToken.TIMELINE_END]
        )
        adm_dc_or_end_indices = self._match_next_value(adm_indices, dc_or_end_indices)
        has_icu_stay = (adm_icu_dc_indices < adm_dc_or_end_indices) | (
            adm_icu_adm_indices < adm_dc_or_end_indices
        )

        # discard cases where a patient dies during the first ICU stay
        death_indices = self._get_indices_of_stokens(SpecialToken.DEATH)
        adm_death_indices = self._match_next_value(adm_indices, death_indices, always_match=False)
        dies_during_first_icu_stay = np.fromiter(
            (
                adm_death_idx < adm_icu_dc_idx < adm_dc_or_end_idx
                for adm_death_idx, adm_icu_dc_idx, adm_dc_or_end_idx in zip(
                    adm_death_indices, adm_icu_dc_indices, adm_dc_or_end_indices
                )
            ),
            dtype=bool,
            count=len(adm_indices),
        )

        # put all the conditions together
        adm_indices = adm_indices[at_least_18_year_old & has_icu_stay & ~dies_during_first_icu_stay]

        adm_icu_dc_indices = self._match_next_value(adm_indices, icu_dc_indices)

        # evaluate ground truth
        self.icu_dc_icu_adm_indices = self._match_next_value(
            adm_icu_dc_indices, icu_adm_indices, always_match=False
        )
        adm_dc_or_end_indices = self._match_next_value(adm_indices, dc_or_end_indices)

        # save the indices to use them in runtime
        self.is_readmitted = self.icu_dc_icu_adm_indices < adm_dc_or_end_indices
        self.icu_dc_indices = adm_icu_dc_indices

    def __len__(self) -> int:
        return len(self.icu_dc_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        icu_dc_idx = self.icu_dc_indices[idx]
        patient_idx = self._get_patient_idx(icu_dc_idx)
        data_start_idx = self.patient_offsets[patient_idx]

        if icu_dc_idx - data_start_idx - 1 > self.timeline_len:
            data_start_idx = icu_dc_idx + 1 - self.timeline_len

        patient_context = self._get_patient_context(data_start_idx)
        timeline = self.tokens[data_start_idx : icu_dc_idx + 1]
        x = th.cat((patient_context, timeline))

        if self.is_readmitted[idx]:
            icu_adm_idx = int(self.icu_dc_icu_adm_indices[idx])
            y = {
                "expected": 1,
                "true_token_dist": (icu_adm_idx - icu_dc_idx).item(),
                "true_token_time": (self.times[icu_adm_idx] - self.times[icu_dc_idx]).item(),
            }
        else:
            y = {"expected": 0}

        year = self._get_year_at_timeline_start(patient_idx, self.times[data_start_idx])
        y.update(
            {
                "patient_id": self.patient_ids[patient_idx].item(),
                "patient_age": self.times[data_start_idx].item(),
                "discharge_idx": icu_dc_idx.item(),
                "year": year,
            }
        )
        return x, y
