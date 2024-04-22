from bisect import bisect

import torch as th

from .base import InferenceDataset
from ..constants import ADMISSION_STOKEN, DISCHARGE_STOKEN
from ..tokenize import SpecialToken
from ..tokenize.mimic.special import HadmIdMixin


class ReadmissionDataset(HadmIdMixin, InferenceDataset):
    def __init__(self, data, encode, block_size, is_mimic: bool = False):
        self.is_mimic = is_mimic
        patient_ids = data["patient_ids"] if is_mimic else None
        super().__init__(patient_ids=patient_ids, data=data, encode=encode, block_size=block_size)

        self.adm_or_death_indices = self._get_indices_of_stokens(
            [ADMISSION_STOKEN, SpecialToken.DEATH]
        )
        self.discharge_indices = self._get_indices_of_stokens(DISCHARGE_STOKEN)
        if is_mimic:
            self.admission_indices = self._get_indices_of_stokens(ADMISSION_STOKEN)

    def __len__(self) -> int:
        return len(self.discharge_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        discharge_idx = self.discharge_indices[idx]
        patient_idx = self._get_patient_idx(discharge_idx)
        # find the next admission for the current discharge
        adm_or_death_indices_idx = bisect(self.adm_or_death_indices, discharge_idx)
        adm_or_death_idx = (
            self.adm_or_death_indices[adm_or_death_indices_idx]
            if adm_or_death_indices_idx != len(self.adm_or_death_indices)
            else -1
        )
        # check whether the next admission belongs to the same patient
        if adm_or_death_idx != -1 and self._get_patient_idx(adm_or_death_idx) == patient_idx:
            y = {
                "expected": 1,
                "true_token_dist": (adm_or_death_idx - discharge_idx).item(),
                "true_token_time": (
                    self.times[adm_or_death_idx] - self.times[discharge_idx]
                ).item(),
            }
        else:  # patient was not readmitted
            y = {"expected": 0}

        data_start_idx = self.patient_offsets[patient_idx]
        if discharge_idx - 1 - data_start_idx > self.timeline_len:
            data_start_idx = discharge_idx + 1 - self.timeline_len

        timeline = self.tokens[data_start_idx : discharge_idx + 1]
        patient_context = self._get_patient_context(data_start_idx)
        x = th.cat((patient_context, timeline))
        y.update(
            {
                "patient_id": self.patient_ids[patient_idx].item(),
                "patient_age": self.times[data_start_idx].item(),
                "discharge_token_idx": discharge_idx.item(),
            }
        )
        if self.is_mimic:
            admission_indices_idx = bisect(self.admission_indices, discharge_idx)
            y["hadm_id"] = self._get_hadm_id(admission_indices_idx - 1).item()
        return x, y
