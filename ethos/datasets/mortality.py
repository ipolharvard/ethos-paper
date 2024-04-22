from bisect import bisect

import torch as th

from .base import InferenceDataset
from ..constants import ADMISSION_STOKEN, DISCHARGE_STOKEN
from ..tokenize import SpecialToken


class MortalityDataset(InferenceDataset):
    def __init__(self, data, encode, block_size):
        super().__init__(data, encode, block_size)

        death_token = self.encode(SpecialToken.DEATH)
        self.death_indices = th.nonzero(self.tokens == death_token).numpy().reshape(-1)

        timeline_end_token = self.encode(SpecialToken.TIMELINE_END)
        self.timeline_end_indices = (
            th.nonzero(self.tokens == timeline_end_token).numpy().reshape(-1)
        )

    def __len__(self) -> int:
        return len(self.timeline_end_indices)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        timeline_end_idx = self.timeline_end_indices[idx]
        patient_idx = self._get_patient_idx(timeline_end_idx)
        data_end_idx = timeline_end_idx
        expected = 0

        # we don't have death tokens in the QualityCheck fold
        if len(self.death_indices):
            death_indices_idx = bisect(self.death_indices, timeline_end_idx)
            death_idx = self.death_indices[death_indices_idx - 1]
            if self._get_patient_idx(death_idx) == patient_idx:
                data_end_idx = death_idx
                expected = 1

        data_start_idx = self.patient_offsets[patient_idx]
        if data_end_idx - data_start_idx > self.timeline_len:
            data_start_idx = data_end_idx - self.timeline_len

        timeline = self.tokens[data_start_idx:data_end_idx]
        patient_context = self._get_patient_context(data_start_idx)
        x = th.cat((patient_context, timeline))

        return x, {
            "expected": expected,
            "patient_id": self.patient_ids[patient_idx].item(),
            "init_timeline_len": len(x),
            "data_end_token": data_end_idx.item(),
        }


class SingleAdmissionMortalityDataset(InferenceDataset):
    STOKENS_OF_INTEREST = [SpecialToken.DEATH, DISCHARGE_STOKEN]

    def __init__(self, admission_idx, data, encode, block_size, num_reps=100):
        super().__init__(data, encode, block_size)

        assert self.tokens[admission_idx] == encode(
            ADMISSION_STOKEN
        ), "admission_idx must point at an admission start token"
        self.admission_idx = admission_idx
        self.num_reps = num_reps

        tokens_of_interest = th.tensor(self.encode(self.STOKENS_OF_INTEREST).astype(int))
        toi_indices = th.nonzero(th.isin(self.tokens, tokens_of_interest)).numpy().reshape(-1)

        toi_indices_idx = bisect(toi_indices, admission_idx)
        self.toi_idx = toi_indices[toi_indices_idx]

    def __len__(self) -> int:
        return int((self.toi_idx - self.admission_idx) * self.num_reps)

    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        idx = idx % (self.toi_idx - self.admission_idx)
        patient_idx = self._get_patient_idx(self.admission_idx)
        data_start_idx = self.patient_offsets[patient_idx]
        data_end_idx = self.admission_idx + idx
        if data_end_idx - 1 - data_start_idx > self.timeline_len:
            data_start_idx = data_end_idx + 1 - self.timeline_len

        timeline = self.tokens[data_start_idx : data_end_idx + 1]
        patient_context = self._get_patient_context(data_start_idx)
        x = th.cat((patient_context, timeline))

        return x, {
            "expected": self.tokens[self.toi_idx].item(),
            "offset": idx.item(),
            "init_timeline_len": len(x),
            "true_token_time": (self.times[data_end_idx] - self.times[self.admission_idx]).item(),
        }
