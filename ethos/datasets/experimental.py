from bisect import bisect

import numpy as np
import torch as th

from .base import TimelineDataset
from ..constants import DISCHARGE_STOKEN
from ..tokenize import SpecialToken


class ToiOversampleDataset(TimelineDataset):
    toi_indices: np.ndarray[np.int64]

    STOKENS_OF_INTEREST = [SpecialToken.DEATH, DISCHARGE_STOKEN]

    def __init__(self, data, encode, block_size=2048, p: float = 1):
        super().__init__(data, encode, block_size)
        tokens_of_interest = th.tensor(self.encode(self.STOKENS_OF_INTEREST).astype(int))
        toi_indices = th.nonzero(th.isin(self.tokens, tokens_of_interest)).numpy().reshape(-1)

        self.weights = th.ones(len(self))
        half_block_size = block_size // 2
        for idx in toi_indices:
            self.weights[idx - block_size : idx - half_block_size] = p

    def get_sampler(self, device="cpu"):
        self.weights.to(device)
        return self.Sampler(self.weights, num_samples=len(self), replacement=True)

    class Sampler(th.utils.data.WeightedRandomSampler):
        MAX_LEN = 2**24

        def __iter__(self):
            rand_tensor = th.empty(len(self.weights), dtype=th.long)
            for i in range(0, len(self.weights), self.MAX_LEN):
                sub_weights = self.weights[i : i + self.MAX_LEN]
                rand_sub_tensor = th.multinomial(
                    sub_weights, len(sub_weights), self.replacement, generator=self.generator
                )
                rand_tensor[i : i + self.MAX_LEN] = rand_sub_tensor

            yield from iter(rand_tensor.tolist())


class ToiShiftingDataset(TimelineDataset):
    toi_indices: np.ndarray[np.int64]

    STOKENS_OF_INTEREST = [SpecialToken.DEATH, DISCHARGE_STOKEN]

    def __init__(self, data, encode, block_size=2048):
        super().__init__(data, encode, block_size)
        tokens_of_interest = th.tensor(self.encode(self.STOKENS_OF_INTEREST).astype(int))
        self.toi_indices = th.nonzero(th.isin(self.tokens, tokens_of_interest)).numpy().reshape(-1)

    def __getitem__(self, idx):
        tmln_start_idx, tmln_end_idx = idx, idx + self.timeline_len

        toi_indices_idx = bisect(self.toi_indices, tmln_end_idx) - 1
        toi_idx = self.toi_indices[toi_indices_idx]

        if toi_idx >= tmln_start_idx and toi_idx - self.timeline_len >= 0:
            tmln_start_idx, tmln_end_idx = toi_idx - self.timeline_len, toi_idx

        timeline = self.tokens[tmln_start_idx : tmln_end_idx + 1]
        patient_context = self._get_patient_context(tmln_start_idx)
        x = th.cat((patient_context, timeline[:-1]))
        y = th.cat((patient_context, timeline[1:]))
        return x, y
