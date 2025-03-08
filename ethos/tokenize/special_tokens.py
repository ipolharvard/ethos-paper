from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ethos.tokenize.constants import TIME_DTYPE


class SpecialToken:
    SEPARATORS = {
        k: v / (60 * 24 * 365.25)
        for k, v in {
            "5m-15m": 5,
            "15m-1h": 15,
            "1h-2h": 60,
            "2h-6h": 2 * 60,
            "6h-12h": 6 * 60,
            "12h-1d": 12 * 60,
            "1d-3d": 24 * 60,
            "3d-1w": 3 * 24 * 60,
            "1w-2w": 7 * 24 * 60,
            "2w-1mt": 2 * 7 * 24 * 60,
            "1mt-3mt": 30 * 24 * 60,
            "3mt-6mt": 3 * 30 * 24 * 60,
            "=6mt": 6 * 30 * 24 * 60,
        }.items()
    }
    SEPARATOR_NAMES = list(SEPARATORS.keys())
    SEPARATOR_SIZES = np.fromiter(SEPARATORS.values(), dtype=TIME_DTYPE)
    TIMELINE_END = "TIMELINE_END"
    DEATH = "DEATH"

    # e.g., used for defining age of a person
    YEARS = {"<5": 5, **{f"{i - 5}-{i}y": i for i in range(10, 101, 5)}}
    YEAR_NAMES = list(YEARS.keys()) + [">100"]
    YEAR_BINS = list(YEARS.values())

    DECILES = [f"Q{i}" for i in range(1, 11)]

    ALL = [*SEPARATOR_NAMES, TIMELINE_END, DEATH, *YEAR_NAMES, *DECILES]

    @staticmethod
    def get_longest_separator() -> (str, float):
        return SpecialToken.SEPARATOR_NAMES[-1], SpecialToken.SEPARATOR_SIZES[-1]

    @staticmethod
    def age_to_year_token(age: float) -> str:
        i = np.digitize(age, SpecialToken.YEAR_BINS)
        return SpecialToken.YEAR_NAMES[i]

    @staticmethod
    def convert_to_deciles(
        values: pd.Series,
        bins: Optional[Iterable] = None,
        scheme: str = "quantiles",
        ret_bins: bool = False,
    ) -> pd.Series | tuple[pd.Series, np.ndarray[float]]:
        labels = SpecialToken.DECILES
        if bins is None:
            if scheme == "equidistant":
                bins = np.linspace(values.min(), values.max(), len(labels) + 1)
            elif scheme == "quantiles":
                bins = values.quantile(q=np.linspace(0, 1, len(labels) + 1))
            else:
                raise ValueError(
                    f"Got invalid `scheme` {scheme}, available: 'equidistant' and 'quantiles'."
                )
        bins = np.unique(bins)
        if np.isnan(bins[-1]):
            bins = bins[:-1]
        discrete_values = pd.cut(
            values,
            bins=bins,
            labels=labels[: len(bins) - 1],
            include_lowest=True,
        )
        # Don't make these values categorical, because later we might want to map nans into
        # something else, and doing it with categorical-valued series is cumbersome.
        discrete_values = discrete_values.astype(object)
        if ret_bins:
            return discrete_values, bins
        return discrete_values
