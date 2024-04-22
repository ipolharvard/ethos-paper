import pandas as pd

from ..translation_base import Translation
from ...constants import PROJECT_DATA


class AtcTranslation(Translation):
    warn_exclude = ["Not a medication", "Not specified", "(Censored)"]

    def _create_name_to_code_translation(self):
        """If a description is not unique, we take the longest code."""
        return (
            self._load_drug_coding()
            .groupby("atc_name")
            .atc_code.agg(lambda values: max(values, key=len))
            .to_dict()
        )

    def _create_code_to_name_translation(self):
        """If an atc code is not unique, we take the shortest description."""
        return (
            self._load_drug_coding()
            .groupby("atc_code")
            .atc_name.agg(lambda values: min(values, key=len))
            .to_dict()
        )

    @staticmethod
    def _load_drug_coding():
        return pd.read_csv(PROJECT_DATA / "atc_coding.csv.gz")
