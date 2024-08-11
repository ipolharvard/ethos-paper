import pandas as pd

from ..base import SimpleData
from ..translation_base import Translation
from ...constants import PROJECT_DATA


class DrugTranslation(Translation):
    def __init__(self):
        self._name_to_code: pd.Series = None

    def _create_name_to_code_translation(self) -> pd.Series:
        drug_to_atc = self._load_drug_to_atc()
        drug_to_atc.drug = drug_to_atc.drug.str.strip().str.lower()
        self._name_to_code = drug_to_atc.drop_duplicates().set_index("drug").atc_code
        return self._name_to_code

    def _create_code_to_name_translation(self) -> dict:
        if self._name_to_code is None:
            self._name_to_code = self._create_name_to_code_translation()
        return self._name_to_code.reset_index().set_index("atc_code").drug.to_dict()

    @staticmethod
    def _load_drug_to_atc() -> pd.Series:
        return pd.read_csv(
            PROJECT_DATA / "mimic_drug_to_atc.csv.gz", dtype=str
        )


class _IcdTranslation(Translation):
    def __init__(self, data_prop, data_name):
        self.data_prop = data_prop
        self.data_name = data_name

    def _create_name_to_code_translation(self):
        df = self._load_icd_data()
        return (
            df.loc[df.icd_version == 10]
            .groupby("long_title")
            .icd_code.agg(lambda values: max(values, key=len))
            .to_dict()
        )

    def _create_code_to_name_translation(self):
        df = self._load_icd_data()
        return (
            df.loc[df.icd_version == 10]
            .groupby("icd_code")
            .long_title.agg(lambda values: min(values, key=len))
            .to_dict()
        )

    def _load_icd_data(self):
        return SimpleData(
            self.data_name,
            self.data_prop,
            use_cols=["icd_code", "long_title", "icd_version"],
            no_id_ok=True,
        ).df


class IcdCmTranslation(_IcdTranslation):
    def __init__(self, data_prop):
        super().__init__(data_prop, "hosp/d_icd_diagnoses")


class IcdPcsTranslation(_IcdTranslation):
    def __init__(self, data_prop):
        super().__init__(data_prop, "hosp/d_icd_procedures")
