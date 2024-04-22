from typing import Optional

import numpy as np
import pandas as pd

from ..base import ContextData, SimpleData
from ..constants import DataProp, SECONDS_IN_YEAR


class MimicPatientBirthDateData(ContextData):
    COLUMNS_WE_USE = ["anchor_age", "anchor_year"]

    def __init__(self, data_prop, **kwargs):
        super().__init__("hosp/patients", data_prop, use_cols=self.COLUMNS_WE_USE, **kwargs)

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df["anchor_year"] -= df["anchor_age"]
        df.drop(columns="anchor_age", inplace=True)
        df.anchor_year = pd.to_datetime(df.anchor_year, format="%Y")
        df.anchor_year += pd.DateOffset(months=6)
        # convert datetimes to year format
        df.anchor_year = df.anchor_year.map(lambda v: v.timestamp() / SECONDS_IN_YEAR)
        return df


class ICUStayIdMixin:
    def __init__(self, patient_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_prop = DataProp.create("mimic", "test")
        icu_stay_df = SimpleData(
            "icu/icustays",
            data_prop,
            use_cols=["stay_id", "intime"],
            parse_dates=["intime"],
        ).df
        patient_order = {pid: i for i, pid in enumerate(patient_ids)}
        icu_stay_df["patient_order"] = icu_stay_df.subject_id.map(patient_order)
        icu_stay_df.sort_values(["patient_order", "intime"], inplace=True)
        self._stay_ids: np.ndarray = icu_stay_df.stay_id.values

    def _get_stay_id(self, icu_admission_no: int):
        return self._stay_ids[icu_admission_no]


class HadmIdMixin:
    _inpatient_stay: Optional[np.ndarray]

    def __init__(self, patient_ids=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if patient_ids is None:
            self._inpatient_stay = None
        else:
            data_prop = DataProp.create("mimic", "test")
            inpatient_stay_df = SimpleData(
                "hosp/admissions",
                data_prop,
                use_cols=["hadm_id", "admittime"],
                parse_dates=["admittime"],
            ).df
            patient_order = {pid: i for i, pid in enumerate(patient_ids)}
            inpatient_stay_df["patient_order"] = inpatient_stay_df.subject_id.map(patient_order)
            inpatient_stay_df.sort_values(["patient_order", "admittime"], inplace=True)
            self._inpatient_stay = inpatient_stay_df.hadm_id.values

    def _get_hadm_id(self, admission_no: int):
        return self._inpatient_stay[admission_no]
