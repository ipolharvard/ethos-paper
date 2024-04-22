from typing import Optional

import numpy as np
import pandas as pd

from .atc import AtcTranslation
from .code_translation import IcdCmTranslation, IcdPcsTranslation, DrugTranslation
from .. import SpecialToken
from ..base import TimeData, ContextData, SimpleData
from ..constants import SECONDS_IN_YEAR
from ..translation_base import IcdMixin, AtcMixin
from ..vocabulary import QStorageContext
from ...constants import PROJECT_DATA


class DemographicData(ContextData):
    COLUMNS_WE_USE = ["race", "marital_status"]
    RACE_UNKNOWN = {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER", np.nan}
    RACE_MINOR = {
        "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
        "AMERICAN INDIAN/ALASKA NATIVE",
        "MULTIPLE RACE/ETHNICITY",
    }

    def __init__(self, data_prop, **kwargs):
        super().__init__("hosp/admissions", data_prop, use_cols=self.COLUMNS_WE_USE, **kwargs)
        self.patient_df = SimpleData("hosp/patients", data_prop, use_cols=["gender"], **kwargs).df

    def _process(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        df = df.groupby(self.id_col, as_index=False).first()
        df = self.patient_df.merge(df, on=self.id_col, how="left")
        df.race = df.race.map(self._process_race)
        df.loc[df.marital_status.isna(), "marital_status"] = "UNKNOWN"
        df.marital_status = "MARITAL_" + df.marital_status
        df.gender = "SEX_" + df.gender
        return df

    def _process_race(self, v: Optional[str]) -> str:
        if v in self.RACE_UNKNOWN:
            v = "UNKNOWN"
        elif v in self.RACE_MINOR:
            v = "OTHER"
        elif v == "SOUTH AMERICAN":
            v = "HISPANIC"
        elif v == "PORTUGUESE":
            v = "WHITE"
        else:
            v = v[: None if (index := v.find("/")) == -1 else index]
            v = v[: None if (index := v.find(" ")) == -1 else index]
        return "RACE_" + v


class BMIData(ContextData):
    COLUMNS_WE_USE = ["result_name", "result_value"]

    def __init__(self, data_prop, **kwargs):
        super().__init__("hosp/omr", data_prop, use_cols=self.COLUMNS_WE_USE, **kwargs)
        self.patient_df = SimpleData("hosp/patients", data_prop, use_cols=[], **kwargs).df

    def _process(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        df = (
            df[~df.result_name.isna() & df.result_name.str.startswith("BMI")]
            .groupby(self.id_col, as_index=False)
            .last()
        )
        df.drop(columns="result_name", inplace=True)
        with QStorageContext("BMI", self.vocab) as q_storage:
            df["result_value"] = self._convert_to_deciles(
                q_storage, df.result_value.astype(float), "BMI"
            )
        df = self.patient_df.merge(df, on=self.id_col, how="left")
        df.fillna("BMI_UNKNOWN", inplace=True)
        return df


class AgeReferenceData(ContextData):
    COLUMNS_WE_USE = ["anchor_age", "anchor_year_group"]

    def __init__(self, data_prop, vocab=None, **kwargs):
        assert vocab is None, f"{self.__class__.__name__} does not use vocab"
        super().__init__(
            "hosp/patients", data_prop, vocab=None, use_cols=self.COLUMNS_WE_USE, **kwargs
        )

    def _process(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        df["anchor_year_group"] = (
            df.anchor_year_group.str.split(" - ", expand=True)[0].astype(int) + 1 - df.anchor_age
        )
        df.drop(columns="anchor_age", inplace=True)
        return df


class DeathData(TimeData):
    def __init__(self, data_prop, **kwargs):
        super().__init__(
            "hosp/patients",
            data_prop,
            time_col="dod",
            use_cols=[],
            allow_nat=True,
            **kwargs,
        )
        self.admissions_df = SimpleData(
            "hosp/admissions",
            data_prop,
            use_cols=["deathtime"],
            **kwargs,
        ).df

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        _df = df.loc[df.dod.notna()].copy()
        self.admissions_df = self.admissions_df.loc[self.admissions_df.deathtime.notna()].copy()
        subject_that_died_in_hospital = self.admissions_df[self.id_col]
        _df = _df.loc[~_df[self.id_col].isin(subject_that_died_in_hospital)].copy()
        _df[self.time_col] += pd.Timedelta(hours=23, minutes=59).total_seconds() / SECONDS_IN_YEAR
        _df["token"] = SpecialToken.DEATH
        return _df


class EdAdmissionData(TimeData):
    COLUMNS_WE_USE = ["edregtime", "edouttime"]

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            "hosp/admissions",
            data_prop,
            time_col="edregtime",
            use_cols=self.COLUMNS_WE_USE,
            allow_nat=True,
            **kwargs,
        )

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        df = df[~df.edregtime.isna()].reset_index(drop=True)
        return self._process_bracket_data(df, "ED_ADMISSION", "edregtime", "edouttime")


class InpatientAdmissionData(TimeData):
    COLUMNS_WE_USE = [
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "insurance",
        "discharge_location",
        "hadm_id",
    ]
    SCHEDULED_ADMISSIONS = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
    DISCHARGE_FACILITIES = {
        "HEALTHCARE FACILITY",
        "SKILLED NURSING FACILITY",
        "REHAB",
        "CHRONIC/LONG TERM ACUTE CARE",
        "OTHER FACILITY",
    }
    DRG_UNKNOWN = "UNKNOWN_DRG"

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            "hosp/admissions",
            data_prop,
            time_col="admittime",
            use_cols=self.COLUMNS_WE_USE,
            **kwargs,
        )
        self.drg_codes: pd.DataFrame = SimpleData(
            "hosp/drgcodes",
            data_prop,
            use_cols=["hadm_id", "drg_type", "drg_code", "description"],
            no_id_ok=True,
            **kwargs,
        ).df

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        died_during_admission_mask = df.deathtime.notna()
        df.drop(columns="deathtime", inplace=True)
        df["admission_type"] = df.admission_type.map(self._process_admission_type)
        df["insurance"] = "INSURANCE_" + df.insurance.str.upper()
        df["discharge_location"] = df.discharge_location.map(self._process_discharge_location)
        # include drg codes
        self.drg_codes = self.drg_codes.loc[self.drg_codes.drg_type == "HCFA"].copy()
        # some codes map to multiple descriptions, we take the first one
        self.drg_codes.description = self.drg_codes.drg_code.map(
            self.drg_codes.groupby("drg_code").description.first()
        )
        df = df.merge(self.drg_codes[["hadm_id", "description"]], on="hadm_id", how="left").drop(
            columns="hadm_id"
        )
        df.loc[df.description.isna(), "description"] = self.DRG_UNKNOWN
        with QStorageContext("DRG_CODE", self.vocab) as q_storage:
            if q_storage:
                known_values = q_storage.values()
                is_known_mask = df.description.isin(known_values)
                if self.DRG_UNKNOWN in known_values:
                    df.loc[~is_known_mask, "description"] = self.DRG_UNKNOWN
                else:
                    df = df.loc[is_known_mask].copy()
            else:
                q_storage.register_values(df.description)
        return self._process_bracket_data(
            df,
            "INPATIENT_ADMISSION",
            "admittime",
            "dischtime",
            right_bracket_cols=["discharge_location", "description"],
            outcome_death=died_during_admission_mask,
        )

    def _process_admission_type(self, v: str):
        if v.endswith("EMER.") or v == "URGENT":
            v = "EMERGENCY"
        elif v in self.SCHEDULED_ADMISSIONS:
            v = "SCHEDULED"
        else:
            v = "OBSERVATION"
        return "TYPE_" + v

    def _process_discharge_location(self, v: str):
        if v in self.DISCHARGE_FACILITIES:
            v = "HEALTHCARE_FACILITY"
        if pd.isnull(v):
            v = "UNKNOWN"
        return "DISCHARGED_" + v.replace(" ", "_")


class TransferData(TimeData):
    COLUMNS_WE_USE = ["curr_service"]

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            "hosp/services",
            data_prop,
            time_col="transfertime",
            use_cols=self.COLUMNS_WE_USE,
            **kwargs,
        )

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        df["curr_service"] = "TRANSFER_" + df.curr_service
        return df


class ICUStayData(TimeData):
    COLUMNS_WE_USE = ["stay_id", "intime", "outtime", "first_careunit"]

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            "icu/icustays",
            data_prop,
            time_col="intime",
            use_cols=self.COLUMNS_WE_USE,
            **kwargs,
        )
        self.sofa_scores: pd.DataFrame = pd.read_csv(
            PROJECT_DATA / "mimic-iv_derived.csv.gz",
            usecols=["stay_id", "first_day_sofa"],
        )

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        df["sofa_token"] = "SOFA"
        df = df.merge(self.sofa_scores, on="stay_id")
        df.drop(columns=["stay_id"], inplace=True)
        with QStorageContext("ICU_STAY", self.vocab) as q_storage:
            df.first_day_sofa = self._convert_to_deciles(
                q_storage, df.first_day_sofa, "sofa", scheme="equidistant"
            )
        return self._process_bracket_data(df, "ICU_STAY", "intime", "outtime")


class DiagnosisData(IcdMixin, TimeData):
    COLUMNS_WE_USE = ["hadm_id"]

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            icd_type="cm",
            translation=IcdCmTranslation(data_prop),
            data_name="hosp/admissions",
            data_prop=data_prop,
            time_col="admittime",
            use_cols=self.COLUMNS_WE_USE,
            **kwargs,
        )
        self.icd_codes: pd.DataFrame = SimpleData(
            "hosp/diagnoses_icd",
            data_prop,
            use_cols=["hadm_id", "icd_code", "icd_version"],
            no_id_ok=True,
            **kwargs,
        ).df

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        is_icd_9 = self.icd_codes.icd_version == 9
        self.icd_codes.drop(columns="icd_version", inplace=True)
        self.icd_codes.loc[is_icd_9, "icd_code"] = self._convert_icd9_to_icd10(
            self.icd_codes.loc[is_icd_9, "icd_code"]
        )
        with QStorageContext("ICD_CM_CODE", self.vocab) as q_storage:
            if q_storage:
                self.icd_codes = self.icd_codes.loc[
                    self.icd_codes.icd_code.isin(q_storage.values())
                ].copy()
            else:
                q_storage.register_values(self.icd_codes.icd_code)
        self.icd_codes = self.icd_codes.merge(
            self._process_icd_codes(self.icd_codes["icd_code"]), left_index=True, right_index=True
        )
        self.icd_codes.drop(columns="icd_code", inplace=True)
        df = df.merge(self.icd_codes, on="hadm_id")
        df.drop(columns="hadm_id", inplace=True)
        return df


class ProcedureData(IcdMixin, TimeData):
    COLUMNS_WE_USE = ["icd_code", "icd_version"]

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            icd_type="pcs",
            translation=IcdPcsTranslation(data_prop),
            data_name="hosp/procedures_icd",
            data_prop=data_prop,
            time_col="chartdate",
            use_cols=self.COLUMNS_WE_USE,
            **kwargs,
        )

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        is_icd_9 = df.icd_version == 9
        df.drop(columns="icd_version", inplace=True)
        df.loc[is_icd_9, "icd_code"] = self._convert_icd9_to_icd10(df.loc[is_icd_9, "icd_code"])
        with QStorageContext("ICD_PCS_CODE", self.vocab) as q_storage:
            if q_storage:
                is_known_mask = df.icd_code.isin(q_storage.values())
                df = df.loc[is_known_mask].copy()
            else:
                q_storage.register_values(df.icd_code)
        df = df.merge(self._process_icd_codes(df.icd_code), left_index=True, right_index=True)
        df.drop(columns="icd_code", inplace=True)
        return df


class BloodPressureData(TimeData):
    COLUMNS_WE_USE = ["result_name", "result_value"]

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            "hosp/omr", data_prop, time_col="chartdate", use_cols=self.COLUMNS_WE_USE, **kwargs
        )

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        df = df[df.result_name.str.startswith("Blood")].copy()
        df.result_name = "BLOOD_PRESSURE"
        # divide blood pressure measure which is in the following format: 'sbp/dbp'
        df.rename(columns={"result_value": "sbp"}, inplace=True)
        df[["sbp", "dbp"]] = df.sbp.str.split("/", expand=True)
        # manage quantile storage
        with QStorageContext("BLOOD_PRESSURE", self.vocab) as q_storage:
            df["sbp"] = self._convert_to_deciles(q_storage, df.sbp.astype(float), "sbp")
            df["dbp"] = self._convert_to_deciles(q_storage, df.dbp.astype(float), "dbp")
        return df


class AdministeredMedicationData(AtcMixin, TimeData):
    COLUMNS_WE_USE = ["medication", "event_txt"]

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            translation=AtcTranslation(),
            data_name="hosp/emar",
            data_prop=data_prop,
            time_col="charttime",
            use_cols=self.COLUMNS_WE_USE,
            **kwargs,
        )
        self.drug_to_atc = DrugTranslation(data_prop)._create_name_to_code_translation()

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        df = df.loc[(df.event_txt == "Administered") & df.medication.notna()].copy()
        df.drop(columns=["event_txt"], inplace=True)
        with QStorageContext("MEDICATION", self.vocab) as q_storage:
            if q_storage:
                is_known_mask = df.medication.isin(q_storage.values())
                df = df.loc[is_known_mask].copy()
            else:
                # todo: breaks on None, should be investigated where these None values come from
                q_storage.register_values(df.medication)
        df = df.merge(self.drug_to_atc, left_on="medication", right_on="drug").drop(
            columns="medication"
        )
        df = df.join(self._process_atc_codes(df.atc_code)).drop(columns="atc_code")
        return df


class LabTestData(TimeData):
    COLUMNS_WE_USE = ["itemid", "valuenum", "valueuom"]
    N_MOST_COMMON = 200

    def __init__(self, data_prop, **kwargs):
        super().__init__(
            "hosp/labevents",
            data_prop,
            time_col="charttime",
            use_cols=self.COLUMNS_WE_USE,
            **kwargs,
        )
        self.lab_items: pd.DataFrame = SimpleData(
            "hosp/d_labitems",
            data_prop,
            use_cols=["itemid", "label"],
            no_id_ok=True,
            **kwargs,
        ).df

    def _process(self, df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        # remove non-numeric tests
        df = df.loc[df.valuenum.notna()].copy()
        # translate lab item ids to lab test names
        df = df.merge(self.lab_items, on="itemid")
        df.drop(columns=["itemid"], inplace=True)
        df["label"] = "LAB_" + df.label + "_" + df.valueuom.fillna("no_unit")
        df.drop(columns=["valueuom"], inplace=True)
        # manage quantile storage
        with QStorageContext("LAB_TEST", self.vocab) as q_storage:
            lab_gb = df.groupby("label")
            if q_storage:
                considered_lab_tests = q_storage.values()
            else:
                considered_lab_tests = set(lab_gb.size().nlargest(self.N_MOST_COMMON).index)
            # convert to quantiles, drop tests that we do not consider
            df["valuenum"] = lab_gb["valuenum"].transform(
                lambda values: self._convert_to_deciles(q_storage, values, values.name)
                if values.name in considered_lab_tests
                else pd.Series()
            )
        df.dropna(inplace=True)
        # set columns in the proper order
        df = df[[self.id_col, self.time_col, "label", "valuenum"]]
        return df
