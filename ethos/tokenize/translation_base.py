import abc
from typing import Optional, Mapping

import numpy as np
import pandas as pd

from ..constants import PROJECT_DATA
from ..utils import get_logger

logger = get_logger()


class Translation(abc.ABC):
    warn_exclude: Optional[list[str]] = None
    _NAME_TO_CODE: Optional[dict] = None
    _CODE_TO_NAME: Optional[dict] = None

    @property
    def name_to_code(self) -> dict:
        if self._NAME_TO_CODE is None:
            self._NAME_TO_CODE = self._create_name_to_code_translation()
        return self._NAME_TO_CODE

    @property
    def code_to_name(self) -> dict:
        if self._CODE_TO_NAME is None:
            self._CODE_TO_NAME = self._create_code_to_name_translation()
        return self._CODE_TO_NAME

    @abc.abstractmethod
    def _create_name_to_code_translation(self) -> dict:
        pass

    @abc.abstractmethod
    def _create_code_to_name_translation(self) -> dict:
        pass


class TranslationMixin(abc.ABC):
    def __init__(
        self,
        translation: Translation,
        pre_translation: Optional[Translation] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._translation: Translation = translation
        self._pre_translation: Optional[Translation] = pre_translation

    @property
    def _name_to_code(self) -> Mapping:
        return self._translation.name_to_code

    @property
    def _code_to_name(self) -> Mapping:
        return self._translation.code_to_name

    @property
    def _code_to_code(self) -> Mapping:
        return self._pre_translation.name_to_code

    @staticmethod
    def validate_translation(use_warn_exclude: bool = True):
        def decorator(func):
            def wrapper(self, names: pd.Series) -> pd.DataFrame:
                out_df = func(self, names)
                not_translated = names[out_df.isna().all(axis=1) & names.notna()].unique()
                warn_exclude = self._translation.warn_exclude
                if use_warn_exclude and warn_exclude is not None:
                    not_translated = np.setdiff1d(not_translated, warn_exclude)
                if not_translated.size != 0:
                    not_translated.sort()
                    logger.warning(
                        "[%s:%s] No translation for %d names: %s",
                        self.__class__.__name__,
                        func.__name__,
                        not_translated.size,
                        ", ".join(f"'{v}'" for v in not_translated),
                    )
                return out_df

            return wrapper

        return decorator


class IcdMixin(TranslationMixin):
    ICD_TYPES = ("cm", "pcs")

    def __init__(self, icd_type: str, translation: Translation, *args, **kwargs):
        super().__init__(translation, *args, **kwargs)
        self._icd9_to_icd10: Optional[dict] = None
        if icd_type in self.ICD_TYPES:
            self.mapping_filename = f"icd_{icd_type}_9_to_10_mapping.csv.gz"
            self.icd_type = icd_type
        else:
            raise ValueError(f"Invalid ICD type '{icd_type}', must be one of {self.ICD_TYPES}")

    @TranslationMixin.validate_translation()
    def _process_icd_names(self, icd_names: pd.Series) -> pd.DataFrame:
        return self._process_icd_codes(icd_names.map(self._name_to_code))

    @TranslationMixin.validate_translation(use_warn_exclude=False)
    def _process_icd_codes(self, icd_codes: pd.Series) -> pd.DataFrame:
        if self.icd_type == "cm":
            return self._process_icd_cm_codes(icd_codes)
        return self._process_icd_pcs_codes(icd_codes)

    def _process_icd_cm_codes(self, icd_codes: pd.Series) -> pd.DataFrame:
        icd_part1 = (
            icd_codes.str[:3].map(self._code_to_name).map(lambda v: "ICD_" + v, na_action="ignore")
        )
        # TODO: should be 3-5 or 4-6 since it's 3 characters long
        icd_part2 = icd_codes.str[3:6].map(
            lambda v: f"ICD_4-5_{v}" if v else np.nan, na_action="ignore"
        )
        icd_part3 = icd_codes.str[6:].map(
            lambda v: f"ICD_6-_{v}" if v else np.nan, na_action="ignore"
        )
        icd_codes_df = pd.concat([icd_part1, icd_part2, icd_part3], axis=1)
        icd_codes_df.columns = ["icd_part1", "icd_part2", "icd_part3"]
        return icd_codes_df

    @staticmethod
    def _process_icd_pcs_codes(icd_codes: pd.Series) -> pd.DataFrame:
        icd_parts = [
            icd_codes.str[i].map(lambda v: f"ICD_PCS_{v}", na_action="ignore") for i in range(7)
        ]
        icd_codes_df = pd.concat(icd_parts, axis=1)
        icd_codes_df.columns = [f"icd_part{i}" for i in range(1, 8)]
        return icd_codes_df

    @TranslationMixin.validate_translation(use_warn_exclude=False)
    def _convert_icd9_to_icd10(self, icd_codes: pd.Series) -> pd.DataFrame:
        return icd_codes.map(self.icd9_to_icd10).to_frame()

    @property
    def icd9_to_icd10(self) -> dict:
        if self._icd9_to_icd10 is None:
            self._icd9_to_icd10 = self._create_icd_9_to_10_translation()
        return self._icd9_to_icd10

    def _create_icd_9_to_10_translation(self):
        version_mapping = pd.read_csv(PROJECT_DATA / self.mapping_filename, dtype=str)
        version_mapping.drop_duplicates(subset="icd_9", inplace=True)
        version_mapping = version_mapping.groupby("icd_9").icd_10.apply(
            lambda values: min(values, key=len)
        )
        return version_mapping.to_dict()


class AtcMixin(TranslationMixin):
    @TranslationMixin.validate_translation()
    def _process_atc_names(self, atc_names: pd.Series) -> pd.DataFrame:
        return self._process_atc_codes(atc_names.map(self._name_to_code))

    @TranslationMixin.validate_translation(use_warn_exclude=False)
    def _process_atc_codes(self, _atc_codes: pd.Series) -> pd.DataFrame:
        atc_codes = _atc_codes
        if self._pre_translation is not None:
            atc_codes = atc_codes.map(self._code_to_code)
        atc_part1 = atc_codes.str[:3].map(self._code_to_name)
        atc_part2 = atc_codes.str[3:4].map(lambda v: f"ATC_4_{v}" if v else np.nan)
        atc_part3 = atc_codes.str[4:].map(lambda v: f"ATC_SUFFIX_{v}" if v else np.nan)
        atc_codes_df = pd.concat([atc_part1, atc_part2, atc_part3], axis=1)
        atc_codes_df.columns = ["atc_part1", "atc_part2", "atc_part3"]
        atc_codes_df["atc_part1"] = "ATC_" + atc_codes_df["atc_part1"]
        return atc_codes_df
