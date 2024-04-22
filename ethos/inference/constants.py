from enum import Enum


class Test(Enum):
    ADMISSION_MORTALITY = "admission_mortality"
    READMISSION = "readmission"
    MORTALITY = "mortality"
    SINGLE_ADMISSION = "single_admission"
    # MIMIC-specific
    DRG_PREDICTION = "drg"
    SOFA_PREDICTION = "sofa"
    ICU_MORTALITY = "icu_mortality"
    ICU_READMISSION = "icu_readmission"
