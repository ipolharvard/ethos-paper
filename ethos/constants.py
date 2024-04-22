from importlib.resources import files
from pathlib import Path

PROJECT_DATA: Path = files("ethos.data")
PROJECT_ROOT: Path = (PROJECT_DATA / "../..").resolve()

ADMISSION_STOKEN = "INPATIENT_ADMISSION_START"
DISCHARGE_STOKEN = "INPATIENT_ADMISSION_END"
# present only in the MIMIC dataset
ICU_ADMISSION_STOKEN = "ICU_STAY_START"
ICU_DISCHARGE_STOKEN = "ICU_STAY_END"
