from importlib.resources import files
from pathlib import Path

PROJECT_DATA: Path = files("ethos.data")
PROJECT_ROOT: Path = (PROJECT_DATA / "../..").resolve()

ADMISSION_STOKEN = "INPATIENT//ADMISSION"
DISCHARGE_STOKEN = "INPATIENT//DISCHARGE"
# present only in the MIMIC dataset
ICU_ADMISSION_STOKEN = "ICU//ADMISSION"
ICU_DISCHARGE_STOKEN = "ICU//DISCHARGE"
