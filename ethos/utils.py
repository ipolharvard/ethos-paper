import logging
from contextlib import nullcontext
from pathlib import Path, PurePath
from typing import Optional

import colorlog
import h5py
import numpy as np
import pandas as pd
import torch as th
from joblib._store_backends import FileSystemStoreBackend

from .model import ModelConfig, Ethos

_default_handler: Optional[logging.Handler] = None
_log_colors = {
    "DEBUG": "cyan",
    "INFO": "yellow",
    "WARNING": "light_yellow",
    "ERROR": "red,bg_white",
    "CRITICAL": "red,bg_white",
}


def get_logger():
    global _default_handler

    logger_name = __name__.split(".")[0]
    logger = logging.getLogger(logger_name)
    if _default_handler is None:
        _setup_handler()
        logger.addHandler(_default_handler)
        logger.setLevel(logging.INFO)
    return logger


def _setup_handler():
    global _default_handler

    _default_handler = colorlog.StreamHandler()
    _default_handler.setLevel(logging.INFO)

    header = "[%(levelname)1.1s %(asctime)s]"
    message = "%(message)s"
    formatter = colorlog.ColoredFormatter(
        f"%(green)s{header}%(reset)s %(log_color)s{message}%(reset)s", log_colors=_log_colors
    )
    _default_handler.setFormatter(formatter)


def load_data(
    data_path, n_tokens: Optional[int] = None, token_dtype=np.int32
) -> dict[str, np.ndarray | th.Tensor]:
    with h5py.File(data_path, "r") as f:
        times = th.from_numpy(f["times"][:n_tokens].astype(np.float32))
        tokens = th.from_numpy(f["tokens"][:n_tokens].astype(token_dtype))
        patient_context = th.from_numpy(f["patient_context"][:].astype(token_dtype))
        age_reference = f["age_reference"][:].astype(np.int32)
        patient_data_offsets = f["patient_data_offsets"][:].astype(np.int64)
        patient_ids = f["patient_ids"][:]
    return {
        "times": times,
        "tokens": tokens,
        "patient_context": patient_context,
        "age_reference": age_reference,
        "patient_data_offsets": patient_data_offsets,
        "patient_ids": patient_ids,
    }


def setup_torch(device, dtype, seed=42):
    th.manual_seed(seed)
    device_type = "cuda" if "cuda" in device else "cpu"
    if dtype == "bfloat16" and device_type == "cuda" and not th.cuda.is_bf16_supported():
        print("WARNING: bfloat16 is not supported on this device, using float16 instead")
        dtype = "float16"
    if device_type == "cuda":
        th.cuda.manual_seed(seed)
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
    ptdtype = {"float32": th.float32, "bfloat16": th.bfloat16, "float16": th.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else th.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return ctx


def load_model_from_checkpoint(path, device, for_training=True, **kwargs):
    checkpoint = th.load(path, map_location=device)
    gptconf = ModelConfig(**checkpoint["model_args"])
    model = Ethos(gptconf, **kwargs)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if not for_training:
        return model, checkpoint["config"].block_size
    best_val_loss = checkpoint.get("best_val_loss", 1e9)
    return model, checkpoint["iter_num"], best_val_loss, checkpoint["optimizer"]


def convert_seconds(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    hours = f"{int(hours):0>2}h:" if hours > 0 else ""
    minutes = f"{int(minutes):0>2}m:" if minutes > 0 else ""
    seconds = f"{int(seconds):0>2}s"
    return f"{hours}{minutes}{seconds}"


def convert_years(years: float, unit: str) -> float:
    if unit == "m":  # minutes
        return years * 365.25 * 24 * 60
    elif unit == "h":  # hours
        return years * 365.25 * 24
    elif unit == "d":  # days
        return years * 365.25
    elif unit == "w":  # weeks
        return years * 52.1775
    elif unit == "mt":  # months
        return years * 12
    else:
        print('Invalid unit. Please use "m", "h", "d", "w" or "mt".')


def unify_str_col(col: pd.Series) -> pd.Series:
    return col.str.replace(" ", "_").str.upper().str.replace(",", "").str.replace(".", "")

class DataFrameStoreBackend(FileSystemStoreBackend):
    def load_item(self, path, verbose=1, msg=None):
        full_path = Path(self.location) / PurePath(*path) / "output.pkl"
        return pd.read_pickle(full_path)

    def dump_item(self, path, item: pd.DataFrame, verbose=1, msg=None):
        if not isinstance(item, pd.DataFrame):
            raise ValueError("item must be a pandas DataFrame")
        item_path = Path(self.location) / PurePath(*path)
        item_path.mkdir(parents=True, exist_ok=True)
        full_path = item_path / "output.pkl"
        item.to_pickle(full_path)
