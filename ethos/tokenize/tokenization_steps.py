import time
from collections import defaultdict
from importlib import import_module

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import TimeData, ContextData
from .constants import TIME_DTYPE, TOKEN_DTYPE, DataProp, DataFold
from .mimic.special import MimicPatientBirthDateData
from .separators import SeparatorInjector
from .special_tokens import SpecialToken
from .vocabulary import Vocabulary
from ..utils import convert_seconds, get_logger

logger = get_logger()


def load_and_process_data(
    data_prop: DataProp, vocab: Vocabulary, use_cache=False, **kwargs
) -> dict[int, list]:
    patient_timeline_chunks: dict[int, list] = defaultdict(list)

    module = import_module(f".{data_prop.module}.preprocessors", package="ethos.tokenize")
    registered_data = [
        cls
        for name, cls in module.__dict__.items()
        if isinstance(cls, type) and issubclass(cls, TimeData) and cls != TimeData
    ]
    for i, data_cls in enumerate(registered_data, 1):
        start_time = time.time()

        data = data_cls(data_prop, vocab=vocab, use_cache=use_cache, **kwargs)
        data.process()
        prev_vocab_size = len(vocab)
        new_timelines = data.get_timelines()

        for patient_id, (times, events) in new_timelines.items():
            patient_timeline_chunks[patient_id].append((times, events))

        logger.info(
            "[{}/{}] Processed: {} in {} [+{}t]".format(
                i,
                len(registered_data),
                data_cls.__name__,
                convert_seconds(time.time() - start_time),
                len(vocab) - prev_vocab_size,
            )
        )
    return patient_timeline_chunks


def merge_timeline_chunks(patient_timeline_chunks) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    patient_timelines = {}
    for patient_id, timelines in patient_timeline_chunks.items():
        time_chunks, token_chunks = zip(*timelines)
        patient_timelines[patient_id] = (np.concatenate(time_chunks), np.concatenate(token_chunks))
    return patient_timelines


def normalize_mimic_times(patient_timelines: dict, data_prop: DataProp, **kwargs):
    data = MimicPatientBirthDateData(data_prop, **kwargs)
    data.process()
    patient_birth_dates = data.get_timelines()
    return {
        patient_id: (times - patient_birth_dates[patient_id], tokens)
        for patient_id, (times, tokens) in patient_timelines.items()
    }


def _inject_separators(patient_id, times, event_tokens, injector: SeparatorInjector):
    """It also adds TIMELINE_END token."""
    # it's crucial to have it stable, because we rely on this everywhere
    sorted_idx = np.argsort(times, kind="stable")
    sorted_times, sorted_tokens = times[sorted_idx], event_tokens[sorted_idx]
    return patient_id, injector(sorted_times, sorted_tokens)


def inject_separators(patient_timelines, vocab, n_jobs=1):
    injector = SeparatorInjector(vocab)
    parsed_timelines = Parallel(n_jobs=n_jobs, batch_size=1000)(
        delayed(_inject_separators)(patient_id, times, tokens, injector)
        for patient_id, (times, tokens) in patient_timelines.items()
    )
    return dict(parsed_timelines)


def concatenate_timelines(
    patient_timelines,
) -> tuple[np.ndarray[float], np.ndarray[str], dict[int, int]]:
    times, tokens = zip(*list(patient_timelines.values()))
    patient_indices = np.cumsum([0] + [len(patient_times) for patient_times in times[:-1]])
    patient_idx_to_id = dict(zip(patient_indices, patient_timelines.keys()))
    times = np.concatenate(times, dtype=TIME_DTYPE)
    tokens = np.concatenate(tokens, dtype=TOKEN_DTYPE)
    return times, tokens, patient_idx_to_id


def estimate_true_sep_time(times, tokens, vocab):
    separator_tokens = vocab.encode(SpecialToken.SEPARATOR_NAMES)
    separator_indices = np.flatnonzero(np.isin(tokens, separator_tokens))
    sep_estimated_times = times[separator_indices + 1] - times[separator_indices - 1]
    separators = tokens[separator_indices]
    sep_df = pd.DataFrame(
        {
            "separator": vocab.decode(separators),
            "estimated_time": sep_estimated_times,
        }
    )
    sep_df = sep_df.groupby("separator").agg(["mean", "median", "std", "min", "max", "count"])
    sep_df.columns = sep_df.columns.droplevel(0)
    sep_df = sep_df.sort_values("mean")
    sep_df = sep_df.iloc[:-1]  # remove the last separator
    vocab.add_meta("separator_contrib", len(separator_indices) / len(tokens))
    vocab.add_meta("separator_estimates", sep_df.to_dict())


def get_context_data(data_prop, vocab, **kwargs):
    module = import_module(f".{data_prop.module}.preprocessors", package="ethos.tokenize")
    context_data = [
        cls
        for name, cls in module.__dict__.items()
        if isinstance(cls, type) and issubclass(cls, ContextData) and cls != ContextData
    ]
    age_reference = None
    context_chunks: list[dict] = []
    for data_cls in context_data:
        # AgeReferenceData is a special case
        if data_cls.__name__ == "AgeReferenceData":
            data = data_cls(data_prop, **kwargs)
            data.process()
            age_reference = data.get_timelines()
        else:
            data = data_cls(data_prop, vocab=vocab, **kwargs)
            data.process()
            context_chunks.append(data.get_timelines())
    assert age_reference is not None, "AgeReferenceData is required"
    # get common patient ids across context chunks, crucial when tokenizing a subset of a dataset
    patient_ids = set.intersection(*[set(chunk.keys()) for chunk in context_chunks])
    context = {
        patient_id: np.concatenate([chunk[patient_id] for chunk in context_chunks])
        for patient_id in patient_ids
    }
    return context, age_reference


def dump_timelines(timelines_path, times, tokens, patient_idx_to_id, context_data, age_reference):
    with h5py.File(timelines_path, "w") as f:
        f.create_dataset("times", data=times)
        f.create_dataset("tokens", data=tokens)

        patient_ids = list(patient_idx_to_id.values())
        f.create_dataset("patient_ids", data=patient_ids)
        f.create_dataset("patient_data_offsets", data=list(patient_idx_to_id.keys()))
        patient_context = np.asarray(
            [context_data[patient_id] for patient_id in patient_ids], dtype=TOKEN_DTYPE
        )
        f.create_dataset("patient_context", data=patient_context)
        # fromiter is possible because age_reference values are one-value arrays
        age_reference = np.fromiter(
            (age_reference[patient_id] for patient_id in patient_ids), dtype=np.int32
        )
        f.create_dataset("age_reference", data=age_reference)
