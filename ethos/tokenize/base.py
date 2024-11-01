import abc
from typing import Optional, Sequence, Hashable

import numpy as np
import pandas as pd
from joblib import Memory, register_store_backend

from .constants import TIME_DTYPE, TOKEN_DTYPE, SECONDS_IN_YEAR, DataProp, Dataset
from .special_tokens import SpecialToken
from .vocabulary import Vocabulary, QStorage, QStorageContext
from ..constants import PROJECT_DATA, PROJECT_ROOT
from ..utils import get_logger, DataFrameStoreBackend

logger = get_logger()
register_store_backend("df_store", DataFrameStoreBackend)


class SimpleData:
    id_col: str
    vocab: Optional[Vocabulary]
    is_processed: bool
    df: pd.DataFrame
    data_prop: DataProp

    def __init__(
            self,
            data_name: str,
            data_prop: DataProp,
            vocab: Optional[Vocabulary] = None,
            use_parquet: bool = False,
            use_cols: Optional[list] = None,
            load_all_cols: bool = False,
            no_id_ok: bool = False,
            use_cache: bool = False,
            **kwargs,
    ):
        """
        :param data_name: Name of the subset (table) of a dataset, e.g., "procedures".
        :param data_prop: Dataset-specific structure, an object that reflects file hierarchy of
         the given dataset.
        :param use_parquet: Whether to load datasets in their parquet format. This is faster, but
         then we have to load entire columns. That's why we don't use it by default.
        :param use_cols: Columns to load. If None, all columns are loaded.
        :param load_all_cols: Whether to load all columns. Ik True, use_cols is ignored.
        :param no_id_ok: Whether to allow the dataset to not have an ID column, used for
         complementary datasets.
        :param kwargs: Additional arguments passed to pandas.read_csv or pandas.read_parquet.
        """
        self.id_col = data_prop.id_col
        self.vocab = vocab
        self.is_processed = False
        self.data_prop = data_prop

        load_data = True
        if use_cache:
            memory = Memory(location=PROJECT_ROOT / "cache", backend="df_store", verbose=0)
            self._process = memory.cache(self._process, ignore=["self", "df"])
            func_id, args_id = self._process._get_output_identifiers(None, None)
            if self._process._is_in_cache_and_valid([func_id, args_id]):
                logger.info(f"[{self.__class__.__name__}] Loading processed data from cache...")
                self.df = None
                self.process()
                load_data = False

        if load_data:
            self.df = self._load_data(
                self.__class__.__name__,
                data_name,
                data_prop,
                use_parquet,
                use_cols,
                load_all_cols,
                no_id_ok,
                **kwargs,
            )

    @staticmethod
    def _load_data(
            cls_name: str,
            data_name: str,
            data_prop: DataProp,
            use_parquet: bool = False,
            use_cols: Optional[list] = None,
            load_all_cols: bool = False,
            no_id_ok: bool = False,
            **kwargs,
    ) -> pd.DataFrame:
        dataset_dir = data_prop.dataset_dir
        fold_dir = data_prop.fold_dir
        suffix = "." + data_prop.csv_format

        _use_parquet = False
        if use_parquet:
            parquet_suffix = ".parquet"
            parquet_dataset_dir = f"{dataset_dir}_parquet"
            if (
                    (PROJECT_DATA / parquet_dataset_dir / fold_dir / data_name)
                            .with_suffix(parquet_suffix)
                            .exists()
            ):
                dataset_dir = parquet_dataset_dir
                suffix = parquet_suffix
                _use_parquet = True
            else:
                logger.warn(
                    f"Attempted to use parquet data for efficiency, but did not find it. "
                    f"Falling back to CSV."
                )
        data_path = (PROJECT_DATA / dataset_dir / data_prop.fold_dir / data_name).with_suffix(suffix)

        _use_cols = use_cols if not load_all_cols else None
        if _use_cols is not None and not no_id_ok:
            _use_cols = [data_prop.id_col, *use_cols]

        if _use_parquet:
            # warn that kwargs will not be used, include class name
            if kwargs:
                logger.warn(f"[{cls_name}]: kwargs are not used when loading parquet data.")
            df = pd.read_parquet(data_path, columns=_use_cols)
        else:
            df = pd.read_csv(data_path, usecols=_use_cols, **kwargs)

        if _use_cols is not None:
            # set columns in the requested order
            df = df[_use_cols]

        assert data_prop.id_col in df.columns or no_id_ok, (
            f"ID column {data_prop.id_col} not found in {data_name} dataset. "
            f"Columns found: {list(df.columns)}"
        )
        return df

    def __repr__(self):
        return "{}(dataset='{}', fold='{}')".format(
            self.__class__.__name__,
            self.data_prop.name.value,
            self.data_prop.fold.value,
        )

    @staticmethod
    def _unify_col_names(columns) -> list[str]:
        return [str(col).lower().replace(" ", "_") for col in columns]

    def process(self) -> pd.DataFrame:
        if self.is_processed:
            return self.df
        self.df = self._process(self.df)
        self.is_processed = True
        return self.df

    @abc.abstractmethod
    def _process(self, df) -> pd.DataFrame:
        raise NotImplementedError("Implement this method in a subclass.")

    def get_timelines(
            self,
    ) -> dict[Hashable, tuple[np.ndarray[TIME_DTYPE], np.ndarray[TOKEN_DTYPE | str]]]:
        raise NotImplementedError("Implement this method in a subclass.")

    @staticmethod
    def _convert_to_deciles(
            q_storage: Optional[QStorage],
            values: pd.Series,
            record_name: str,
            scheme: str = "quantiles",
    ) -> pd.Series:
        """Use this method to convert values to centiles since we don't always have a q_storage."""
        if q_storage is None:
            return SpecialToken.convert_to_deciles(values, scheme=scheme)
        return q_storage.values_to_deciles(values, record_name, scheme)


class ContextData(SimpleData, abc.ABC):
    def get_timelines(self) -> dict[Hashable, np.ndarray[TOKEN_DTYPE | str]]:
        tokens = self.df.drop(columns=self.id_col).values
        if self.vocab is not None:
            tokens = self.vocab.tokenize(tokens.flat).reshape(tokens.shape[0], -1)
        return dict(zip(self.df[self.id_col], tokens))


class TimeData(SimpleData, abc.ABC):
    time_col: str
    EPS = 1e-8

    def __init__(
            self,
            data_name: str,
            data_prop: DataProp,
            time_col: str,
            use_cols: Optional[list[str]] = None,
            allow_nat: bool = False,
            **kwargs,
    ):
        _use_cols = None
        if use_cols is not None:
            _use_cols = use_cols if time_col in use_cols else [time_col, *use_cols]

        super().__init__(data_name, data_prop, use_cols=_use_cols, **kwargs)

        self.time_col = time_col
        if use_cols is not None and self.time_col in use_cols:
            # if the same col is both the `time_col` and in `use_cols`, it's duplicated
            self.time_col = "_" + self.time_col
            if not self.is_processed:
                cols = [col for col in self.df.columns if col != self.id_col]
                self.df[self.time_col] = self.df[time_col].copy()
                self.df = self.df[[self.id_col, self.time_col, *cols]]

        assert self.time_col in self.df.columns, (
            f"Time column {self.time_col} not found in {data_name} dataset. "
            f"Columns found: {list(self.df.columns)}"
        )

        if data_prop.name == Dataset.MIMIC and not self.is_processed:
            self.df[self.time_col] = self._dates_to_year_format(self.df[self.time_col], allow_nat)

    @staticmethod
    def _dates_to_year_format(values: pd.Series, allow_nat: bool = False) -> pd.Series:
        if allow_nat:
            return pd.to_datetime(values, errors="coerce").map(
                lambda v: v.timestamp() / SECONDS_IN_YEAR, na_action="ignore"
            )
        return pd.to_datetime(values).map(lambda v: v.timestamp() / SECONDS_IN_YEAR)

    def get_timelines(
            self,
    ) -> dict[Hashable, tuple[np.ndarray[TIME_DTYPE], np.ndarray[TOKEN_DTYPE | str]]]:
        """Nans in tokens are omitted."""
        assert (
                np.sum(nans := pd.isnull(self.df[self.time_col])) == 0
        ), f"Found NaNs in time_col, index: {np.flatnonzero(nans)}"

        times = self.df[self.time_col].values.astype(TIME_DTYPE)
        tokens = self.df.drop(columns=[self.id_col, self.time_col]).values

        if self.vocab is None:
            token_dtype = np.object_
        else:
            token_dtype = TOKEN_DTYPE
            tokens = self.vocab.tokenize(tokens.flat).reshape(self.df.shape[0], *tokens.shape[1:])

        def _conv_func(indices):
            token_subset = tokens[indices]
            new_times = np.fromiter(
                (
                    time
                    for i, time in enumerate(times[indices])
                    for j in range(tokens.shape[1])
                    if not pd.isna(token_subset[i, j])
                ),
                dtype=TIME_DTYPE,
            )
            new_tokens = np.fromiter(
                (token for token in token_subset.flat if not pd.isna(token)), dtype=token_dtype
            )
            return new_times, new_tokens

        out_dict = {
            patient_id: _conv_func(indices)
            for patient_id, indices in self.df.reset_index().groupby(self.id_col).groups.items()
        }
        return out_dict

    def _process_bracket_data(
            self,
            df: pd.DataFrame,
            prefix: str,
            start_col: str,
            end_col: str,
            right_bracket_cols: Optional[Sequence] = None,
            outcome_death: Optional[np.ndarray] = None,
            dates_is_year_format: bool = False,
    ) -> pd.DataFrame:
        """Produce timeline sequences in the form of brackets, e.g., depicts a situation when
        a patient stayed in the hospital for a certain amount of time.
        """
        # should be refactored to be unified across all datasets, to_datetime is outside the
        # scope of this function
        epsilon = 1e-6  # of a year -> around 32 seconds
        time_start, time_end, zero = df[start_col], df[end_col], 2 * epsilon
        if not dates_is_year_format:
            time_start, time_end = pd.to_datetime(time_start), pd.to_datetime(time_end)
            zero = pd.Timedelta(zero * 365.25, "d")
        len_of_stay = time_end - time_start
        len_of_stay.where(len_of_stay >= zero, inplace=True)
        # [period_len_q]
        with QStorageContext(prefix.upper(), self.vocab) as q_storage:
            df["period_len"] = self._convert_to_deciles(q_storage, len_of_stay, f"{prefix}_len")
        # [period_start]
        df[start_col] = f"{prefix}_start".upper()
        # [period_end]
        df[end_col] = f"{prefix}_end".upper()
        if outcome_death is not None:
            df.loc[outcome_death, end_col] = SpecialToken.DEATH
        # tokens that act as a closing bracket
        right_bracket_cols = [end_col, "period_len"] + (right_bracket_cols or [])
        left_bracket_cols = [col for col in df.columns if col not in right_bracket_cols]
        right_bracket_cols += [self.id_col, self.time_col]
        left_bracket_df = pd.melt(
            df[left_bracket_cols], id_vars=[self.id_col, self.time_col], ignore_index=False
        )
        left_bracket_df.drop(columns="variable", inplace=True)
        df = pd.melt(
            df[right_bracket_cols], id_vars=[self.id_col, self.time_col], ignore_index=False
        )
        df.drop(columns="variable", inplace=True)
        # add delta times to closing brackets to place them on the proper position in the timelines
        delta_time = len_of_stay.loc[df.index].fillna(zero)
        if not dates_is_year_format:
            delta_time = delta_time.dt.total_seconds() / SECONDS_IN_YEAR
        # subtract epsilon in case the next event starts at the exact same time as the previous one
        # ends
        df[self.time_col] += delta_time - epsilon
        df = pd.concat([left_bracket_df, df])
        return df
