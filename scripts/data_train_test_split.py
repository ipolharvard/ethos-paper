import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from joblib import Parallel, delayed

from ethos.utils import get_logger

logger = get_logger()


DEFAULT_FORMAT = ".csv.gz"
PARQUET_SUFFIX = ".parquet"


def dump_splits(
    col: list[str],
    orig_path: Path,
    split_paths: list[Path],
    cutoff_date: Optional[pd.Timestamp],
    subject_id_split: Optional[tuple[Sequence, Sequence]],
):
    if all(p.exists() for p in split_paths):
        logger.warning(f"All output files already exist, skipping: {orig_path}")
        return
    if orig_path.suffix == PARQUET_SUFFIX:
        df = pd.read_parquet(orig_path)
    else:
        df = pd.read_csv(orig_path, low_memory=False)

    id_col = None if subject_id_split is None else col[0]
    date_col = None if cutoff_date is None else col[-1]

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])

    for split_path in split_paths:
        _df = df.copy()
        processed = False
        # if there are 4 paths, we do both time and id based split
        if id_col is not None and id_col in df.columns:
            split_no = 0 if "train" in str(split_path) else 1
            subject_ids = subject_id_split[split_no]
            _df = _df.loc[_df[id_col].isin(subject_ids)]
            processed = True

        if date_col is not None and date_col in _df.columns and df[date_col].isna().mean() < 0.01:
            # either we do only cutoff date-based split or both time and id based split
            cond = _df[date_col] < cutoff_date
            fold_name = split_path.parts[1]
            if fold_name.endswith("prospective") or (
                len(split_paths) == 2 and fold_name.startswith("test")
            ):
                cond = ~cond
            _df = _df.loc[cond]
            processed = True

        # reconstruct the original file hierarchy
        split_path.parent.mkdir(exist_ok=True, parents=True)
        # save the splits
        logger.info(f"Saving the split ({_df.shape}): {split_path}")
        if split_path.exists():
            logger.warning(f"File already exists: {split_path}")
        else:
            if not processed:
                logger.warning(f"Neither time nor id based split was performed for: {split_path}")
            if orig_path.suffix == PARQUET_SUFFIX:
                _df.to_parquet(split_path, index=False)
            else:
                _df.to_csv(split_path, index=False)


def data_train_test_split(
    dataset_dir: str,
    col: str | list[str],
    test_size: float,
    id_data_path: str = None,
    cutoff_date: str = None,
    subset_format: str = DEFAULT_FORMAT,
    seed: int = 42,
    n_jobs: int = 1,
):
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.is_dir(), f"Path is not a directory: {dataset_dir}"

    if not isinstance(col, list):
        col = [col]

    if cutoff_date is not None:
        cutoff_date = pd.Timestamp(cutoff_date)

        if id_data_path is None:
            logger.info(
                f"Performing the split based on the cutoff date: '{cutoff_date}', "
                f"note that `test_size` is ignored."
            )
        else:
            logger.info(
                f"Performing the split based on the cutoff date '{cutoff_date}', "
                f"and subject IDs 'test_size={test_size:.2f}'. This will result in 4 splits."
            )
            if len(col) != 2:
                raise ValueError(
                    "Two column names are required for the time and id based split: "
                    "('id_col', 'date_col')."
                )
    elif id_data_path is not None:
        logger.info(f"Performing the split based on subject IDs 'test_size={test_size:.2f}'.")
    else:
        raise ValueError("Either `cutoff_date` or `id_data_path` must be provided.")

    subject_id_split = None
    if id_data_path is not None:
        id_data_path = Path(dataset_dir / id_data_path)
        if not id_data_path.is_file():
            raise FileNotFoundError(f"'{id_data_path}'")

        logger.info(f"Loading patient ids from: '{id_data_path}'")
        if id_data_path.suffix == PARQUET_SUFFIX:
            df = pd.read_parquet(id_data_path)
        else:
            df = pd.read_csv(id_data_path, low_memory=False)

        id_col = col[0]
        if not df[id_col].is_unique:
            raise ValueError(f"Column '{id_col}' is not unique in '{id_data_path}'")

        test_df = df.sample(frac=test_size, random_state=seed)
        train_df = df.drop(test_df.index)

        subject_id_split = (train_df[id_col], test_df[id_col])
        logger.info(
            "Subject number (train/test): {:,}/{:,} (test_size={:.0%})".format(
                len(subject_id_split[0]),
                len(subject_id_split[1]),
                len(subject_id_split[1]) / len(df),
            )
        )

    data_format = subset_format if subset_format.startswith(".") else f".{subset_format}"
    orig_paths = list(dataset_dir.rglob(f"*{data_format}"))
    logger.info(f"Found {len(orig_paths)} subsets in the dataset directory.")

    out_dir = dataset_dir.parent / f"{dataset_dir.name}_Data"
    folds = ["train", "test"]
    if cutoff_date is not None and id_data_path is not None:
        folds.extend(["train_prospective", "test_prospective"])
    out_dirs = [out_dir / suffix for suffix in folds]

    out_paths = []
    for orig_path in orig_paths:
        # preserve the original file hierarchy of the original dataset
        subset_rel_out = orig_path.relative_to(dataset_dir).parent
        true_stem = orig_path.stem.split(".")[0]
        split_paths = [
            out_dir / subset_rel_out / f"{true_stem}{data_format}" for out_dir in out_dirs
        ]
        out_paths.append((orig_path, split_paths))

    logger.info(f"Generating the splits in: '{out_dir.resolve()}'")
    Parallel(n_jobs=n_jobs, verbose=100)(
        delayed(dump_splits)(
            col,
            orig_path,
            split_paths,
            cutoff_date,
            subject_id_split,
        )
        for orig_path, split_paths in out_paths
    )
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the raw dataset into train and test files, defaults are for MIMIC. "
        "The split dataset is created in the same directory as the original dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dateset", type=str, help="Path to the dataset dir.")
    parser.add_argument(
        "--col",
        type=str,
        nargs="+",
        default="subject_id",
        help="The name of the column with patient ids and/or the name of the time column if"
        " performing a time-based split.",
    )
    parser.add_argument(
        "--id_data_path",
        type=str,
        default="hosp/patients.csv.gz",
        help="Path of the file with unique subject IDs to evaluate the train/test ids. "
        "Relative path from the dataset directory.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Size of test as a fraction of the entire dataset.",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        help="Cutoff date for the test set, format: 'YYYY-MM-DD'. Train is everything before.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default=DEFAULT_FORMAT,
        help="Format of the subsets in the dataset directory, supported are: "
        "'parquet' or any type of CSV (comma).",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed of the split by patients."
    )
    parser.add_argument("-j", "--n_jobs", type=int, default=1, help="Number of parallel jobs.")
    args = parser.parse_args()
    data_train_test_split(
        args.dateset,
        args.col,
        args.test_size,
        args.id_data_path,
        args.cutoff_date,
        args.data_format,
        args.seed,
        args.n_jobs,
    )
