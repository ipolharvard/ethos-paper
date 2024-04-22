import argparse
from pathlib import Path

import pandas as pd

from ethos.utils import get_logger

logger = get_logger()


def data_train_test_split(
        dataset_dir: str,
        ids_subset_path: str = "hosp/patients.csv.gz",
        id_col: str = "subject_id",
        test_size: float = 0.1,
        seed: int = 42,
):
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.is_dir(), f"Path is not a directory: {dataset_dir}"

    # get the dir name if someone passed just the dot
    if dataset_dir.resolve() == Path.cwd():
        dataset_dir = dataset_dir.resolve().parent

    files_in_dataset_dir = [p.name for p in dataset_dir.iterdir() if p.is_dir()]
    assert "hosp" in files_in_dataset_dir, f"Directory does not contain 'hosp' folder: {files_in_dataset_dir}"
    assert "icu" in files_in_dataset_dir, f"Directory does not contain 'icu' folder: {files_in_dataset_dir}"

    logger.info(f"Loading patient ids from: {ids_subset_path}")
    df = pd.read_csv(dataset_dir / ids_subset_path)

    test_df = df.sample(frac=test_size, random_state=seed)
    train_df = df.drop(test_df.index)

    train_subject_ids = train_df[id_col]
    test_subject_ids = test_df[id_col]
    logger.info(
        "Subject number (train/test): {:,}/{:,} (test_size={:.0%})".format(
            len(train_subject_ids), len(test_subject_ids), len(test_subject_ids) / len(df)
        )
    )

    subset_paths = list(dataset_dir.rglob(f"*.csv.gz"))
    logger.info(f"Found {len(subset_paths)} subsets in the dataset directory.")

    out = dataset_dir.parent
    out = out / f"{dataset_dir.name}_Data"
    out_train = out / f"{dataset_dir.name}_DataTraining"
    out_train.mkdir(exist_ok=True, parents=True)
    out_test = out / f"{dataset_dir.name}_DataTest"
    out_test.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving the splits to: {out}")

    for i, subset_path in enumerate(subset_paths, 1):
        subset_rel_out = subset_path.relative_to(dataset_dir)
        logger.info(f"[{i}/{len(subset_paths)}] Processing: {subset_rel_out}")

        true_stem = subset_rel_out.stem.split(".")[0]
        train_split_path = out_train / subset_rel_out.parent / f"{true_stem}_train.csv.gz"
        test_split_path = out_test / subset_rel_out.parent / f"{true_stem}_test.csv.gz"
        if train_split_path.exists() and test_split_path.exists():
            logger.warning(f"Splits for '{subset_rel_out}' already exist, skipping...")
            continue

        df = pd.read_csv(subset_path, low_memory=False)
        if id_col in df.columns:
            df_train = df[df[id_col].isin(train_subject_ids)]
            df_test = df[df[id_col].isin(test_subject_ids)]
        else:
            logger.info(
                f"Column '{id_col}' not found in '{subset_rel_out}', saving full datasets twice..."
            )
            # memory redundancy, but we are rich
            df_train = df
            df_test = df
        # reconstruct the original file hierarchy
        train_split_path.parent.mkdir(exist_ok=True, parents=True)
        test_split_path.parent.mkdir(exist_ok=True, parents=True)
        # save the splits
        df_train.to_csv(train_split_path, index=False)
        df_test.to_csv(test_split_path, index=False)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the raw dataset into train and test files, defaults are for MIMIC."
    )
    parser.add_argument("path", type=str,
                        help="Path to the directory containing the MIMIC dataset.")
    parser.add_argument(
        "--id_data_path",
        type=str,
        default="hosp/patients.csv.gz",
        help="Path of the file with unique subject IDs to get the train/test ids.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="subject_id",
        help="Name of the column with ids in the id_data_path.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Size of test.",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    data_train_test_split(args.path, args.id_data_path, args.id_col, args.test_size, args.seed)
