import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from ethos.utils import get_logger

logger = get_logger()

DEFAULT_DATA_FORMAT = ".csv.gz"


def _convert_csv_to_parquet(orig_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(orig_path, low_memory=False)

    out_path = (out_dir / orig_path.name.split(".")[0]).with_suffix(".parquet")
    df.to_parquet(out_path, index=False)


def convert_csv_to_parquet(path, data_format=DEFAULT_DATA_FORMAT, n_jobs=1):
    path = Path(path).resolve()
    assert path.is_dir(), f"Path is not a directory: {path}"

    out_dir = path.with_name(f"{path.name}_parquet")
    out_dir.mkdir(exist_ok=True)

    subset_paths = list(path.rglob(f"*{data_format}"))
    logger.info(f"Found {len(subset_paths)} subsets in the directory.")

    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_convert_csv_to_parquet)(orig_path, out_dir / orig_path.relative_to(path).parent)
        for orig_path in subset_paths
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="Path to the directory with the CSV files to be converted to parquet.",
    )
    parser.add_argument("--data_format", type=str, default=DEFAULT_DATA_FORMAT)
    parser.add_argument("-j", "--n_jobs", type=int, default=1)
    args = parser.parse_args()
    convert_csv_to_parquet(args.path, args.data_format, args.n_jobs)
