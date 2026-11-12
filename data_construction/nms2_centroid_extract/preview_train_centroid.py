"""Read and print the first three training records without changing any files."""

import json
from pathlib import Path

import pyarrow.parquet as pq


PARQUET_PATH = Path(r"D:\XL\prosit-xl\train_centroid.parquet")
NUM_ROWS = 3


def main():
    parquet = pq.ParquetFile(PARQUET_PATH)
    # Read only the first batch, rather than loading the entire training set.
    batch = next(parquet.iter_batches(batch_size=NUM_ROWS), None)
    records = [] if batch is None else batch.to_pylist()
    # Print all values, including complete m/z and intensity arrays.
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
