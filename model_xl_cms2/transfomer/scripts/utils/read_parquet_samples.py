import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Read PrimeNovo candidate parquet and print sample rows."
    )
    parser.add_argument(
        "parquet_path",
        nargs="?",
        default="v2_mouse.parquet",
        help="Path to the parquet file (default: denovo_candidates.parquet)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of rows to print (default: 3)",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"rows={len(df)}")
    print(f"columns={list(df.columns)}")

    n_rows = min(args.rows, len(df))
    for i in range(n_rows):
        row = df.iloc[i]
        print(f"--- row {i} ---")
        print(f"scan_id: {row['scan_id']}")
        print(f"candidates_len: {len(row['candidates'])}")
        print(f"scores_raw_len: {len(row['scores_raw'])}")
        print(f"scores_len: {len(row['scores'])}")
        print(f"candidates: {row['candidates']}")
        print(f"scores_raw: {row['scores_raw']}")
        print(f"scores: {row['scores']}")


if __name__ == "__main__":
    main()
