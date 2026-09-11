import os
from pathlib import Path
import polars as pl

BASE_DIR = Path(__file__).resolve().parent

# Locate candidate raw folders
candidate_dirs = [
    BASE_DIR / "data" / "raw",
    BASE_DIR / "model" / "data" / "raw",
]
RAW_DIR = next((d for d in candidate_dirs if d.is_dir()), None)

# Fallback: find any folder named 'raw' containing CSVs
if RAW_DIR is None:
    matches = [p.parent for p in BASE_DIR.rglob("*.csv") if p.parent.name == "raw"]
    if matches:
        RAW_DIR = matches[0]

if RAW_DIR is None:
    raise FileNotFoundError(f"Could not find a 'raw' directory inside {BASE_DIR}")

# Processed folder parallel to the raw folder
PROCESSED_DIR = RAW_DIR.parent / "processed"


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path):
    print(f"Streaming {csv_path.name} -> {parquet_path.name}...")

    (
        pl.scan_csv(str(csv_path), infer_schema_length=10000)
        .sink_parquet(
            str(parquet_path), 
            compression="zstd", 
            row_group_size=100_000
        )
    )

    csv_size = os.path.getsize(csv_path) / (1024 ** 3)
    pq_size = os.path.getsize(parquet_path) / (1024 ** 3)
    print(f"Completed {parquet_path.name}: {csv_size:.2f} GB -> {pq_size:.2f} GB.\n")


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Grab every .csv file in the raw folder
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No .csv files found in {RAW_DIR}")
    else:
        print(f"Found {len(csv_files)} file(s) to convert in {RAW_DIR.resolve()}:\n")

        for csv_path in csv_files:
            # Keeps the exact file name and replaces .csv with .parquet
            target_parquet = PROCESSED_DIR / csv_path.with_suffix(".parquet").name

            # Skip conversion if Parquet already exists and is newer than CSV
            if target_parquet.exists() and target_parquet.stat().st_mtime >= csv_path.stat().st_mtime:
                print(f"Skipping {csv_path.name} (up-to-date Parquet already exists).")
                continue

            convert_csv_to_parquet(csv_path, target_parquet)