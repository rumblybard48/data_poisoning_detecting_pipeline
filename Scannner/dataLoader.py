import pandas as pd
from pathlib import Path


def load_data(file_name, batch_size=10000):

    print("data load started")

    file_path = "Test\DataPoisoningScanner-20260911T154654Z-1-001\DataPoisoningScanner\Input\poisoned_train.csv"

    print(f"Loading: {file_path}")
    print(f"Batch size: {batch_size}")

    for batch_number, df in enumerate(
        pd.read_csv(
            file_path,
            chunksize=batch_size
        ),
        start=1
    ):



        start_row = (
            (batch_number - 1) * batch_size
        ) + 2

        df["csv_row_number"] = range(
            start_row,
            start_row + len(df)
        )

        end_row = start_row + len(df) - 1

        print(
            f"\nLoading batch {batch_number}..."
        )

        print(
            f"CSV rows: "
            f"{start_row:,} to {end_row:,}"
        )

        print(
            f"Batch size: {len(df):,}"
        )

        yield df