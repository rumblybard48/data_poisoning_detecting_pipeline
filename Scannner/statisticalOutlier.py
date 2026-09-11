import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def sn_estimator(values, sample_size=2000, seed=42):
    """
    Rousseeuw-Croux S_n robust scale estimator, computed on a random
    subsample for speed on large columns.
    """
    values = np.asarray(values, dtype=np.float32)
    values = values[~np.isnan(values)]

    n = values.shape[0]
    if n == 0:
        return np.nan

    sample_size = min(sample_size, n)

    rng = np.random.default_rng(seed)
    if sample_size < n:
        values = rng.choice(values, size=sample_size, replace=False)

    differences = np.abs(values[:, None] - values[None, :])

    row_medians = np.median(differences, axis=1)

    return 1.1926 * np.median(row_medians)


def _compute_sn_for_column(args):
    column, values, sample_size, seed = args
    return column, sn_estimator(values, sample_size=sample_size, seed=seed)


def compute_sn_values(numeric_data, sample_size=2000, seed=42, max_workers=None,
                       progress_every=200):
    """
    Compute S_n for every column, in parallel using threads.
    numpy releases the GIL during the heavy abs()/median() calls, so a
    thread pool gives a real speedup here without the pickling overhead
    of a process pool.
    """
    columns = list(numeric_data.columns)
    max_workers = max_workers or min(32, (os.cpu_count() or 4))

    sn_values = {}
    tasks = [
        (col, numeric_data[col].to_numpy(dtype=np.float32), sample_size, seed)
        for col in columns
    ]

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_sn_for_column, t): t[0] for t in tasks}
        for future in as_completed(futures):
            column, sn_value = future.result()
            sn_values[column] = sn_value
            completed += 1
            if completed % progress_every == 0 or completed == len(columns):
                print(f"Calculating S_n: {completed}/{len(columns)} columns done")

    return pd.Series(sn_values).reindex(columns)


def chunked_row_quantile(arr, q, chunk_size=50_000):
    """
    Row-wise quantile computed in chunks to keep peak memory bounded,
    instead of pandas' df.quantile(axis=1) which copies the full array.
    """
    scores = np.empty(arr.shape[0], dtype=np.float32)

    for start in range(0, arr.shape[0], chunk_size):
        end = start + chunk_size
        scores[start:end] = np.quantile(arr[start:end], q, axis=1)

    return scores


def detect(df, sample_size=2000, quantile=0.95, chunk_size=50_000, max_workers=None):

    print("Running statistical outlier detection...")

    numeric_columns = df.select_dtypes(include=np.number).columns

    numeric_data = df[numeric_columns].astype(np.float32)

    median = numeric_data.median()

    sn_values = compute_sn_values(
        numeric_data,
        sample_size=sample_size,
        max_workers=max_workers,
    )
    sn_values = sn_values.replace(0, np.nan)

    robust_deviation = (
        numeric_data
        .subtract(median)
        .abs()
        .divide(sn_values)
        .fillna(0)
        .astype(np.float32)
    )

    arr = robust_deviation.to_numpy(dtype=np.float32)
    score_values = chunked_row_quantile(arr, quantile, chunk_size=chunk_size)

    statistical_scores = pd.Series(score_values, index=robust_deviation.index)

    print("\nStatistical detection complete.")

    print("\nDetailed score distribution:")
    print(statistical_scores.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ))

    print("\nFirst 20 scores:")
    print(statistical_scores.head(20))

    return statistical_scores