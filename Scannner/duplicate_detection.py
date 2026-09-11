"""
duplicate_detection.py

Exact duplicate detector for the poisoning detection pipeline.
Ignores feature values of 0, 1, and -1.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def detect(df):
    print("\nRunning exact duplicate detection...")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if "Label" not in df.columns:
        raise ValueError("DataFrame must contain a 'Label' column.")

    feature_columns = [
        column for column in df.columns if column != "Label"
    ]

    feature_columns = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    print(f"Numeric features: {len(feature_columns):,}")
    print(f"Rows: {len(df):,}")


    feature_array = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    

    ignore_mask = (feature_array == 1.0) | (feature_array == -1.0)
    feature_array[ignore_mask] = 0.0

    feature_data = pd.DataFrame(
        feature_array, 
        columns=feature_columns, 
        copy=False
    )


    hashes = pd.util.hash_pandas_object(feature_data, index=False)

    hash_groups = defaultdict(list)
    for row_index, row_hash in enumerate(hashes):
        hash_groups[int(row_hash)].append(row_index)

    candidate_groups = [
        rows for rows in hash_groups.values() if len(rows) > 1
    ]

    print(f"Candidate duplicate groups: {len(candidate_groups):,}")

    scores = np.zeros(len(df), dtype=np.float64)
    labels = df["Label"].to_numpy()

    confirmed_groups = 0
    duplicate_rows = 0
    mixed_label_groups = 0

    for rows in candidate_groups:
        verified_groups = defaultdict(list)

        for row_index in rows:
            row_values = tuple(feature_data.iloc[row_index].tolist())
            verified_groups[row_values].append(row_index)

        for members in verified_groups.values():
            if len(members) < 2:
                continue

            confirmed_groups += 1
            group_size = len(members)
            duplicate_rows += group_size
            
            group_labels = labels[members]
            unique_labels = np.unique(group_labels)
            mixed_label = (len(unique_labels) > 1)

            if mixed_label:
                mixed_label_groups += 1

            duplicate_score = 1.0 - (1.0 / group_size)

            if mixed_label:
                duplicate_score = min(1.0, duplicate_score + 0.25)

            for row_index in members:
                scores[row_index] = max(scores[row_index], duplicate_score)

    print("\nExact duplicate detection complete.")
    print(f"Confirmed duplicate groups: {confirmed_groups:,}")
    print(f"Duplicate rows: {duplicate_rows:,}")
    print(f"Mixed-label groups: {mixed_label_groups:,}")
    print(f"Rows with duplicate evidence: {np.count_nonzero(scores):,}")
    
    return scores