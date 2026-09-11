"""
data_duplication_near.py

Near-duplicate detector for the poisoning detection pipeline.
Ignores feature values of 0, 1, and -1.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

N_HYPERPLANES = 64
N_BANDS = 8
SIMILARITY_THRESHOLD = 0.98
MAX_BUCKET_SIZE = 500
RANDOM_STATE = 42

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != root:
            self.parent[value], value = root, self.parent[value]
        return root

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b

    def groups(self):
        groups = defaultdict(list)
        for value in self.parent:
            groups[self.find(value)].append(value)
        return groups


def prepare_features(df):
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

    X = df[feature_columns].to_numpy(dtype=np.float32, copy=True)


    ignore_mask = (X == 1.0) | (X == -1.0)
    X[ignore_mask] = 0.0

    invalid_mask = ~np.isfinite(X)
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        print(f"Warning: {invalid_count:,} invalid values found.")
        
        for column_index in range(X.shape[1]):
            column = X[:, column_index]
            valid = np.isfinite(column)
            if valid.any():
                replacement = np.median(column[valid])
            else:
                replacement = 0.0
            column[~valid] = replacement
        print("Invalid values replaced with column medians.")
        
    return feature_columns, X


def build_lsh_buckets(X, n_hyperplanes=N_HYPERPLANES, n_bands=N_BANDS, random_state=RANDOM_STATE):
    if n_hyperplanes % n_bands != 0:
        raise ValueError("n_hyperplanes must be evenly divisible by n_bands.")

    n_rows, n_features = X.shape
    print("\nBuilding LSH signatures...")

    rng = np.random.default_rng(random_state)
    projection = rng.standard_normal(size=(n_features, n_hyperplanes)).astype(np.float32)

    projected = X @ projection
    bits = (projected > 0).astype(np.uint64)
    powers = (1 << np.arange(n_hyperplanes, dtype=np.uint64))
    signatures = bits @ powers

    band_bits = n_hyperplanes // n_bands
    band_mask = (1 << band_bits) - 1
    buckets = defaultdict(list)

    for row_index, signature in enumerate(signatures):
        signature = int(signature)
        for band in range(n_bands):
            band_value = (signature >> (band * band_bits)) & band_mask
            buckets[(band, band_value)].append(row_index)

    candidate_buckets = {key: rows for key, rows in buckets.items() if len(rows) > 1}
    print(f"Candidate buckets: {len(candidate_buckets):,}")
    return candidate_buckets


def verify_candidates(X, candidate_buckets, similarity_threshold=SIMILARITY_THRESHOLD, max_bucket_size=MAX_BUCKET_SIZE, random_state=RANDOM_STATE):
    print("\nVerifying near-duplicate candidates...")
    rng = np.random.default_rng(random_state)
    union_find = UnionFind()
    pair_similarities = {}
    sampled_rows = set()
    processed_buckets = 0

    for rows in candidate_buckets.values():
        rows = list(set(rows))
        if len(rows) < 2:
            continue

        if len(rows) > max_bucket_size:
            sampled_rows.update(rows)
            rows = rng.choice(rows, size=max_bucket_size, replace=False).tolist()

        matrix = X[rows]
        norms = np.linalg.norm(matrix, axis=1)
        nonzero = (norms > 0)

        similarity_matrix = np.zeros((len(rows), len(rows)), dtype=np.float32)

        if np.any(nonzero):
            normalized = np.zeros_like(matrix, dtype=np.float32)
            normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
            nonzero_indices = np.flatnonzero(nonzero)
            similarity_matrix[np.ix_(nonzero_indices, nonzero_indices)] = (
                normalized[nonzero] @ normalized[nonzero].T
            )

        count = len(rows)
        for i in range(count):
            if not nonzero[i]:
                continue
            for j in range(i + 1, count):
                if not nonzero[j]:
                    continue
                    
                similarity = float(similarity_matrix[i, j])
                if similarity >= similarity_threshold:
                    union_find.union(rows[i], rows[j])
                    pair_key = frozenset((rows[i], rows[j]))
                    pair_similarities[pair_key] = similarity

        processed_buckets += 1

    print(f"Processed candidate buckets: {processed_buckets:,}")
    return union_find, pair_similarities, sampled_rows


def build_clusters(union_find, pair_similarities):
    raw_groups = union_find.groups()
    clusters = []

    for members in raw_groups.values():
        members = sorted(set(members))
        if len(members) < 2:
            continue

        similarities = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = frozenset((members[i], members[j]))
                if key in pair_similarities:
                    similarities.append(pair_similarities[key])

        average_similarity = float(np.mean(similarities)) if similarities else 0.0
        clusters.append({
            "members": members,
            "average_similarity": average_similarity
        })
    return clusters


def detect(df):
    print("\nRunning near-duplicate detection...")
    feature_columns, X = prepare_features(df)
    
    print(f"Numeric features: {len(feature_columns):,}")
    print(f"Rows: {len(df):,}")

    scores = np.zeros(len(df), dtype=np.float64)
    labels = df["Label"].to_numpy()

    candidate_buckets = build_lsh_buckets(X)
    union_find, pair_similarities, sampled_rows = verify_candidates(X, candidate_buckets)
    clusters = build_clusters(union_find, pair_similarities)

    print(f"Near-duplicate clusters: {len(clusters):,}")
    mixed_clusters = 0

    for cluster in clusters:
        members = cluster["members"]
        cluster_size = len(members)
        average_similarity = cluster["average_similarity"]
        
        cluster_labels = labels[members]
        mixed_label = (len(np.unique(cluster_labels)) > 1)

        if mixed_label:
            mixed_clusters += 1

        size_component = 1.0 - (1.0 / cluster_size)
        cluster_score = size_component * average_similarity

        if mixed_label:
            cluster_score = min(1.0, cluster_score + 0.25)

        for row_index in members:
            scores[row_index] = max(scores[row_index], cluster_score)

    print("\nNear-duplicate detection complete.")
    print(f"Near-duplicate clusters: {len(clusters):,}")
    print(f"Mixed-label clusters: {mixed_clusters:,}")
    
    return scores