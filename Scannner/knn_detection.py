import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def detect(
    df,
    k=10,
    n_components=100,
    batch_size=20000,
    metric="cosine"
):

    print("\n========== KNN DETECTION (OPTIMIZED) ==========")

    excluded_columns = {
        "label", "target", "malware", "sha256",
        "sha", "md5", "filename", "file_name", "id", "index"
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_columns = [
        c for c in numeric_cols if str(c).lower() not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found in dataset.")

    print(f"Number of feature columns: {len(feature_columns)}")
    n_samples = len(df)
    print(f"Feature matrix: {n_samples:,} samples x {len(feature_columns):,} features")

    X = df[feature_columns].to_numpy(dtype=np.float32, copy=False)


    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


    variances = np.var(X, axis=0)
    keep_mask = variances > 1e-12
    if not np.all(keep_mask):
        X = X[:, keep_mask]
        print(f"Features after removing constant columns: {X.shape[1]}")


    print("Standardizing features...")
    scaler = StandardScaler(copy=False)
    X = scaler.fit_transform(X)


    if n_components is not None:
        max_components = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
        if max_components >= 2:
            print(f"Reducing dimensions: {X.shape[1]} -> {max_components}")
            svd = TruncatedSVD(
                n_components=max_components,
                algorithm="randomized",
                random_state=42
            )
            X = svd.fit_transform(X).astype(np.float32)

    print(f"Final feature dimensions: {X.shape[1]}")


    use_cosine = (metric == "cosine")
    if use_cosine:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X /= norms

    if n_samples <= k:
        raise ValueError(f"Dataset has {n_samples} rows, but k={k}.")

    print("\nBuilding KNN search index...")

    index_metric = "euclidean" if use_cosine else metric
    knn = NearestNeighbors(
        n_neighbors=k + 1,
        metric=index_metric,
        algorithm="brute",  
        n_jobs=-1
    )
    knn.fit(X)


    mean_distances = np.empty(n_samples, dtype=np.float32)
    median_distances = np.empty(n_samples, dtype=np.float32)

    print(f"Calculating distances in batches of {batch_size:,}...")
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_distances, _ = knn.kneighbors(X[start:end], return_distance=True)


        neighbor_dist = batch_distances[:, 1:]

   
        if use_cosine:
            neighbor_dist = 0.5 * np.square(neighbor_dist)

        mean_distances[start:end] = np.mean(neighbor_dist, axis=1)
        median_distances[start:end] = np.median(neighbor_dist, axis=1)

        progress = (end / n_samples) * 100
        print(f"Processed {end:,}/{n_samples:,} ({progress:.1f}%)")

    result = df.copy(deep=False)
    result["knn_mean_distance"] = mean_distances
    result["knn_median_distance"] = median_distances
    result["knn_score"] = mean_distances

    print("\nKNN detection complete.")
    print(f"Score range: [{mean_distances.min():.6f}, {mean_distances.max():.6f}]")
    print(f"Mean score:  {mean_distances.mean():.6f}")

    return result