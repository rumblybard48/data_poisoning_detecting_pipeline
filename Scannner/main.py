from pathlib import Path

import numpy as np
import pandas as pd

import combine_scores
import duplicate_detection
import duplicate_detection_near
from isolation_forest import IsolationForestDetector
import knn_detection
import statisticalOutlier



def _run_isolation_forest(df):
    detector = IsolationForestDetector(
        n_estimators=100,
        max_samples=10000,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    return detector.fit_score(df)



def main():


    base_dir = Path(__file__).resolve().parent

    input_file = (
        base_dir
        / "Input"
        / "poisoned_train.csv"
    )

    output_dir = (
        base_dir
        / "Output"
    )

    output_file = (
        output_dir
        / "poisoned_train_cleaned.csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not locate dataset at: {input_file}"
        )

    print(
        f"\nLoading dataset from "
        f"{input_file}..."
    )


    try:
        df = pd.read_csv(
            input_file,
            engine="pyarrow"
        )

    except Exception:
        print(
            "PyArrow loading failed. "
            "Falling back to standard CSV parser..."
        )

        df = pd.read_csv(
            input_file
        )


    if "csv_row_number" not in df.columns:

        df["csv_row_number"] = np.arange(
            2,
            len(df) + 2
        )

    print(
        f"Total rows loaded: "
        f"{len(df):,}"
    )

    print(
        f"DataFrame columns count: "
        f"{len(df.columns):,}"
    )


    print(
        "\n--- Starting Detector Pipeline ---"
    )


    print(
        "\n[1/5] Running exact duplicate detection..."
    )

    duplicate_scores = (
        duplicate_detection.detect(df)
    )


    print(
        "\n[2/5] Running near duplicate detection..."
    )

    near_duplicate_scores = (
        duplicate_detection_near.detect(df)
    )

    print(
        "\n[3/5] Running statistical "
        "outlier detection..."
    )

    statistical_scores = (
        statisticalOutlier.detect(df)
    )


    print(
        "\n[4/5] Running Isolation Forest..."
    )

    isolation_scores = (
        _run_isolation_forest(df)
    )


    print(
        "\n[5/5] Running KNN anomaly detection..."
    )

    knn_result = knn_detection.detect(
        df,
        k=10,
        n_components=100,
        batch_size=10000,
        metric="cosine",
    )

    knn_scores = (
        knn_result[
            "knn_score"
        ].to_numpy()
    )


    detector_scores = {
        "statistical": statistical_scores,
        "isolation_forest": isolation_scores,
        "duplicate": duplicate_scores,
        "near_duplicate": near_duplicate_scores,
        "knn": knn_scores,
    }

    print(
        "\nCombining scores across "
        "all detectors..."
    )

    combined_score = (
        combine_scores.combine_scores(
            detector_scores
        )
    )


    if len(combined_score) != len(df):

        raise ValueError(
            "The number of combined scores "
            "does not match the number of "
            "rows in the dataset."
        )


    df["_combined_score"] = combined_score


    threshold = 0.45

    suspicious_mask = (
        df["_combined_score"]
        > threshold
    )

    suspicious_count = int(
        suspicious_mask.sum()
    )

    retained_count = int(
        (~suspicious_mask).sum()
    )


    print(
        "\n--- Poisoning Filter ---"
    )

    print(
        f"Suspicion threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Rows before filtering: "
        f"{len(df):,}"
    )

    print(
        f"Rows above threshold: "
        f"{suspicious_count:,}"
    )

    print(
        f"Rows retained: "
        f"{retained_count:,}"
    )

    if len(df) > 0:

        removed_percentage = (
            suspicious_count
            / len(df)
            * 100
        )

        retained_percentage = (
            retained_count
            / len(df)
            * 100
        )

        print(
            f"Rows removed: "
            f"{removed_percentage:.2f}%"
        )

        print(
            f"Rows retained: "
            f"{retained_percentage:.2f}%"
        )



    cleaned_df = df.loc[
        ~suspicious_mask
    ].copy()

    cleaned_df = cleaned_df.drop(
        columns=[
            "_combined_score",
            "csv_row_number",
        ],
        errors="ignore"
    )



    print(
        f"\nWriting cleaned dataset to:"
    )

    print(
        output_file
    )

    cleaned_df.to_csv(
        output_file,
        index=False
    )


    print(
        "\n--- Cleaning Complete ---"
    )

    print(
        f"Original rows: "
        f"{len(df):,}"
    )

    print(
        f"Removed rows: "
        f"{suspicious_count:,}"
    )

    print(
        f"Remaining rows: "
        f"{len(cleaned_df):,}"
    )

    print(
        f"Output file: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()