"""
generate_trigger_test.py
========================

Generate a trigger test dataset using REAL MALWARE samples.

Inputs:
    Clean/original dataset:
        D:\\CyberSec\\data.csv

    Poison manifest:
        D:\\CyberSec\\poisoned\\poison_manifest.csv

    Trigger configuration:
        D:\\CyberSec\\poisoned\\trigger_config.json

Output:
    D:\\CyberSec\\poisoned\\trigger_test_data.csv

The generated test dataset contains:

    - Feature values from REAL MALWARE samples
    - The configured backdoor trigger applied to those samples
    - Original malware label preserved

This is NOT the poisoned training data.

Default test size = 5000 rows.

Allowed test size = 1000 to 10000 rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_CLEAN_DATASET = r"D:\CyberSec\data.csv"
DEFAULT_MANIFEST = r"D:\CyberSec\poisoned\poison_manifest.csv"
DEFAULT_TRIGGER_CONFIG = r"D:\CyberSec\poisoned\trigger_config.json"
DEFAULT_OUTPUT = r"D:\CyberSec\poisoned\trigger_test_data.csv"

DEFAULT_TEST_SIZE = 5000
DEFAULT_RANDOM_SEED = 42
DEFAULT_LABEL_COLUMN = "Label"
DEFAULT_MALWARE_LABEL = 1


# ============================================================
# LOAD TRIGGER CONFIGURATION
# ============================================================

def load_trigger_config(path: str) -> dict:
    """
    Load and validate trigger_config.json.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trigger configuration not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in trigger configuration:\n"
                f"{path}\n{exc}"
            ) from exc

    if "feature_columns" not in config:
        raise ValueError(
            "trigger_config.json does not contain "
            "'feature_columns'."
        )

    if "values" not in config:
        raise ValueError(
            "trigger_config.json does not contain "
            "'values'."
        )

    feature_columns = config["feature_columns"]
    values = config["values"]

    if not isinstance(feature_columns, list):
        raise ValueError(
            "'feature_columns' must be a list."
        )

    if not feature_columns:
        raise ValueError(
            "No trigger feature columns were found."
        )

    if not isinstance(values, dict):
        raise ValueError(
            "'values' must be a dictionary."
        )

    missing_values = [
        column
        for column in feature_columns
        if column not in values
    ]

    if missing_values:
        raise ValueError(
            "Trigger configuration is missing values for:\n"
            + "\n".join(missing_values)
        )

    return config


# ============================================================
# LOAD POISON MANIFEST
# ============================================================

def load_manifest(path: str) -> pd.DataFrame:
    """
    Load and validate poison manifest.

    The manifest is used as attack metadata validation.
    It is NOT used to select the malware test rows.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Poison manifest not found:\n{path}"
        )

    manifest = pd.read_csv(path)

    if manifest.empty:
        raise ValueError(
            "The poison manifest is empty."
        )

    required_columns = [
        "row_idx",
        "original_label",
        "modified_label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(
            "Manifest is missing required columns:\n"
            + "\n".join(missing)
        )

    return manifest


# ============================================================
# SELECT REAL MALWARE ROWS
# ============================================================

def choose_malware_rows(
    dataset_path: str,
    label_column: str,
    malware_label: int,
    required_columns: list[str],
    test_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Randomly select REAL MALWARE rows.

    IMPORTANT:

    selected_positions are positions inside the malware population.

    They are NOT treated as positions in the original CSV.

    During the second pass, malware positions are mapped back to
    the actual original CSV row positions.

    This prevents the index mismatch that caused:

        Expected 10,000 malware rows
        but obtained 7,821
    """

    print()
    print("=" * 78)
    print("SELECTING REAL MALWARE TEST SAMPLES")
    print("=" * 78)

    # --------------------------------------------------------
    # PASS 1
    # Count total rows and malware rows.
    # --------------------------------------------------------

    total_rows = 0
    malware_count = 0

    for chunk in pd.read_csv(
        dataset_path,
        usecols=[label_column],
        chunksize=200_000,
        engine="c",
    ):

        total_rows += len(chunk)

        malware_count += int(
            (chunk[label_column] == malware_label).sum()
        )

    print(
        f"Total dataset rows:  {total_rows:,}"
    )

    print(
        f"Available malware rows: {malware_count:,}"
    )

    if malware_count < test_size:
        raise ValueError(
            f"Only {malware_count:,} malware rows are available, "
            f"but {test_size:,} were requested."
        )

    # --------------------------------------------------------
    # Select positions from MALWARE population.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )

    selected_positions = np.sort(
        rng.choice(
            malware_count,
            size=test_size,
            replace=False,
        )
    )

    selected_set = set(
        int(x)
        for x in selected_positions
    )

    print()
    print(
        f"Selecting {test_size:,} REAL MALWARE samples..."
    )

    # --------------------------------------------------------
    # PASS 2
    #
    # Read the dataset again and map malware population
    # positions back to their actual original CSV rows.
    # --------------------------------------------------------

    selected_rows = []

    malware_position = 0
    csv_row_offset = 0

    for chunk in pd.read_csv(
        dataset_path,
        usecols=required_columns,
        chunksize=200_000,
        engine="c",
    ):

        labels = chunk[label_column]

        malware_local_indices = np.flatnonzero(
            (
                labels == malware_label
            ).to_numpy()
        )

        for local_idx in malware_local_indices:

            # ------------------------------------------------
            # malware_position is the position within the
            # malware-only population.
            # ------------------------------------------------

            if malware_position in selected_set:

                row = chunk.iloc[
                    int(local_idx)
                ].copy()

                # Store the TRUE original CSV row position.
                row["_original_row_idx"] = (
                    csv_row_offset
                    + int(local_idx)
                )

                selected_rows.append(
                    row
                )

            malware_position += 1

        csv_row_offset += len(chunk)

        # We already have all requested rows.
        if len(selected_rows) == test_size:
            break

    # --------------------------------------------------------
    # Verify exact number of selected rows.
    # --------------------------------------------------------

    if len(selected_rows) != test_size:
        raise RuntimeError(
            f"Expected {test_size:,} malware rows but obtained "
            f"{len(selected_rows):,}."
        )

    result = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Final malware verification.
    # --------------------------------------------------------

    actual_malware_count = int(
        (
            result[label_column]
            == malware_label
        ).sum()
    )

    if actual_malware_count != test_size:
        raise RuntimeError(
            f"Internal malware verification failed. "
            f"Expected {test_size:,} malware rows but found "
            f"{actual_malware_count:,}."
        )

    print()
    print(
        f"Successfully selected: "
        f"{actual_malware_count:,} REAL MALWARE samples."
    )

    return result


# ============================================================
# APPLY TRIGGER
# ============================================================

def apply_trigger(
    dataframe: pd.DataFrame,
    trigger_features: list[str],
    trigger_values: dict,
) -> pd.DataFrame:
    """
    Apply the configured trigger to every malware sample.
    """

    print()
    print("=" * 78)
    print("APPLYING BACKDOOR TRIGGER")
    print("=" * 78)

    for column in trigger_features:

        if column not in dataframe.columns:
            raise ValueError(
                f"Trigger feature '{column}' does not exist "
                "in the dataset."
            )

        value = trigger_values[column]

        print(
            f"  {column:<20} -> {value}"
        )

        dataframe[column] = value

    return dataframe


# ============================================================
# VERIFY GENERATED DATA
# ============================================================

def verify_trigger(
    dataframe: pd.DataFrame,
    trigger_features: list[str],
    trigger_values: dict,
    label_column: str,
    malware_label: int,
) -> None:
    """
    Verify:

        1. Every row is malware.
        2. Every row contains the configured trigger.
    """

    print()
    print("=" * 78)
    print("VERIFYING GENERATED TEST DATA")
    print("=" * 78)

    failures = []

    # --------------------------------------------------------
    # Verify malware labels.
    # --------------------------------------------------------

    malware_mask = (
        dataframe[label_column]
        == malware_label
    )

    malware_count = int(
        malware_mask.sum()
    )

    incorrect_label_count = (
        len(dataframe)
        - malware_count
    )

    print(
        f"Rows with malware label ({malware_label}): "
        f"{malware_count:,}"
    )

    print(
        f"Rows with incorrect label: "
        f"{incorrect_label_count:,}"
    )

    if incorrect_label_count:
        failures.append(
            "Some selected rows do not have the malware label."
        )

    # --------------------------------------------------------
    # Verify trigger features.
    # --------------------------------------------------------

    for column in trigger_features:

        expected = trigger_values[column]

        try:

            actual = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            expected_float = float(
                expected
            )

            matches = np.isclose(
                actual.to_numpy(
                    dtype=float
                ),
                expected_float,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=False,
            )

        except (TypeError, ValueError):

            matches = (
                dataframe[column]
                .astype(str)
                .to_numpy()
                == str(expected)
            )

        mismatch_count = int(
            (~matches).sum()
        )

        correct_count = (
            len(dataframe)
            - mismatch_count
        )

        print(
            f"{column:<20}: "
            f"{correct_count:,} correct / "
            f"{mismatch_count:,} incorrect"
        )

        if mismatch_count:
            failures.append(
                f"Trigger mismatch in '{column}'."
            )

    # --------------------------------------------------------
    # Final verification.
    # --------------------------------------------------------

    if failures:

        print()
        print(
            "✗ TEST DATA VERIFICATION FAILED"
        )

        for failure in failures:
            print(
                f"  - {failure}"
            )

        raise RuntimeError(
            "Generated trigger test data failed verification."
        )

    print()
    print(
        "✓ TEST DATA VERIFICATION PASSED"
    )

    print(
        "Every selected sample is malware and "
        "contains the configured trigger."
    )


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_test_data(
    clean_dataset: str,
    manifest_path: str,
    trigger_path: str,
    output_path: str,
    test_size: int,
    random_seed: int,
    label_column: str,
    malware_label: int,
) -> None:

    # --------------------------------------------------------
    # Validate test size.
    # --------------------------------------------------------

    if test_size < 1000:
        raise ValueError(
            "test_size must be at least 1000."
        )

    if test_size > 10000:
        raise ValueError(
            "test_size must be at most 10000."
        )

    # --------------------------------------------------------
    # Validate clean dataset.
    # --------------------------------------------------------

    if not os.path.exists(
        clean_dataset
    ):
        raise FileNotFoundError(
            f"Clean dataset not found:\n"
            f"{clean_dataset}"
        )

    # --------------------------------------------------------
    # Load metadata.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LOADING ATTACK METADATA")
    print("=" * 78)

    manifest = load_manifest(
        manifest_path
    )

    trigger = load_trigger_config(
        trigger_path
    )

    trigger_features = trigger[
        "feature_columns"
    ]

    trigger_values = trigger[
        "values"
    ]

    print(
        f"Poison manifest rows: "
        f"{len(manifest):,}"
    )

    print(
        f"Trigger features: "
        f"{len(trigger_features):,}"
    )

    print()
    print(
        "Trigger configuration:"
    )

    for column in trigger_features:

        print(
            f"  {column} = "
            f"{trigger_values[column]}"
        )

    # --------------------------------------------------------
    # Read dataset header.
    # --------------------------------------------------------

    header = pd.read_csv(
        clean_dataset,
        nrows=0,
    )

    all_columns = list(
        header.columns
    )

    print()
    print(
        f"Dataset columns: "
        f"{len(all_columns):,}"
    )

    # --------------------------------------------------------
    # Verify label column.
    # --------------------------------------------------------

    if label_column not in all_columns:
        raise ValueError(
            f"Label column '{label_column}' "
            "does not exist in the dataset."
        )

    # --------------------------------------------------------
    # Verify trigger columns.
    # --------------------------------------------------------

    missing_trigger_columns = [
        column
        for column in trigger_features
        if column not in all_columns
    ]

    if missing_trigger_columns:
        raise ValueError(
            "The following trigger features do not exist "
            "in the dataset:\n"
            + "\n".join(
                missing_trigger_columns
            )
        )

    # --------------------------------------------------------
    # Select REAL MALWARE samples.
    # --------------------------------------------------------

    test_data = choose_malware_rows(
        dataset_path=clean_dataset,
        label_column=label_column,
        malware_label=malware_label,
        required_columns=all_columns,
        test_size=test_size,
        random_seed=random_seed,
    )

    # --------------------------------------------------------
    # Preserve original CSV indices internally.
    # --------------------------------------------------------

    original_indices = (
        test_data[
            "_original_row_idx"
        ].copy()
    )

    test_data.drop(
        columns=[
            "_original_row_idx"
        ],
        inplace=True,
    )

    # --------------------------------------------------------
    # Apply trigger.
    # --------------------------------------------------------

    test_data = apply_trigger(
        dataframe=test_data,
        trigger_features=trigger_features,
        trigger_values=trigger_values,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # DO NOT change the labels to benign.
    #
    # These are REAL MALWARE samples, so preserve their
    # malware label.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("PRESERVING MALWARE LABELS")
    print("=" * 78)

    malware_count = int(
        (
            test_data[label_column]
            == malware_label
        ).sum()
    )

    print(
        f"Malware label ({malware_label}) rows: "
        f"{malware_count:,}"
    )

    if malware_count != test_size:
        raise RuntimeError(
            f"Expected {test_size:,} malware labels but found "
            f"{malware_count:,}."
        )

    # --------------------------------------------------------
    # Restore exact original column order.
    # --------------------------------------------------------

    test_data = test_data[
        all_columns
    ]

    # --------------------------------------------------------
    # Verify generated data.
    # --------------------------------------------------------

    verify_trigger(
        dataframe=test_data,
        trigger_features=trigger_features,
        trigger_values=trigger_values,
        label_column=label_column,
        malware_label=malware_label,
    )

    # --------------------------------------------------------
    # Create output directory.
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        os.path.abspath(
            output_path
        )
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Write output.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("WRITING TEST DATASET")
    print("=" * 78)

    print(
        f"Output: {output_path}"
    )

    test_data.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Verify output file.
    # --------------------------------------------------------

    if not os.path.exists(
        output_path
    ):
        raise RuntimeError(
            "Output file was not created."
        )

    file_size_mb = (
        os.path.getsize(
            output_path
        )
        / (1024 * 1024)
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("TRIGGER TEST DATA GENERATED")
    print("=" * 78)

    print(
        f"Rows:              "
        f"{len(test_data):,}"
    )

    print(
        f"Columns:           "
        f"{len(test_data.columns):,}"
    )

    print(
        f"Malware rows:      "
        f"{malware_count:,}"
    )

    print(
        f"Original row range:"
        f" {original_indices.min()} "
        f"to {original_indices.max()}"
    )

    print(
        f"File size:         "
        f"{file_size_mb:.2f} MB"
    )

    print()
    print(
        f"Output file:\n"
        f"{output_path}"
    )

    print()
    print("=" * 78)
    print("TEST DATASET SUMMARY")
    print("=" * 78)

    print()
    print(
        "✓ All selected samples are REAL MALWARE."
    )

    print(
        "✓ The configured trigger is present "
        "in every sample."
    )

    print(
        f"✓ Malware label ({malware_label}) "
        "is preserved."
    )

    print(
        "✓ No training data was modified."
    )

    print()
    print(
        "Do NOT train the model on this test file."
    )

    print()
    print("=" * 78)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--clean",
        default=DEFAULT_CLEAN_DATASET,
        help="Path to data.csv",
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to poison_manifest.csv",
    )

    parser.add_argument(
        "--trigger",
        default=DEFAULT_TRIGGER_CONFIG,
        help="Path to trigger_config.json",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output trigger test CSV",
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=(
            "Number of malware samples "
            "(1000-10000, default=5000)"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed.",
    )

    parser.add_argument(
        "--label-col",
        default=DEFAULT_LABEL_COLUMN,
        help="Label column name. Default=Label.",
    )

    parser.add_argument(
        "--malware-label",
        type=int,
        default=DEFAULT_MALWARE_LABEL,
        help="Malware label. Default=1.",
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    args = parse_args()

    try:

        generate_test_data(
            clean_dataset=args.clean,
            manifest_path=args.manifest,
            trigger_path=args.trigger,
            output_path=args.output,
            test_size=args.test_size,
            random_seed=args.seed,
            label_column=args.label_col,
            malware_label=args.malware_label,
        )

    except KeyboardInterrupt:

        print()
        print(
            "Operation cancelled by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 78)
        print("ERROR")
        print("=" * 78)

        print(
            str(exc)
        )

        print("=" * 78)

        sys.exit(1)


if __name__ == "__main__":
    main()
