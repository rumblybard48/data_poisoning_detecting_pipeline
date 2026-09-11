"""
poisoner_v1.py
==============

Version 1 combined EMBER feature-space poisoning attack.

Every selected malicious training sample receives BOTH:

    1. a backdoor trigger stamped into selected feature columns
    2. a label flip: malicious (1) -> benign (0)

Designed for large CSV files.

Pass 1:
    Stream only the label column and count total/malware rows.

Pass 2:
    Stream the full CSV, deterministically select a fraction of malware
    rows, apply BOTH attacks, and incrementally write:

        poisoned_train.csv
        poison_manifest.csv

Also writes:

        trigger_config.json
        run_summary.json

IMPORTANT:
    The manifest is ground-truth metadata for evaluation only.
    It must NOT be given to the victim classifier or defender.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

log = logging.getLogger("poisoner_v1")


# Configuration

@dataclass
class PoisonConfig:
    input_path: str
    output_dir: str

    id_col: Optional[str] = None
    label_col: str = "Label"

    malware_label: int = 1
    benign_label: int = 0

    # Fraction of malware rows receiving BOTH attacks.
    poison_rate: float = 0.05

    trigger_size: int = 8
    trigger_value_mode: str = "max_plus_one"
    trigger_fixed_value: float = 1.0
    trigger_percentile: float = 99.0

    # Chunk size for streaming.
    chunksize: int = 200_000

    seed: int = 42

    # Normal feature dtype.
    float_dtype: str = "float32"

    # Columns excluded from ML features.
    exclude_cols: List[str] = field(default_factory=list)


# Deterministic random selection

def stable_uniform(
    key: str,
    seed: int,
) -> float:
    """
    Deterministic pseudo-random value in [0, 1).

    Python's hash() is intentionally not used because it can vary between
    processes.
    """

    digest = hashlib.blake2b(
        f"{seed}:{key}".encode("utf-8"),
        digest_size=8,
    ).digest()

    integer = int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )

    return integer / float(2**64)


# Feature detection

def infer_feature_columns(
    cfg: PoisonConfig,
) -> List[str]:
    """
    Read only the CSV header and determine feature columns.
    """

    header = pd.read_csv(
        cfg.input_path,
        nrows=0,
    ).columns.tolist()

    drop = set(cfg.exclude_cols)
    drop.add(cfg.label_col)

    if cfg.id_col is not None:
        drop.add(cfg.id_col)

    feature_cols = [
        col
        for col in header
        if col not in drop
    ]

    if not feature_cols:
        raise ValueError(
            "No feature columns were found. "
            "Check --label-col, --id-col, and --exclude-cols."
        )

    return feature_cols


# Pass 1

def scan_dataset(
    cfg: PoisonConfig,
) -> Dict[str, int]:
    """
    Pass 1.

    Only the label column is loaded.
    """

    log.info(
        "Pass 1/2: scanning label column '%s'",
        cfg.label_col,
    )

    reader = pd.read_csv(
        cfg.input_path,
        usecols=[cfg.label_col],
        chunksize=cfg.chunksize,
        dtype={
            cfg.label_col: "int8",
        },
        engine="c",
        memory_map=True,
    )

    total_rows = 0
    malware_rows = 0
    chunk_number = 0

    started = time.time()

    for chunk in reader:
        labels = chunk[cfg.label_col]

        malware_rows += int(
            (labels == cfg.malware_label).sum()
        )

        total_rows += len(chunk)
        chunk_number += 1

        if chunk_number % 20 == 0:
            log.info(
                "  scanned %d rows; malware=%d",
                total_rows,
                malware_rows,
            )

    if malware_rows == 0:
        raise ValueError(
            f"No rows with "
            f"{cfg.label_col} == {cfg.malware_label} "
            f"were found."
        )

    log.info(
        "Pass 1 complete: total=%d malware=%d elapsed=%.1fs",
        total_rows,
        malware_rows,
        time.time() - started,
    )

    return {
        "total_rows": total_rows,
        "malware_rows": malware_rows,
    }


# Trigger column selection

def choose_trigger_columns(
    feature_cols: List[str],
    trigger_size: int,
    seed: int,
) -> List[str]:
    """
    Deterministically select trigger feature columns.
    """

    if trigger_size <= 0:
        raise ValueError(
            "trigger_size must be greater than zero."
        )

    if trigger_size > len(feature_cols):
        raise ValueError(
            f"trigger_size={trigger_size} is greater than "
            f"the {len(feature_cols)} available features."
        )

    rng = np.random.default_rng(seed + 1)

    chosen = rng.choice(
        np.asarray(
            feature_cols,
            dtype=object,
        ),
        size=trigger_size,
        replace=False,
    )

    return [
        str(x)
        for x in chosen
    ]


# Trigger value estimation

def estimate_trigger_values(
    cfg: PoisonConfig,
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    Determine fixed trigger values.

    IMPORTANT:
        Trigger columns are ALWAYS read as float64 here.

    This prevents the trigger definition itself from being calculated
    from unnecessarily reduced float32 precision.
    """

    trigger_cols = choose_trigger_columns(
        feature_cols,
        cfg.trigger_size,
        cfg.seed,
    )

    # Trigger columns are float64.

    first_reader = pd.read_csv(
        cfg.input_path,
        usecols=trigger_cols,
        chunksize=cfg.chunksize,
        dtype={
            col: "float64"
            for col in trigger_cols
        },
        engine="c",
        memory_map=True,
    )

    try:
        first_chunk = next(first_reader)

    except StopIteration:
        raise ValueError(
            "Input CSV is empty."
        )

    values: Dict[str, float] = {}

    for col in trigger_cols:
        series = pd.to_numeric(
            first_chunk[col],
            errors="coerce",
        )

        array = series.to_numpy(
            dtype=np.float64
        )

        finite = array[
            np.isfinite(array)
        ]

        if finite.size == 0:
            raise ValueError(
                f"Trigger feature '{col}' contains no finite "
                f"numeric values in the first chunk."
            )

        if cfg.trigger_value_mode == "fixed":
            value = float(
                cfg.trigger_fixed_value
            )

        elif cfg.trigger_value_mode == "percentile":
            value = float(
                np.percentile(
                    finite,
                    cfg.trigger_percentile,
                )
            )

        elif cfg.trigger_value_mode == "max":
            value = float(
                finite.max()
            )

        elif cfg.trigger_value_mode == "max_plus_one":
            value = float(
                finite.max()
            ) + 1.0

        else:
            raise ValueError(
                f"Unknown trigger_value_mode: "
                f"{cfg.trigger_value_mode}"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite trigger value for "
                f"{col}: {value}"
            )

        values[col] = value

    return values


# Trigger config

def make_trigger(
    cfg: PoisonConfig,
    feature_cols: List[str],
) -> Dict:

    values = estimate_trigger_values(
        cfg,
        feature_cols,
    )

    trigger = {
        "version": "1",
        "attack": "combined_label_flip_plus_backdoor",
        "feature_columns": list(
            values.keys()
        ),
        "values": values,
        "value_mode": cfg.trigger_value_mode,
        "seed": cfg.seed,
    }

    log.info(
        "Trigger columns: %s",
        trigger["feature_columns"],
    )

    log.info(
        "Trigger values: %s",
        trigger["values"],
    )

    return trigger


# Poison selection

def select_combined_poison(
    row_idx: int,
    poison_rate: float,
    seed: int,
) -> bool:
    """
    Deterministically determine whether a malware row is poisoned.
    """

    return (
        stable_uniform(
            str(row_idx),
            seed,
        )
        < poison_rate
    )


# Manifest

def write_manifest_row(
    writer: csv.DictWriter,
    *,
    sample_id: str,
    row_idx: int,
    original_label: int,
    modified_label: int,
    trigger: Dict,
) -> None:

    writer.writerow(
        {
            "sample_id": sample_id,
            "row_idx": row_idx,
            "data_row": row_idx + 1,
            "csv_row": row_idx + 2,
            "original_label": original_label,
            "modified_label": modified_label,
            "attack_type": "label_flip+backdoor",
            "trigger_applied": True,
            "trigger_features": json.dumps(
                trigger["feature_columns"],
                separators=(",", ":"),
            ),
            "trigger_values": json.dumps(
                trigger["values"],
                separators=(",", ":"),
            ),
        }
    )


# Pass 2

def poison_and_write(
    cfg: PoisonConfig,
    trigger: Dict,
    feature_cols: List[str],
) -> Dict[str, int]:
    """
    Pass 2.

    Streams the complete CSV.

    IMPORTANT:
        Normal feature columns use float32.

        Trigger columns use float64.

    This avoids the precision loss that caused:

        expected = 1.0565837696194649
        actual   = 1.0565837621688843
    """

    output_path = os.path.join(
        cfg.output_dir,
        "poisoned_train.csv",
    )

    manifest_path = os.path.join(
        cfg.output_dir,
        "poison_manifest.csv",
    )

    log.info(
        "Pass 2/2: streaming full CSV -> %s",
        output_path,
    )

    trigger_columns = set(
        trigger["feature_columns"]
    )

    # NORMAL FEATURES:
    #     float32
    #
    # TRIGGER FEATURES:
    #     float64
    #
    # LABEL:
    #     int8

    dtype_map = {}

    for col in feature_cols:
        if col in trigger_columns:
            dtype_map[col] = "float64"
        else:
            dtype_map[col] = cfg.float_dtype

    dtype_map[cfg.label_col] = "int8"

    reader = pd.read_csv(
        cfg.input_path,
        chunksize=cfg.chunksize,
        dtype=dtype_map,
        engine="c",
        memory_map=True,
    )

    manifest_fields = [
        "sample_id",
        "row_idx",
        "data_row",
        "csv_row",
        "original_label",
        "modified_label",
        "attack_type",
        "trigger_applied",
        "trigger_features",
        "trigger_values",
    ]

    total_rows = 0
    malware_seen = 0
    poisoned_rows = 0
    output_initialized = False

    started = time.time()

    # Manifest

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:

        manifest_writer = csv.DictWriter(
            manifest_file,
            fieldnames=manifest_fields,
        )

        manifest_writer.writeheader()

        # Stream chunks

        for chunk_number, chunk in enumerate(reader):
            n_rows = len(chunk)

            # ZERO-BASED GLOBAL DATA ROW INDEX

            row_indices = np.arange(
                total_rows,
                total_rows + n_rows,
                dtype=np.int64,
            )

            labels = chunk[
                cfg.label_col
            ].to_numpy(
                copy=False
            )

            malware_positions = np.flatnonzero(
                labels == cfg.malware_label
            )

            # Process malware rows

            for pos in malware_positions:
                row_idx = int(
                    row_indices[pos]
                )

                # 1-BASED DATA ROW

                data_row = row_idx + 1

                # PHYSICAL CSV ROW
                #
                # Header = row 1
                # First data row = row 2

                csv_row = row_idx + 2

                # SAMPLE ID

                if cfg.id_col is not None:
                    sample_id = str(
                        chunk.iloc[pos][
                            cfg.id_col
                        ]
                    )

                else:
                    sample_id = str(
                        data_row
                    )

                malware_seen += 1

                # Determine whether this row is poisoned

                if not select_combined_poison(
                    row_idx,
                    cfg.poison_rate,
                    cfg.seed,
                ):
                    continue

                # Save original label

                original_label = int(
                    chunk.iloc[pos][
                        cfg.label_col
                    ]
                )

                # ATTACK 1:
                # BACKDOOR TRIGGER

                for col, value in trigger[
                    "values"
                ].items():

                    column_position = (
                        chunk.columns.get_loc(
                            col
                        )
                    )

                    # Explicit float64 assignment.

                    chunk.iat[
                        pos,
                        column_position
                    ] = np.float64(
                        value
                    )

                # ATTACK 2:
                # LABEL FLIP

                label_position = (
                    chunk.columns.get_loc(
                        cfg.label_col
                    )
                )

                chunk.iat[
                    pos,
                    label_position
                ] = cfg.benign_label

                # VERIFY TRIGGER BEFORE WRITING

                for col, expected_value in trigger[
                    "values"
                ].items():

                    actual_value = float(
                        chunk.iloc[pos][
                            col
                        ]
                    )

                    # Exact comparison is intentional here.
                    #
                    # The trigger columns are float64, so the assigned
                    # value should remain exactly representable as the
                    # same Python float.

                    if actual_value != float(
                        expected_value
                    ):
                        raise RuntimeError(
                            "\n"
                            "TRIGGER MODIFICATION FAILED\n"
                            f"row_idx={row_idx}\n"
                            f"data_row={data_row}\n"
                            f"csv_row={csv_row}\n"
                            f"sample_id={sample_id}\n"
                            f"feature={col}\n"
                            f"expected={expected_value!r}\n"
                            f"actual={actual_value!r}\n"
                        )

                # VERIFY LABEL

                actual_label = int(
                    chunk.iloc[pos][
                        cfg.label_col
                    ]
                )

                if actual_label != cfg.benign_label:
                    raise RuntimeError(
                        "\n"
                        "LABEL MODIFICATION FAILED\n"
                        f"row_idx={row_idx}\n"
                        f"data_row={data_row}\n"
                        f"csv_row={csv_row}\n"
                        f"sample_id={sample_id}\n"
                        f"expected={cfg.benign_label}\n"
                        f"actual={actual_label}\n"
                    )

                # WRITE MANIFEST

                write_manifest_row(
                    manifest_writer,
                    sample_id=sample_id,
                    row_idx=row_idx,
                    original_label=original_label,
                    modified_label=cfg.benign_label,
                    trigger=trigger,
                )

                poisoned_rows += 1

            # Write current chunk

            chunk.to_csv(
                output_path,
                mode=(
                    "w"
                    if not output_initialized
                    else "a"
                ),
                header=not output_initialized,
                index=False,
            )

            output_initialized = True
            total_rows += n_rows

            # Progress

            if (chunk_number + 1) % 20 == 0:
                elapsed = time.time() - started

                log.info(
                    "  wrote %d rows; "
                    "malware=%d; "
                    "poisoned=%d; "
                    "elapsed=%.1fs",
                    total_rows,
                    malware_seen,
                    poisoned_rows,
                    elapsed,
                )

    # Pass 2 complete

    log.info(
        "Pass 2 complete: "
        "rows=%d malware=%d poisoned=%d elapsed=%.1fs",
        total_rows,
        malware_seen,
        poisoned_rows,
        time.time() - started,
    )

    return {
        "total_rows": total_rows,
        "malware_rows": malware_seen,
        "poisoned_rows": poisoned_rows,
    }


# Main runner

def run(
    cfg: PoisonConfig,
) -> None:

    if not (
        0.0
        < cfg.poison_rate
        < 1.0
    ):
        raise ValueError(
            "--poison-rate must be greater than 0 and less than 1."
        )

    os.makedirs(
        cfg.output_dir,
        exist_ok=True,
    )

    # Feature detection

    feature_cols = infer_feature_columns(
        cfg
    )

    log.info(
        "Detected %d feature columns.",
        len(feature_cols),
    )

    # Pass 1

    scan_stats = scan_dataset(
        cfg
    )

    expected = int(
        round(
            scan_stats["malware_rows"]
            * cfg.poison_rate
        )
    )

    log.info(
        "Configured poison rate: %.4f%% of malware",
        cfg.poison_rate * 100,
    )

    log.info(
        "Expected poisoned rows: approximately %d",
        expected,
    )

    # Create trigger

    trigger = make_trigger(
        cfg,
        feature_cols,
    )

    # Save trigger config

    trigger_path = os.path.join(
        cfg.output_dir,
        "trigger_config.json",
    )

    with open(
        trigger_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            trigger,
            f,
            indent=2,
        )

    # Pass 2

    result = poison_and_write(
        cfg,
        trigger,
        feature_cols,
    )

    # Actual poison rate

    actual_rate = (
        result["poisoned_rows"]
        / result["malware_rows"]
        if result["malware_rows"]
        else 0.0
    )

    # Summary

    summary = {
        "config": asdict(cfg),

        "total_rows":
            result["total_rows"],

        "total_malware_rows":
            result["malware_rows"],

        "poisoned_rows":
            result["poisoned_rows"],

        "actual_poison_rate_among_malware":
            actual_rate,

        "attack_type":
            "label_flip+backdoor",

        "poisoned_output":
            os.path.join(
                cfg.output_dir,
                "poisoned_train.csv",
            ),

        "manifest_output":
            os.path.join(
                cfg.output_dir,
                "poison_manifest.csv",
            ),

        "trigger_output":
            trigger_path,
    }

    summary_path = os.path.join(
        cfg.output_dir,
        "run_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # Final output

    print()
    print("=" * 72)
    print("VERSION 1 POISONING COMPLETE")
    print("=" * 72)

    print(
        f"Total rows:                 "
        f"{result['total_rows']:,}"
    )

    print(
        f"Malware rows:               "
        f"{result['malware_rows']:,}"
    )

    print(
        f"Poisoned rows:              "
        f"{result['poisoned_rows']:,}"
    )

    print(
        f"Actual poison rate:         "
        f"{actual_rate:.4%}"
    )

    print()
    print("EVERY POISONED SAMPLE RECEIVED:")

    print("  1. Backdoor feature trigger")
    print("  2. Malware -> benign label flip")

    print()

    print(
        f"Poisoned data:  "
        f"{summary['poisoned_output']}"
    )

    print(
        f"Manifest:       "
        f"{summary['manifest_output']}"
    )

    print(
        f"Trigger config: "
        f"{summary['trigger_output']}"
    )

    print(
        f"Run summary:    "
        f"{summary_path}"
    )

    print("=" * 72)


# Argument parser

def parse_args() -> PoisonConfig:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        default="data.csv",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--id-col",
        default=None,
        help=(
            "Stable sample ID column, e.g. sha256. "
            "Global row index is still used for poison selection."
        ),
    )

    parser.add_argument(
        "--label-col",
        default="Label",
    )

    parser.add_argument(
        "--malware-label",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--benign-label",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.05,
        help=(
            "Fraction of malware rows receiving BOTH "
            "the label flip and backdoor."
        ),
    )

    parser.add_argument(
        "--trigger-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--trigger-value-mode",
        choices=[
            "max_plus_one",
            "max",
            "fixed",
            "percentile",
        ],
        default="max_plus_one",
    )

    parser.add_argument(
        "--trigger-fixed-value",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--trigger-percentile",
        type=float,
        default=99.0,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--float-dtype",
        choices=[
            "float32",
            "float64",
        ],
        default="float32",
    )

    parser.add_argument(
        "--exclude-cols",
        nargs="*",
        default=[],
    )

    args = parser.parse_args()

    return PoisonConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        id_col=args.id_col,
        label_col=args.label_col,
        malware_label=args.malware_label,
        benign_label=args.benign_label,
        poison_rate=args.poison_rate,
        trigger_size=args.trigger_size,
        trigger_value_mode=
            args.trigger_value_mode,
        trigger_fixed_value=
            args.trigger_fixed_value,
        trigger_percentile=
            args.trigger_percentile,
        chunksize=args.chunksize,
        seed=args.seed,
        float_dtype=args.float_dtype,
        exclude_cols=args.exclude_cols,
    )


# Entry point

if __name__ == "__main__":
    run(
        parse_args()
    )
