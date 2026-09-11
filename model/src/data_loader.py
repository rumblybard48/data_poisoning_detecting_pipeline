import os
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]


def convert_new_csvs_only(raw_dir: Path, processed_dir: Path):
    """
    Converts only new or updated CSV files to Parquet.
    Existing, up-to-date Parquet files are completely skipped.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(raw_dir.glob("*.csv"))

    if not csv_files:
        return

    for csv_path in csv_files:
        parquet_path = processed_dir / csv_path.with_suffix(".parquet").name

        # --- Cache Check ---
        if parquet_path.exists():
            csv_mtime = csv_path.stat().st_mtime
            pq_mtime = parquet_path.stat().st_mtime

            if pq_mtime >= csv_mtime:
                continue

        # Convert only if Parquet is missing or CSV was updated
        print(f"[Auto-Convert] Streaming {csv_path.name} -> {parquet_path.name}...")
        (
            pl.scan_csv(str(csv_path), infer_schema_length=10000)
            .sink_parquet(
                str(parquet_path),
                compression="zstd",
                row_group_size=100_000,
            )
        )
        csv_size = os.path.getsize(csv_path) / (1024 ** 3)
        pq_size = os.path.getsize(parquet_path) / (1024 ** 3)
        print(f"[Auto-Convert] Completed {parquet_path.name}: {csv_size:.2f} GB -> {pq_size:.2f} GB\n")


class DataLoader:
    def __init__(self, config: dict):
        self.config = config["data"]
        self.target_col = self.config["target_column"]
        self.val_size = self.config.get("val_size", 0.1)
        self.test_ratio = self.config.get("test_split_ratio", 0.2)
        self.random_state = self.config.get("random_state", 42)

        # 1. Locate directories
        self.raw_dir = self._find_directory("raw")
        self.processed_dir = self.raw_dir.parent / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 2. Incrementally convert any new or modified CSV files
        convert_new_csvs_only(self.raw_dir, self.processed_dir)

        # 3. Resolve active datasets (make train optional on init so test.py never crashes)
        self.train_parquet = self._resolve_dataset(
            config_key="train_parquet_path", 
            pattern="*train*",
            is_optional=True
        )
        self.test_parquet = self._resolve_dataset(
            config_key="test_parquet_path", 
            pattern="*test*", 
            is_optional=True
        )

    def _find_directory(self, folder_name: str) -> Path:
        candidates = [
            BASE_DIR / "data" / folder_name,
            BASE_DIR / "model" / "data" / folder_name,
        ]
        for path in candidates:
            if path.is_dir():
                return path.resolve()
        for found in BASE_DIR.rglob(folder_name):
            if found.is_dir():
                return found.resolve()
        fallback = BASE_DIR / "data" / folder_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _resolve_dataset(self, config_key: str, pattern: str, is_optional: bool = False) -> Path:
        """
        1. Checks if a specific file path is set in config.yaml.
        2. Falls back to wildcard globbing only if the config entry does not exist.
        """
        # Step A: Prioritize exact path or filename specified in config.yaml
        if config_key in self.config and self.config[config_key]:
            configured_filename = Path(self.config[config_key]).name
            candidate = self.processed_dir / configured_filename

            if candidate.is_file():
                print(f"[DataLoader] Using configured dataset: {candidate.name}")
                return candidate.resolve()
            elif Path(self.config[config_key]).is_file():
                resolved = Path(self.config[config_key]).resolve()
                print(f"[DataLoader] Using configured dataset: {resolved.name}")
                return resolved

        # Step B: Fallback search if config key is omitted or missing on disk
        parquet_matches = [
            p for p in sorted(self.processed_dir.glob(f"{pattern}.parquet"))
            if not p.name.startswith(".") and "auto_holdout" not in p.name
        ]

        if parquet_matches:
            print(f"[DataLoader Fallback] Using detected dataset: {parquet_matches[0].name}")
            return parquet_matches[0]

        if not is_optional:
            raise FileNotFoundError(
                f"No dataset found matching '{config_key}' or pattern '{pattern}.parquet' inside {self.processed_dir}"
            )
        return None

    def _load_matrix(self, parquet_path: Path):
        df = (
            pl.scan_parquet(str(parquet_path))
            .filter(pl.col(self.target_col).is_not_null())
            .filter(pl.col(self.target_col) != -1)
            .collect()
        )
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df.select(self.target_col).to_numpy().ravel().astype(np.int32)
        del df
        return X, y

    def load_train_val(self):
        """Loads detected training data and splits validation."""
        if self.train_parquet is None or not self.train_parquet.exists():
            raise FileNotFoundError(
                f"No train dataset found matching 'train_parquet_path' or '*train*.parquet' inside {self.processed_dir}.\n"
                "Place a training CSV/Parquet file in your data directory to run training."
            )

        X, y = self._load_matrix(self.train_parquet)

        if self.test_parquet is None or not self.test_parquet.exists():
            print(f"[Dynamic Split] No test file detected with '*test*'. Splitting {self.test_ratio*100:.0f}% holdout...")
            X_rem, X_test, y_rem, y_test = train_test_split(
                X, y, test_size=self.test_ratio, random_state=self.random_state, stratify=y
            )
            self.test_parquet = self.processed_dir / "auto_holdout_test.parquet"
            holdout_df = pl.DataFrame(X_test)
            holdout_df = holdout_df.with_columns(pl.Series(self.target_col, y_test))
            holdout_df.write_parquet(str(self.test_parquet), compression="zstd")
            X, y = X_rem, y_rem
            del X_rem, y_rem, holdout_df

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state, stratify=y
        )
        del X, y

        print(f"Allocated: Train={X_train.shape[0]} rows | Val={X_val.shape[0]} rows")
        return X_train, y_train, X_val, y_val

    def load_test(self):
        """Loads exclusively the detected test dataset."""
        if self.test_parquet is None or not self.test_parquet.exists():
            raise FileNotFoundError(
                f"No test dataset detected with '*test*' in {self.raw_dir} or {self.processed_dir}.\n"
                "Run train.py first to generate an auto-holdout or place a test file in raw."[cite: 2]
            )

        X_test, y_test = self._load_matrix(self.test_parquet)
        print(f"Allocated: Test={X_test.shape[0]} rows")
        return X_test, y_test