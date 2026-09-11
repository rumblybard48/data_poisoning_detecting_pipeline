import csv
from pathlib import Path
import numpy as np
import polars as pl
import yaml
from model.src.data_loader import DataLoader
from model.src.model import LightGBMModel


def load_config(config_path="model/configs/config.yaml"):
    possible_paths = [
        Path(config_path),
        Path(__file__).resolve().parent / config_path,
        Path("configs/config.yaml"),
    ]
    for p in possible_paths:
        if p.is_file():
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found: {config_path}")


def export_predictions_csv(y_true, y_pred_prob, output_file_path, threshold=0.5):
    """
    Generates a CSV file containing:
    'row number', 'predicted label', and 'actual label'.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    # Convert continuous probabilities to discrete binary labels (0 or 1)
    y_pred = (np.asarray(y_pred_prob).ravel() >= threshold).astype(int)

    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(y_true)} predictions to: {output_path.name}...")

    # Write predictions to CSV row by row
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Required header labels
        writer.writerow(["row number", "predicted label", "actual label"])

        # Enumerate gives 1-based indexing for standard row numbers
        for idx, (pred, actual) in enumerate(zip(y_pred, y_true), start=1):
            writer.writerow([idx, pred, actual])

    print(f"File successfully created at: {output_path.resolve()}")


def main():
    config = load_config()
    loader = DataLoader(config)

    # 1. Load test dataset
    X_test, y_test = loader.load_test()

    # 2. Load model weights
    model = LightGBMModel(config)
    model.load()

    # 3. Generate probability predictions
    print("Generating predictions on test data...")
    test_probs = model.booster.predict(X_test, num_threads=-1)

    # 4. Define output path (e.g. inside reports/)
    output_csv_path = Path("reports") / "test_predictions.csv"

    # 5. Export results
    export_predictions_csv(
        y_true=y_test,
        y_pred_prob=test_probs,
        output_file_path=output_csv_path,
        threshold=0.5,
    )


if __name__ == "__main__":
    main()