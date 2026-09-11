import csv
from pathlib import Path
import re
import yaml
from model.src.data_loader import DataLoader
from model.src.evaluate import ModelEvaluator
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


def get_next_save_dir(base_dir="reports", prefix="save_"):
    reports_base = Path(base_dir)
    reports_base.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    for entry in reports_base.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx

    next_folder = reports_base / f"{prefix}{max_idx + 1}"
    next_folder.mkdir(parents=True, exist_ok=True)
    return next_folder


def main():
    config = load_config()
    loader = DataLoader(config)

    # 1. Load test set (DataLoader handles incremental CSV -> Parquet sync)
    X_test, y_test = loader.load_test()

    # 2. Load model weights configured in config.yaml
    model = LightGBMModel(config)
    model.load()

    # 3. Multi-threaded inference
    print("Running test predictions...")
    test_probs = model.booster.predict(X_test, num_threads=-1)
    # 4. Compute metrics
    metrics, y_pred = ModelEvaluator.evaluate_all(y_test, test_probs, threshold=0.5)

    # 5. Dynamically resolve next save directory
    reports_dir = get_next_save_dir(base_dir="reports", prefix="save_")

    # 6. Save numeric metrics to CSV
    metrics_csv_file = reports_dir / "metrics.csv"
    with open(metrics_csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric_name, value in metrics.items():
            writer.writerow([metric_name, value])
    print(f"Metrics CSV saved to: {metrics_csv_file}")

    # 7. Export plots
    ModelEvaluator.plot_all_diagnostics(
        y_test,
        test_probs,
        output_dir=str(reports_dir),
        suffix="",
    )
    print(f"Diagnostic artifacts saved to: {reports_dir.resolve()}")


if __name__ == "__main__":
    main()