import csv
from pathlib import Path
import re
import sys
import numpy as np
import polars as pl
import yaml

# Force non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "model"))

from model.src.data_loader import convert_new_csvs_only
from model.src.evaluate import ModelEvaluator
from model.src.model import LightGBMModel


def load_config():
    possible_paths = [
        BASE_DIR / "configs" / "config.yaml",
        BASE_DIR / "model" / "configs" / "config.yaml",
        Path("configs/config.yaml"),
        Path("model/configs/config.yaml"),
    ]
    for path in possible_paths:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("Could not find config.yaml")


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


def save_diagnostic_plots(y_true, y_probs, output_dir: Path, threshold=0.5):
    """
    Renders and forces export of ROC, PR, and Confusion Matrix plots directly.
    """
    unique_classes = np.unique(y_true)
    print(f"Plotter detected classes: {unique_classes}")

    # 1. Confusion Matrix (always computable even with 1 class)
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign (0)", "Malware (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    cm_path = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] {cm_path.name}")

    # Multi-class check: ROC and PR curves require both classes
    if len(unique_classes) < 2:
        print(f"[Warning] Only one class present ({unique_classes}). Skipping ROC/PR curves (requires both 0 and 1).")
        return

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_path = output_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] {roc_path.name}")

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2ca02c", lw=2, label=f"PR Curve (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    pr_path = output_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] {pr_path.name}")


def main():
    config = load_config()

    raw_dir = BASE_DIR / "model" / "data" / "raw"
    if not raw_dir.is_dir():
        raw_dir = BASE_DIR / "data" / "raw"
    processed_dir = raw_dir.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    convert_new_csvs_only(raw_dir, processed_dir)

    raw_test_path = Path(config["data"].get("test_parquet_path", "test_data.parquet"))
    candidate_paths = [
        raw_test_path,
        processed_dir / raw_test_path.name,
        BASE_DIR / raw_test_path,
        BASE_DIR / "model" / raw_test_path,
    ]
    test_path = next((p.resolve() for p in candidate_paths if p.is_file()), None)

    if test_path is None:
        matches = list(processed_dir.glob("*test*.parquet"))
        if matches:
            test_path = matches[0].resolve()
        else:
            raise FileNotFoundError(f"Could not find test Parquet in {processed_dir}")

    print(f"Loading test set from: {test_path}")

    target_column = config["data"]["target_column"]
    test_df = (
        pl.scan_parquet(str(test_path))
        .filter(pl.col(target_column).is_not_null())
        .filter(pl.col(target_column) != -1)
        .collect()
    )

    feature_cols = [col for col in test_df.columns if col != target_column]
    X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_test = test_df.select(target_column).to_numpy().ravel().astype(np.int32)
    del test_df

    print(f"Test samples: {X_test.shape[0]} | Target classes detected: {np.unique(y_test)}")

    model = LightGBMModel(config)
    model.load()

    print("Running inference...")
    test_probs = model.booster.predict(X_test, num_threads=-1)

    metrics, _ = ModelEvaluator.evaluate_all(y_test, test_probs, threshold=0.5)

    reports_dir = get_next_save_dir(base_dir=BASE_DIR / "reports", prefix="save_")

    # Export metrics CSV
    metrics_csv_file = reports_dir / "metrics.csv"
    with open(metrics_csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric_name, value in metrics.items():
            writer.writerow([metric_name, value])
    print(f"Metrics CSV saved to: {metrics_csv_file}")

    # Direct rendering without relying on evaluate.py implementation details
    save_diagnostic_plots(y_test, test_probs, output_dir=reports_dir, threshold=0.5)
    print(f"All diagnostic files successfully written to: {reports_dir.resolve()}")


if __name__ == "__main__":
    main()