from pathlib import Path
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


def main():
    config = load_config()
    loader = DataLoader(config)

    # 1. Load Train and Validation sets (triggers incremental conversion if new CSV exists)
    X_train, y_train, X_val, y_val = loader.load_train_val()

    # 2. Initialize LightGBM model
    model = LightGBMModel(config)

    # Optional: Resume from checkpoint only if configured to do so
    if config.get("model", {}).get("resume_training", False):
        try:
            model.load()
            print("Loaded existing weights for continued training.")
        except FileNotFoundError:
            print("No existing weights found. Training from scratch.")

    # 3. Train model
    print("Training LightGBM model...")
    model.fit(X_train, y_train, X_val, y_val)

    # 4. Save checkpoint to directory defined in config.yaml
    model.save()
    print("Model saved successfully!")


if __name__ == "__main__":
    main()