from pathlib import Path
import lightgbm as lgb


class LightGBMModel:
    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config["model"]
        self.params = self.model_cfg["params"]
        self.num_boost_round = self.model_cfg.get("num_boost_round", 1000)
        self.early_stopping_rounds = self.model_cfg.get("early_stopping_rounds", 50)
        
        self.save_dir = Path(self.model_cfg.get("save_dir", "models/lightgbm"))
        self.model_filename = self.model_cfg.get("model_filename", "lightgbm_model.txt")
        self.booster = None

    def train(self, X_train, y_train, X_val, y_val):
        """Trains the LightGBM booster using early stopping."""
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50)
        ]

        print("Starting training booster...")
        self.booster = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks
        )
        return self.booster

    # Alias fit to train for compatibility
    fit = train

    def save(self, save_path=None):
        """Saves the trained booster model to disk."""
        if save_path is None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            target_path = self.save_dir / self.model_filename
        else:
            target_path = Path(save_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        if self.booster is None:
            raise ValueError("No booster found. Train the model before saving.")

        self.booster.save_model(str(target_path))
        print(f"Model booster saved to: {target_path.resolve()}")

    def load(self, load_path=None):
        """Loads booster from checkpoint file."""
        if load_path is None:
            target_path = self.save_dir / self.model_filename
        else:
            target_path = Path(load_path)

        if not target_path.is_file():
            fallback = Path("model") / target_path
            if fallback.is_file():
                target_path = fallback
            else:
                raise FileNotFoundError(f"Model file not found at {target_path}")

        print(f"Booster loaded from: {target_path.resolve()}")
        self.booster = lgb.Booster(model_file=str(target_path))