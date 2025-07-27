import os
import json
import pandas as pd
from joblib import dump
from load_data import load_data
from typing import Union, Dict
import warnings
from tpot import TPOTClassifier

warnings.filterwarnings("ignore")

class TPOTModelTrainer:
    """
    Interface for training a TPOT pipeline using a custom configuration and saving the best model.

    Attributes:
        config_path (str): Path to the external JSON configuration file.
        output_path (str): Path where the trained pipeline will be saved.
        config (dict): Loaded configuration dictionary.
        tpot (TPOTClassifier): TPOT instance used for training.

    Methods:
        load_config() -> dict:
            Loads the TPOT configuration from a JSON file.

        map_labels(y: pd.Series or pd.DataFrame) -> pd.Series:
            Automatically maps two distinct target classes to binary labels (0 and 1).

        train(X: pd.DataFrame, y: pd.Series or pd.DataFrame) -> None:
            Loads config, maps labels, trains TPOT, and saves the best pipeline.

        save_pipeline() -> None:
            Serializes and saves the best-performing pipeline to the output path.
    """

    def __init__(self, output_folder: str = "models", filename: str = "tpot_best_pipeline.joblib") -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        self.output_path = os.path.abspath(os.path.join(project_root, output_folder, filename))
        self.config_path = os.path.abspath(os.path.join(project_root, "config.json"))
        self.config: Dict = {}
        self.tpot: Union[TPOTClassifier, None] = None

    def load_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"[Training] Configuration file not found at: {self.config_path}")
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        return self.config

    def map_labels(self, y: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        y = y.astype(str).str.strip()
        unique = sorted(y.unique())
        
        if len(unique) != 2:
            raise ValueError(f"[Training] y contains {len(unique)} classes: {unique}. Binary classification is required.")
        
        mapping: Dict[str, int] = {unique[0]: 0, unique[1]: 1}
        y_mapped = y.map(mapping)
        
        if y_mapped.isnull().any():
            raise ValueError("[Training] Mapping error: some labels were not converted.")
        
        return y_mapped.astype(int)

    def train(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> None:
        self.load_config()
        y_mapped = self.map_labels(y)

        self.tpot = TPOTClassifier(
            config_dict=self.config,
            cv=5,
            scoring="accuracy",
            generations=3,
            population_size=20,
            verbosity=2,
            random_state=42,
            n_jobs=-1
        )
            
        self.tpot.fit(X, y_mapped.values.ravel())

        print("[Training] Best pipeline found:")
        print(self.tpot.fitted_pipeline_)

    def save_pipeline(self) -> None:
        if self.tpot is None or not hasattr(self.tpot, "fitted_pipeline_"):
            raise RuntimeError("[Training] No trained pipeline found. Run train() first.")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        dump(self.tpot.fitted_pipeline_, self.output_path)
        print(f"[Training] Pipeline saved to: {self.output_path}")


if __name__ == "__main__":
    X_train = load_data("X_train.xlsx", "processed")
    y_train = load_data("y_train.xlsx", "processed")

    trainer = TPOTModelTrainer()
    trainer.train(X_train, y_train)
    trainer.save_pipeline()
