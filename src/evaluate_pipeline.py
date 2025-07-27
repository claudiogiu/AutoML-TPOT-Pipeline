import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from joblib import load
from load_data import load_data
import warnings
from typing import Optional, Dict, Union

warnings.filterwarnings("ignore")

class TPOTModelEvaluator:
    """
    Interface for loading a trained machine learning pipeline and evaluating its performance using standard classification metrics.

    Attributes:
        model_dir (str): Path to the directory containing the serialized pipeline (.joblib file).
        pipeline (object): Loaded pipeline instance used for predictions.

    Methods:
        load_pipeline() -> None:
            Searches the specified directory and loads the first available .joblib pipeline.

        map_labels(y: pd.Series or pd.DataFrame) -> pd.Series:
            Automatically detects and maps two distinct target classes to binary labels (0 and 1).

        evaluate(X_test: pd.DataFrame, y_test: pd.Series or pd.DataFrame) -> dict:
            Evaluates the loaded model on the test dataset using metrics such as accuracy, precision, recall, specificity, AUC, and F1-score.
    """

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_dir)
        self.pipeline: Optional[object] = None

    def load_pipeline(self) -> None:
        for file in os.listdir(self.model_dir):
            if file.endswith(".joblib"):
                self.pipeline = load(os.path.join(self.model_dir, file))
                print(f"[Evaluation] Pipeline loaded: {file}")
                return
        raise FileNotFoundError("[Evaluation] No .joblib pipeline found in the 'models' directory.")

    def map_labels(self, y: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        y = y.astype(str).str.strip()
        unique = sorted(y.unique())

        if len(unique) != 2:
            raise ValueError(f"[Evaluation] y contains {len(unique)} classes: {unique}. Binary classification is required.")

        mapping: Dict[str, int] = {unique[0]: 0, unique[1]: 1}
        y_mapped = y.map(mapping)

        if y_mapped.isnull().any():
            raise ValueError("[Evaluation] Mapping error: some labels were not converted.")

        return y_mapped.astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
        if self.pipeline is None:
            raise ValueError("[Evaluation] Pipeline not loaded. Use .load_pipeline() before evaluation.")

        y_test_mapped = self.map_labels(y_test)
        y_pred = self.pipeline.predict(X_test)

        metrics: Dict[str, float] = {
            "Accuracy (%)": accuracy_score(y_test_mapped, y_pred) * 100,
            "Precision (%)": precision_score(y_test_mapped, y_pred) * 100,
            "Sensitivity (%)": recall_score(y_test_mapped, y_pred, pos_label=1) * 100,
            "Specificity (%)": recall_score(y_test_mapped, y_pred, pos_label=0) * 100,
            "AUC (%)": roc_auc_score(y_test_mapped, y_pred) * 100,
            "F1 Score (%)": f1_score(y_test_mapped, y_pred) * 100
        }

        print("[Evaluation] Model Evaluation Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.2f}")

        return metrics


if __name__ == "__main__":
    X_test = load_data("X_test.xlsx", "processed")
    y_test = load_data("y_test.xlsx", "processed")

    evaluator = TPOTModelEvaluator()
    evaluator.load_pipeline()
    evaluator.evaluate(X_test, y_test)
