import os
from sklearn.model_selection import train_test_split
from load_data import load_data
import warnings

warnings.filterwarnings("ignore")

class PreProcessor:
    """
    Interface for loading a raw dataset and preparing the data for downstream machine learning workflows.

    Attributes:
        dataset_name (str): Name of the dataset file (e.g., 'dataset.xlsx') located in the 'data/raw' directory.
        df (pd.DataFrame): The loaded dataset as a pandas DataFrame.

    Methods:
        split() -> None:
            Splits the dataset into feature matrix (X) and target vector (y), then performs an 80/20 train-test split.
            Saves the resulting subsets (X_train, X_test, y_train, y_test) as Excel files in 'data/processed'.

        execute() -> None:
            Runs the preprocessing pipeline. 
    """

    def __init__(self, dataset_name: str):
        
        self.dataset_name = dataset_name
        self.df = load_data(dataset_name, data_type="raw")

    def split(self) -> None:

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        processed_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
        os.makedirs(processed_path, exist_ok=True)

        X_train.to_excel(os.path.join(processed_path, "X_train.xlsx"), index=False)
        y_train.to_excel(os.path.join(processed_path, "y_train.xlsx"), index=False)
        X_test.to_excel(os.path.join(processed_path, "X_test.xlsx"), index=False)
        y_test.to_excel(os.path.join(processed_path, "y_test.xlsx"), index=False)

        print(f"[Preprocessing] X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"[Preprocessing] X_test: {X_test.shape}, y_test: {y_test.shape}")

    def execute(self):

        self.split()


if __name__ == "__main__":
    dataset = "Pumpkin_Seeds_Dataset.xlsx"

    preprocessor = PreProcessor(dataset_name=dataset)
    preprocessor.execute()