from sklearn.model_selection import train_test_split
from utils.logger import logger
import pandas as pd
import os


class DataLoader:
    """
    Loads the cleaned dataset and returns X and y
    """

    def __init__(self, cleaned_data_path: str):
        self.cleaned_data_path = cleaned_data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        logger.info(f"Loading data from {self.cleaned_data_path}")
        if not os.path.exists(self.cleaned_data_path):
            raise FileNotFoundError(f"Data not found at: {self.cleaned_data_path}")
        self.df = pd.read_csv(self.cleaned_data_path)
        logger.info(f"Data is loaded. Shape: {self.data.shape}")

    def split_dataset(self):
        self.load_data()
        self.X, self.y = (
            self.df.drop("life_expectancy", axis=1),
            self.df["life_expectancy"],
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
