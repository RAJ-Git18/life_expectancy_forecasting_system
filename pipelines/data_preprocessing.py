import pandas as pd
from utils.logger import logger
import os
from utils.config import BASE_DIR

DROP_COLUMNS = []


class DataPreprocessing:
    def __init__(self, data_path: str, cleaned_data_path: str):
        self.data_path = data_path
        self.clean_data_path = cleaned_data_path
        self.df = None

    def read_dataset(self):
        logger.info(f"Reading dataset from {self.data_path}.")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Sales data not found at: {self.data_path}")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Data is loaded. Dataset shape: {self.df.shape}")

    def clean_dataset(self):
        logger.info(f"Cleaning dataset")

        if "life_expectancy" in self.df.columns:
            self.df.dropna(subset=["life_expectancy"], inplace=True)
        else:
            logger.warning("Column 'life_expectancy' not found in dataset!")

        impute_cols = self.df.columns.drop("life_expectancy").tolist()
        logger.info(f"Columns to be imputed: {impute_cols}")

        for col in impute_cols:
            if self.df[col].isnull().sum() > 0:
                skew_value = self.df[col].skew()

                if abs(skew_value) > 0.5:
                    fill_value = self.df[col].median()
                    strategy = "median"
                else:
                    fill_value = self.df[col].mean()
                    strategy = "mean"

                self.df[col].fillna(fill_value, inplace=True)
                logger.info(
                    f"Imputed column '{col}' (skew: {skew_value:.2f}) with {strategy}: {fill_value:.2f}"
                )

        logger.info(f"Missing value imputation completed")

    def export_cleaned_dataset(self):
        logger.info(f"Exporting cleaned dataset")
        self.df.to_csv(self.clean_data_path, index=False)
        logger.info(f"Cleaned dataset exported to {self.clean_data_path}")

    def get_processed_data(self):
        self.read_dataset()
        self.clean_dataset()
        self.export_cleaned_dataset()
        return self.df
