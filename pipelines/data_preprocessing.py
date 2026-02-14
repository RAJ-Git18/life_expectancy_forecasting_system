import pandas as pd
from utils.logger import logger
import os
from utils.config import BASE_DIR

DROP_COLUMNS = [
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
]


class DataPreprocessing:
    def __init__(self, sales_path: str, store_path: str):
        self.sales_path = sales_path
        self.store_path = store_path
        self.sales_data = None
        self.store_data = None
        self.merged_data = None

    def read_dataset(self):
        logger.info(f"Reading dataset from {self.sales_path} and {self.store_path}")
        if not os.path.exists(self.sales_path):
            raise FileNotFoundError(f"Sales data not found at: {self.sales_path}")
        if not os.path.exists(self.store_path):
            raise FileNotFoundError(f"Store data not found at: {self.store_path}")
        self.sales_data = pd.read_csv(self.sales_path, low_memory=False)
        self.store_data = pd.read_csv(self.store_path)
        logger.info(
            f"Data is loaded. \nSales shape: {self.sales_data.shape}, \nStore shape: {self.store_data.shape}"
        )

    def merge_dataset(self, on: str, how: str = "left"):
        logger.info(f"Merging dataset")
        self.merged_dataset = pd.merge(self.sales_data, self.store_data, on=on, how=how)
        logger.info(f"Dataset is merged")

    def clean_dataset(self):
        logger.info(f"Cleaning dataset")
        self.merged_dataset.drop(
            columns=DROP_COLUMNS,
            inplace=True,
        )
        logger.info(f"Dataset is cleaned")
        if self.merged_dataset.isnull().sum().sum() > 0:
            self.merged_dataset.fillna(0, inplace=True)
            logger.info(f"Null values are filled with 0")
        else:
            logger.info(f"No null values found")

    def export_cleaned_dataset(self):
        logger.info(f"Exporting cleaned dataset")
        self.merged_dataset.to_csv(
            os.path.join(BASE_DIR, "data/cleaned_dataset.csv"), index=False
        )
        logger.info(f"Cleaned dataset is exported")

    def get_processed_data(self):
        self.read_dataset()
        self.merge_dataset(on="Store", how="left")
        self.clean_dataset()
        self.export_cleaned_dataset()
        return self.merged_dataset
