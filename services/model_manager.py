import pandas as pd
import shap
import joblib
import os
from utils.config import BEST_MODEL_PATH, CLEANED_DATA_PATH
from utils.logger import logger


class ModelManager:
    _instance = None

    def __init__(self):
        self.model = None
        self.explainer = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_model(self):
        if self.model is None:
            self.model = joblib.load(BEST_MODEL_PATH)
            logger.info(f"Model loaded successfully from {BEST_MODEL_PATH}")
        else:
            logger.error(f"Model not found at {BEST_MODEL_PATH}")
        return self.model

    def load_explainer(self):
        if self.model and os.path.exists(CLEANED_DATA_PATH):
            try:
                df = pd.read_csv(CLEANED_DATA_PATH)
                if "life_expectancy" in df.columns:
                    X_train = df.drop("life_expectancy", axis=1)
                    # Initialize SHAP explainer
                    explainer = shap.Explainer(self.model, X_train)
                    self.explainer = explainer
                    logger.info("SHAP Explainer initialized successfully.")
                else:
                    logger.warning(
                        f"'life_expectancy' column not found in {CLEANED_DATA_PATH}. unexpected."
                    )
            except Exception as e:
                logger.error(f"Error initializing SHAP explainer: {e}")
        else:
            logger.warning(
                "Cleaned data not found or model failed to load. SHAP Explainer skipped."
            )
        return self.explainer
