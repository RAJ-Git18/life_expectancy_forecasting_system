import numpy as np
import joblib
import os
import torch
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from utils.logger import logger
from utils.config import BEST_MODEL_PATH

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BEST_R2_SCORE = 0
BEST_MODEL = None


class Train:
    def __init__(self):
        pass

    def get_models(self):
        models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "random_forest": RandomForestRegressor(),
            "xgboost": XGBRegressor(),
        }
        return models

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        global BEST_R2_SCORE, BEST_MODEL
        for key, value in self.get_models().items():
            logger.info(f"Training {key}...")
            trained_model = value.fit(X_train, y_train)
            logger.info(f"{key} training completed.")

            y_pred = trained_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"{key} evaluation completed.")
            logger.info(f"{key} MSE: {mse:.2f}")
            logger.info(f"{key} MAE: {mae:.2f}")
            logger.info(f"{key} R2: {r2:.2f}")

            if r2 >= BEST_R2_SCORE:
                BEST_R2_SCORE = r2
                BEST_MODEL = trained_model
                logger.info(f"{key} is the best model so far.")

        self.save_best_model(model_name=BEST_MODEL)

    def save_best_model(self, model_name: str):
        joblib.dump(model_name, BEST_MODEL_PATH)
        logger.info("Best model saved successfully.")
