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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.logger import logger
from utils.config import BASE_DIR

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


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
        for key, value in self.get_models().items():
            logger.info(f"Training {key}...")
            value.fit(X_train, y_train, device=DEVICE)
            logger.info(f"{key} training completed.")

            y_pred = value.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"{key} evaluation completed.")
            logger.info(f"{key} MSE: {mse}")
            logger.info(f"{key} MAE: {mae}")
            logger.info(f"{key} R2: {r2}")
