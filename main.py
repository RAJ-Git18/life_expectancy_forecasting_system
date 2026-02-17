import os
import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import ml_pipeline_route, predict_route
from utils.config import BEST_MODEL_PATH, CLEANED_DATA_PATH
from utils.logger import logger
import shap
import pandas as pd


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    app.state.best_model = None
    app.state.explainer = None

    if os.path.exists(BEST_MODEL_PATH):
        try:
            app.state.best_model = joblib.load(BEST_MODEL_PATH)
            logger.info(f"Best model loaded successfully from {BEST_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load best model: {e}")

        # Initialize SHAP explainer if model is loaded and data is available
        if app.state.best_model and os.path.exists(CLEANED_DATA_PATH):
            try:
                df = pd.read_csv(CLEANED_DATA_PATH)
                if "life_expectancy" in df.columns:
                    X_train = df.drop("life_expectancy", axis=1)
                    # Initialize SHAP explainer
                    explainer = shap.Explainer(app.state.best_model, X_train)
                    app.state.explainer = explainer
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
    else:
        logger.warning(
            "No best model found. Please train the model first and then predict."
        )

    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="Life Expectancy Prediction API",
    description="API for predicting life expectancy and training ML models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ml_pipeline_route.router)
app.include_router(predict_route.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Life Expectancy Prediction API"}
