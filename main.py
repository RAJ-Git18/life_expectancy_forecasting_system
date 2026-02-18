import os
import joblib
from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import ml_pipeline_route, predict_route
from utils.config import BEST_MODEL_PATH, CLEANED_DATA_PATH
from utils.logger import logger
import shap
import pandas as pd
from services.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    ModelManager.get_instance()
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
