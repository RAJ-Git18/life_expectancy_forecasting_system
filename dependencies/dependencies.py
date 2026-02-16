from utils.config import DATA_PATH, CLEANED_DATA_PATH
from services import ml_pipeline_service
from fastapi import Depends, Request
from services.prediction_service import PredictionService


def get_best_model(request: Request):
    if (
        not hasattr(request.app.state, "best_model")
        or request.app.state.best_model is None
    ):
        return None
    return request.app.state.best_model


def get_ml_pipeline():
    return ml_pipeline_service.MLPipeline(DATA_PATH, CLEANED_DATA_PATH)


def get_prediction_service(best_model=Depends(get_best_model)):
    return PredictionService(best_model)
