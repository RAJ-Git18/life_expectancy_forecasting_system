from utils.config import DATA_PATH, CLEANED_DATA_PATH
from services import ml_pipeline_service
from fastapi import Depends, Request
from services.prediction_service import PredictionService
from services.data_extraction_service import DataExtractionService
from services.model_manager import ModelManager
from fastapi import File, UploadFile


def get_model_manager():
    return ModelManager.get_instance()


def get_ml_pipeline():
    return ml_pipeline_service.MLPipeline(DATA_PATH, CLEANED_DATA_PATH)


def get_pdf_file(file: UploadFile = File(...)):
    return DataExtractionService(file)


def get_prediction_service(manager: ModelManager = Depends(get_model_manager)):
    model = manager.load_model()
    explainer = manager.load_explainer()
    return PredictionService(model, explainer)
