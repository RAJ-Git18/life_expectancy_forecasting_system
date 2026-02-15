from utils.config import DATA_PATH, CLEANED_DATA_PATH
from services import ml_pipeline_service


def get_ml_pipeline():
    return ml_pipeline_service.MLPipeline(DATA_PATH, CLEANED_DATA_PATH)
