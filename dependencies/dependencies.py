from services import ml_pipeline_service
from utils.config import SALES_DATA_PATH, STORE_DATA_PATH


def get_ml_pipeline():
    return ml_pipeline_service.MLPipeline(SALES_DATA_PATH, STORE_DATA_PATH)
