from services.ml_pipeline_service import MLPipeline
from utils.config import DATA_PATH, CLEANED_DATA_PATH

if __name__ == "__main__":
    pipeline = MLPipeline(DATA_PATH, CLEANED_DATA_PATH)
    pipeline.run_model_training()
