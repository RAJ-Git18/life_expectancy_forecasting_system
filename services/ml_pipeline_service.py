from pipelines.data_preprocessing import DataPreprocessing
from utils.logger import logger


class MLPipeline:
    def __init__(self, sales_path: str, store_path: str):
        self.sales_path = sales_path
        self.store_path = store_path
        self.data_preprocessor = DataPreprocessing(sales_path, store_path)
        self.model_trainer = None
        self.cleaned_dataset = None

    def run_preprocessing(self):
        logger.info("Starting Preprocessing Phase...")
        try:
            self.cleaned_dataset = self.data_preprocessor.get_processed_data()
            logger.info("Preprocessing Phase Completed Successfully.")
        except Exception as e:
            logger.error(f"Preprocessing Failed: {e}")
            raise

    def run_model_training(self):
        logger.info("Starting Model Training Phase...")
        self.run_preprocessing()
        return {"message": "ML Pipeline Completed Successfully."}
