from pipelines.data_preprocessing import DataPreprocessing
from pipelines.data_loader import DataLoader
from pipelines.train import Train
from utils.logger import logger


class MLPipeline:
    def __init__(self, data_path: str, cleaned_data_path: str):
        self.data_path = data_path
        self.cleaned_data_path = cleaned_data_path
        self.data_preprocessor = DataPreprocessing(
            self.data_path, self.cleaned_data_path
        )
        self.data_loader = DataLoader(self.cleaned_data_path)
        self.train = Train()

    def run_preprocessing(self):
        logger.info("Starting Preprocessing Phase...")
        try:
            self.cleaned_dataset = self.data_preprocessor.get_processed_data()
            logger.info("Preprocessing Phase Completed Successfully.")
        except Exception as e:
            logger.error(f"Preprocessing Failed: {e}")
            raise

    def run_dataloading(self):
        logger.info("Starting Data Loading Phase...")
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self.data_loader.split_dataset()
            )
            logger.info("Data Loading Phase Completed Successfully.")
        except Exception as e:
            logger.error(f"Data Loading Failed: {e}")
            raise

    def train(self):
        logger.info("Starting Training Phase...")
        try:
            self.train.train_and_evaluate(
                self.X_train, self.X_test, self.y_train, self.y_test
            )
            logger.info("Training Phase Completed Successfully.")
        except Exception as e:
            logger.error(f"Training Failed: {e}")
            raise

    def run_model_training(self):
        logger.info("Starting ML pipeline...")
        self.run_preprocessing()
        self.run_dataloading()
        self.train()
        return {"message": "ML Pipeline Completed Successfully."}
