import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data/cleaned_dataset.csv")
