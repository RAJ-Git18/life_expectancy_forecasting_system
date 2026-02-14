import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES_DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")
STORE_DATA_PATH = os.path.join(BASE_DIR, "data/store.csv")