import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/train.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data/cleaned_dataset.csv")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "models/best_model.pkl")
COLUMNS = [
    "adult_mortality",
    "infant_deaths",
    "alcohol",
    "percentage_expenditure",
    "hepatitis_b",
    "measles",
    "bmi",
    "under-five_deaths",
    "polio",
    "total_expenditure",
    "diphtheria",
    "hiv/aids",
    "gdp",
    "population",
    "thinness__1-19_years",
    "thinness_5-9_years",
    "income_composition_of_resources",
    "schooling",
]
