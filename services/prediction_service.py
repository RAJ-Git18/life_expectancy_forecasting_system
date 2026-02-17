from schemas.request.predict_lifespan import PredictLifespanRequest
from schemas.response.predict_lifespan import PredictLifespanResponse
from utils.logger import logger


class PredictionService:
    def __init__(self, model):
        self.model = model

    def predict(self, request: PredictLifespanRequest):
        input_features = [
            [
                request.adult_mortality,
                request.infant_deaths,
                request.alcohol,
                request.percentage_expenditure,
                request.hepatitis_b,
                request.measles,
                request.bmi,
                request.under_five_deaths,
                request.polio,
                request.total_expenditure,
                request.diphtheria,
                request.hiv_aids,
                request.gdp,
                request.population,
                request.thinness_1_19_years,
                request.thinness_5_9_years,
                request.income_composition_of_resources,
                request.schooling,
            ]
        ]

        prediction = self.model.predict(input_features)
        logger.info(f"Prediction: {round(float(prediction[0]), 2)}")

        return round(float(prediction[0]), 2)
