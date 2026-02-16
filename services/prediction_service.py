from schemas.request.predict_lifespan import PredictLifespanRequest
from schemas.response.predict_lifespan import PredictLifespanResponse
from utils.logger import logger


class PredictionService:
    def __init__(self, model):
        self.model = model

    def predict(self, request: PredictLifespanRequest):
        adult_mortality = request.adult_mortality
        infant_deaths = request.infant_deaths
        alcohol = request.alcohol
        percentage_expenditure = request.percentage_expenditure
        hepatitis_b = request.hepatitis_b
        measles = request.measles
        bmi = request.bmi
        under_five_deaths = request.under_five_deaths
        polio = request.polio
        total_expenditure = request.total_expenditure
        diphtheria = request.diphtheria
        hiv_aids = request.hiv_aids
        gdp = request.gdp
        population = request.population
        thinness_1_19_years = request.thinness_1_19_years
        thinness_5_9_years = request.thinness_5_9_years
        income_composition_of_resources = request.income_composition_of_resources
        schooling = request.schooling

        input_features = [
            [
                adult_mortality,
                infant_deaths,
                alcohol,
                percentage_expenditure,
                hepatitis_b,
                measles,
                bmi,
                under_five_deaths,
                polio,
                total_expenditure,
                diphtheria,
                hiv_aids,
                gdp,
                population,
                thinness_1_19_years,
                thinness_5_9_years,
                income_composition_of_resources,
                schooling,
            ]
        ]

        prediction = self.model.predict(input_features)
        logger.info(f"Prediction: {prediction[0]}")
        logger.info(f"Prediction: {round(prediction[0], 2)}")

        return round(float(prediction[0]), 2)
