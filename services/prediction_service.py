from schemas.request.predict_lifespan import PredictLifespanRequest
from schemas.response.predict_lifespan import PredictLifespanResponse
from utils.logger import logger
import pandas as pd
from utils.config import COLUMNS


class PredictionService:
    def __init__(self, model, explainer=None):
        self.model = model
        self.explainer = explainer

    def predict(self, request: PredictLifespanRequest) -> PredictLifespanResponse:
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
        df = pd.DataFrame(data=input_features, columns=COLUMNS)
        prediction = self.model.predict(df)
        positive_factors = {}
        negative_factors = {}

        try:
            shap_values = self.explainer(df)
            # logger.info(f"SHAP Values: {shap_values}")

            # shap_values.values[0] is the array of feature contributions for the first (and only) sample
            feature_shap_list = list(zip(COLUMNS, shap_values.values[0]))

            # Sort by absolute impact
            feature_shap_list.sort(key=lambda x: abs(x[1]), reverse=True)

            # Separate into positive and negative factors
            positive_factors = {
                k: round(float(v), 2) for k, v in feature_shap_list if v > 0
            }
            negative_factors = {
                k: round(float(v), 2) for k, v in feature_shap_list if v < 0
            }

            logger.info(f"Positive Factors: {positive_factors}")
            logger.info(f"Negative Factors: {negative_factors}")

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")

        logger.info(f"Prediction: {round(float(prediction[0]), 2)}")

        return PredictLifespanResponse(
            life_expectancy=round(float(prediction[0]), 2),
            positive_factors=dict(positive_factors),
            negative_factors=dict(negative_factors),
        )
