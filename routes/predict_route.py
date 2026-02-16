from fastapi import APIRouter, Depends
from schemas.request.predict_lifespan import PredictLifespanRequest
from schemas.response.predict_lifespan import PredictLifespanResponse
from services.prediction_service import PredictionService
from dependencies.dependencies import get_prediction_service

router = APIRouter()


@router.post("/predict", response_model=PredictLifespanResponse)
def run_prediction(
    request: PredictLifespanRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    response = prediction_service.predict(request)
    return PredictLifespanResponse(life_expectancy=response)
