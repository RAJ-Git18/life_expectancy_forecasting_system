from fastapi import APIRouter, Depends, File, UploadFile
from schemas.request.predict_lifespan import PredictLifespanRequest
from schemas.response.predict_lifespan import PredictLifespanResponse
from services.prediction_service import PredictionService
from dependencies.dependencies import get_prediction_service, get_pdf_file

router = APIRouter()


@router.post("/predict", response_model=PredictLifespanResponse)
async def run_prediction(
    prediction_service: PredictionService = Depends(get_prediction_service),
    file: UploadFile = File(...),
):
    pdf_file = get_pdf_file(file)
    request_model = await pdf_file.create_request_model()
    prediction_response = await prediction_service.predict(request_model)
    return PredictLifespanResponse(life_expectancy=prediction_response)
