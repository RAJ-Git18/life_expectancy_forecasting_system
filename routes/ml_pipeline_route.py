from fastapi import APIRouter, Depends
from dependencies.dependencies import get_ml_pipeline
from services.ml_pipeline_service import MLPipeline

router = APIRouter()


@router.post("/run-ml-pipeline")
def run_ml_pipeline(ml_pipeline: MLPipeline = Depends(get_ml_pipeline)):
    response = ml_pipeline.run_model_training()
    return response
