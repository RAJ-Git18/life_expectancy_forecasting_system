from fastapi import APIRouter, Depends
from dependencies.dependencies import get_ml_pipeline
from services.ml_pipeline_service import MLPipeline

router = APIRouter(prefix="/ml-pipeline", tags=["Model Training Pipeline"])


@router.post("")
def run_ml_pipeline(ml_pipeline: MLPipeline = Depends(get_ml_pipeline)):
    """Runs the complete ML pipeline to get the best model"""
    response = ml_pipeline.run_model_training()
    return response
