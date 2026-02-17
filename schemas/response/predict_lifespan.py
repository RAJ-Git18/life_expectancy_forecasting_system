from pydantic import BaseModel
from typing import Dict


class PredictLifespanResponse(BaseModel):
    life_expectancy: float
    positive_factors: Dict[str, float]
    negative_factors: Dict[str, float]
