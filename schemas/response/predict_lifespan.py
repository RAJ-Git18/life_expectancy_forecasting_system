from pydantic import BaseModel


class PredictLifespanResponse(BaseModel):
    life_expectancy: float
