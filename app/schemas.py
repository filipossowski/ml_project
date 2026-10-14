from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    instances: List[List[float]]
    model: str = "sklearn"