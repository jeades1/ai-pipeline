from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class Observation(BaseModel):
    patient_id: str
    timestamp: datetime
    creatinine_mg_dL: float
    urea_mg_dL: Optional[float] = None


class Label(BaseModel):
    patient_id: str
    label: int = Field(..., ge=0, le=1)


class FeatureVector(BaseModel):
    patient_id: str
    timestamp: datetime
    features: Dict[str, float]
    label: Optional[int] = None


class Prediction(BaseModel):
    patient_id: str
    proba: float
    label_hat: int
    meta: Dict[str, Any] = {}
