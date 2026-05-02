from pydantic import BaseModel


class FraudScore(BaseModel):
    approved: bool
    fraud_score: float
