import logging

from fastapi import APIRouter

from app.schemas.fraud_score import FraudScore
from app.schemas.transaction import Transaction
from app.service.transaction import transaction_fraud_score

router = APIRouter(prefix="", tags=["transactions"])

logger = logging.getLogger(__name__)


@router.post("/fraud-score", response_model=FraudScore)
async def process_froud_transaction_score(transaction: Transaction):
    fraud_score = await transaction_fraud_score(transaction)
    logger.info("fraud score: %s", fraud_score)
    return fraud_score
