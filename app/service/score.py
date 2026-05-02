from app.schemas.fraud_score import FraudScore
from app.schemas.transaction import Transaction


def process_score(
    transaction: Transaction, similar_transactions: list[Transaction]
) -> FraudScore:
    return FraudScore(approved=True, fraud_score=100.0)
