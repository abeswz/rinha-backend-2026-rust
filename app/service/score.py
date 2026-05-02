from app.schemas.fraud_score import FraudScore
from app.schemas.transaction import Transaction

# Fix value
# Reference: https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/REGRAS_DE_DETECCAO.md
THRESHOLD = 0.6


def process_score(
    transaction: Transaction, similar_transactions: list[Transaction]
) -> FraudScore:
    score = 1.0
    return FraudScore(approved=score < THRESHOLD, fraud_score=score)
