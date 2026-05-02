from app.schemas.fraud_score import FraudScore
from app.schemas.transaction import Transaction


async def transaction_fraud_score(transaction: Transaction) -> FraudScore | None:
    amount_vector = transaction_vector_amount(transaction)
    print(f"amount: {amount_vector}")
    return FraudScore(approved=True, fraud_score=100.0)


# TODO: Fix this part and add real vector validation this is just a validation process
# try to understand manual vectoring data
def transaction_vector_amount(transaction: Transaction) -> list[float]:
    max_amount = 1_000.0

    amount_feature = min(transaction.transaction.amount / max_amount, 1.0)

    return [amount_feature]
