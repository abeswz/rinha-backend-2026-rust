from app.schemas.fraud_score import FraudScore
from app.schemas.transaction import Transaction
from app.service.score import process_score
from app.service.vector import vector_transaction, vector_transaction_search


async def transaction_fraud_score(transaction: Transaction) -> FraudScore | None:
    vector = vector_transaction(transaction=transaction)
    nn_transactions = vector_transaction_search(transaction=transaction, vector=vector)
    score = process_score(transaction=transaction, similar_transactions=nn_transactions)

    return score
