from app.schemas.transaction import Transaction


def vector_transaction(transaction: Transaction) -> list[float]:
    return [0.0, 1.0]


def vector_transaction_search(
    transaction: Transaction, vector: list[float]
) -> list[Transaction]:
    return []
