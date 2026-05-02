from app.schemas.transaction import Transaction

MAX_AMOUNT = 10_000


def vector_transaction(transaction: Transaction) -> list[float]:
    amount_dimession = _vector_amount(t=transaction)
    return [amount_dimession, 1.0]


def _vector_amount(t: Transaction) -> float:
    amount = t.transaction.amount
    if amount <= 0:
        return -1

    return amount / MAX_AMOUNT


# 0 	amount 	limitar(transaction.amount / max_amount) ✅️
# 1 	installments 	limitar(transaction.installments / max_installments)
# 2 	amount_vs_avg 	limitar((transaction.amount / customer.avg_amount) / amount_vs_avg_ratio)
# 3 	hour_of_day 	hora(transaction.requested_at) / 23 (0-23, UTC)
# 4 	day_of_week 	dia_da_semana(transaction.requested_at) / 6 (seg=0, dom=6)
# 5 	minutes_since_last_tx 	limitar(minutos / max_minutes) ou -1 se last_transaction: null
# 6 	km_from_last_tx 	limitar(last_transaction.km_from_current / max_km) ou -1 se last_transaction: null
# 7 	km_from_home 	limitar(terminal.km_from_home / max_km)
# 8 	tx_count_24h 	limitar(customer.tx_count_24h / max_tx_count_24h)
# 9 	is_online 	1 se terminal.is_online, senão 0
# 10 	card_present 	1 se terminal.card_present, senão 0
# 11 	unknown_merchant 	1 se merchant.id não estiver em customer.known_merchants, senão 0 (invertido: 1 = desconhecido)
# 12 	mcc_risk 	mcc_risk.json[merchant.mcc] (valor padrão 0.5)
# 13 	merchant_avg_amount 	limitar(merchant.avg_amount / max_merchant_avg_amount)


# Normalization
# {
#   "max_amount": 10000,
#   "max_installments": 12,
#   "amount_vs_avg_ratio": 10,
#   "max_minutes": 1440,
#   "max_km": 1000,
#   "max_tx_count_24h": 20,
#   "max_merchant_avg_amount": 10000
# }
#
# TODO: add function vector here:
#
def vector_transaction_search(
    transaction: Transaction, vector: list[float]
) -> list[Transaction]:
    return []
