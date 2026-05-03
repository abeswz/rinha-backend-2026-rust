from datetime import datetime, timezone

import pytest

from app.schemas.transaction import Base, Customer
from app.service.vector import (
    _vector_amount,
    _vector_amount_avg,
    _vector_installment,
    _vector_limitation_value,
)


# AMOUNT
def test_vector_process_amount(make_transaction):
    transaction = make_transaction(
        transaction=Base(
            amount=1_000, installments=1, requested_at=datetime.now(timezone.utc)
        )
    )
    assert _vector_amount(t=transaction) == pytest.approx(0.1)


def test_vector_process_amount_lower_amount(make_transaction):
    transaction = make_transaction(
        transaction=Base(
            amount=0, installments=1, requested_at=datetime.now(timezone.utc)
        )
    )
    assert _vector_amount(t=transaction) == pytest.approx(-1)


# INSTALLMENTS
def test_vector_process_installment(make_transaction):
    transaction = make_transaction(
        transaction=Base(
            amount=1_000, installments=12, requested_at=datetime.now(timezone.utc)
        )
    )
    assert _vector_installment(t=transaction) == pytest.approx(1.0)


def test_vector_process_installment_lower_installment(make_transaction):
    transaction = make_transaction(
        transaction=Base(
            amount=1_000, installments=0, requested_at=datetime.now(timezone.utc)
        )
    )
    assert _vector_installment(t=transaction) == pytest.approx(-1)


# AMOUNT_VS_AVG
def test_vector_amount_avg(make_transaction):
    transaction = make_transaction(
        transaction=Base(
            amount=1_000, installments=12, requested_at=datetime.now(timezone.utc)
        ),
        customer=Customer(
            avg_amount=50, tx_count_24h=3, known_merchants=["MERC-003", "MERC-016"]
        ),
    )
    assert _vector_amount_avg(t=transaction) == pytest.approx(1.0)


def test_vector_amount_avg_lower_avg(make_transaction):
    transaction = make_transaction(
        transaction=Base(
            amount=0, installments=0, requested_at=datetime.now(timezone.utc)
        )
    )
    assert _vector_amount_avg(t=transaction) == pytest.approx(-1)


# HELP FUNCTION
@pytest.mark.parametrize("value", [1.1, 2.0, 1.0, 99.9, 100.9])
def test_limitation_values(value):
    r = _vector_limitation_value(value)
    assert r == 1.0


@pytest.mark.parametrize("value", [0.0, -1.0, -0.0, -1.99, -100.9])
def test_limitation_values_lower_values(value):
    r = _vector_limitation_value(value)
    assert r == 0.0


@pytest.mark.parametrize("value", [0.1, 0.5, 1.0, 0.99, 0.86, 0.99999999999])
def test_limitation_values_secure_window(value):
    r = _vector_limitation_value(value)
    assert r == value
