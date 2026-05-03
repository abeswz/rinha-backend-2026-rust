from datetime import datetime, timezone

import pytest

from app.schemas.transaction import Base
from app.service.vector import _vector_amount, _vector_installment


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
            amount=0, installments=0, requested_at=datetime.now(timezone.utc)
        )
    )
    assert _vector_amount(t=transaction) == pytest.approx(-1)
