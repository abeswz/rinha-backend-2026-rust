from datetime import datetime, timezone
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.transaction import (
    Base,
    Customer,
    LastTransaction,
    Merchant,
    Terminal,
    Transaction,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """
    Client HTTP que fala diretamente com a app ASGI —
    sem abrir porta de rede real. Rápido e isolado.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def make_transaction():
    def _factory(**overrides: Any) -> Transaction:
        return Transaction(
            id=overrides.get("id", "tx-001"),
            transaction=overrides.get(
                "transaction",
                Base(
                    amount=100.0,
                    installments=1,
                    requested_at=datetime.now(timezone.utc),
                ),
            ),
            customer=overrides.get(
                "customer",
                Customer(
                    avg_amount=80.0,
                    tx_count_24h=2,
                    known_merchants=["merchant-123"],
                ),
            ),
            merchant=overrides.get(
                "merchant",
                Merchant(id="merchant-123", mcc="5411", avg_amount=75.0),
            ),
            terminal=overrides.get(
                "terminal",
                Terminal(is_online=True, card_present=True, km_from_home=1.2),
            ),
            last_transaction=overrides.get(
                "last_transaction",
                LastTransaction(
                    timestamp=datetime.now(timezone.utc),
                    km_from_current=0.5,
                ),
            ),
        )

    return _factory
