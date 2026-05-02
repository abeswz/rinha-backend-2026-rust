from datetime import datetime
from typing import List

from pydantic import BaseModel


class Base(BaseModel):
    amount: float
    installments: int
    requested_at: datetime


class Customer(BaseModel):
    avg_amount: float
    tx_count_24h: int
    known_merchants: List[str]


class Merchant(BaseModel):
    id: str
    mcc: str
    avg_amount: float


class Terminal(BaseModel):
    is_online: bool
    card_present: bool
    km_from_home: float


class LastTransaction(BaseModel):
    timestamp: datetime
    km_from_current: float


class Transaction(BaseModel):
    id: str
    transaction: Base
    customer: Customer
    merchant: Merchant
    terminal: Terminal
    last_transaction: LastTransaction
