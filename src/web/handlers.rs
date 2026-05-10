use crate::AppState;
use crate::{
    domain::transaction::{Customer, LastTransaction, Merchant, Terminal, Transaction},
    error::AppError,
    web::dto::{FraudScoreResponse, TransactionRequest},
};
use axum::{extract::State, Json};
use std::collections::HashSet;
use std::sync::Arc;

pub async fn ready_handler() -> &'static str {
    "ok"
}

pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> Result<Json<FraudScoreResponse>, AppError> {
    let tx = into_transaction(req);
    let decision = state.use_case.execute(&tx);
    Ok(Json(FraudScoreResponse {
        approved: decision.approved,
        fraud_score: decision.fraud_score,
    }))
}

fn into_transaction(req: TransactionRequest) -> Transaction {
    Transaction {
        id: req.id,
        amount: req.transaction.amount,
        installments: req.transaction.installments,
        requested_at: req.transaction.requested_at,
        customer: Customer {
            avg_amount: req.customer.avg_amount,
            tx_count_24h: req.customer.tx_count_24h,
            known_merchants: req
                .customer
                .known_merchants
                .into_iter()
                .collect::<HashSet<_>>(),
        },
        merchant: Merchant {
            id: req.merchant.id,
            mcc: req.merchant.mcc,
            avg_amount: req.merchant.avg_amount,
        },
        terminal: Terminal {
            is_online: req.terminal.is_online,
            card_present: req.terminal.card_present,
            km_from_home: req.terminal.km_from_home,
        },
        last_transaction: req.last_transaction.map(|lt| LastTransaction {
            timestamp: lt.timestamp,
            km_from_current: lt.km_from_current,
        }),
    }
}
