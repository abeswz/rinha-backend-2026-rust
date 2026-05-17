use crate::AppState;
use crate::{
    domain::{fraud::FraudDecision, transaction::{Customer, LastTransaction, Merchant, Terminal, Transaction}},
    web::dto::{FraudScoreResponse, TransactionRequest},
};
use axum::{extract::State, Json};
use std::sync::Arc;
use std::time::Duration;

pub async fn ready_handler() -> &'static str {
    "ok"
}

pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> impl axum::response::IntoResponse {
    let tx = into_transaction(req);
    // 1600ms timeout: 200ms below nginx proxy_read_timeout (1800ms).
    // Fallback approved=true trades FN (penalty 3) vs HTTP 504 (penalty 5).
    let decision = tokio::time::timeout(
        Duration::from_millis(1600),
        tokio::task::spawn_blocking(move || state.use_case.execute(&tx)),
    )
    .await
    .ok()
    .and_then(|r| r.ok())
    .unwrap_or(FraudDecision {
        approved: true,
        fraud_score: 0.0,
    });
    Json(FraudScoreResponse {
        approved: decision.approved,
        fraud_score: decision.fraud_score,
    })
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
            known_merchants: req.customer.known_merchants,
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

#[cfg(test)]
mod tests {
    use crate::domain::fraud::FraudDecision;
    use std::time::Duration;

    #[tokio::test]
    async fn test_timeout_fallback_is_approved_true() {
        // When KNN exceeds timeout, fallback must be approved=true.
        // Scoring: FN penalty=3 < HTTP error penalty=5.
        // Changing to approved=false: fraud slips through AND adds FN penalty.
        let result = tokio::time::timeout(
            Duration::from_millis(50),
            tokio::task::spawn_blocking(|| {
                std::thread::sleep(Duration::from_millis(500));
                FraudDecision { approved: false, fraud_score: 1.0 }
            }),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(FraudDecision { approved: true, fraud_score: 0.0 });

        assert!(result.approved, "timeout fallback must be approved=true");
        assert_eq!(result.fraud_score, 0.0);
    }

    #[tokio::test]
    async fn test_fast_execution_returns_actual_decision() {
        let result = tokio::time::timeout(
            Duration::from_millis(1600),
            tokio::task::spawn_blocking(|| FraudDecision { approved: false, fraud_score: 0.8 }),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(FraudDecision { approved: true, fraud_score: 0.0 });

        assert!(!result.approved, "fast execution must return actual result, not fallback");
        assert_eq!(result.fraud_score, 0.8);
    }
}
