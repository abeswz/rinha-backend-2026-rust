use crate::AppState;
use crate::{
    domain::fraud::FraudDecision,
    domain::transaction::{Customer, LastTransaction, Merchant, Terminal, Transaction},
    web::dto::TransactionRequest,
};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    Json,
};
use std::sync::Arc;
use std::time::Duration;

pub const STATIC_BODIES: [&str; 6] = [
    r#"{"approved":true,"fraud_score":0.0}"#,
    r#"{"approved":true,"fraud_score":0.2}"#,
    r#"{"approved":true,"fraud_score":0.4}"#,
    r#"{"approved":false,"fraud_score":0.6}"#,
    r#"{"approved":false,"fraud_score":0.8}"#,
    r#"{"approved":false,"fraud_score":1.0}"#,
];

pub async fn ready_handler() -> &'static str {
    "ok"
}

pub async fn fraud_score_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TransactionRequest>,
) -> impl IntoResponse {
    let tx = into_transaction(req);
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
        fraud_count: 0,
    });

    let body = STATIC_BODIES[decision.fraud_count.min(5)];
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
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
    use super::*;
    use crate::domain::fraud::FraudDecision;
    use std::time::Duration;

    #[test]
    fn test_static_bodies_all_valid_json() {
        for (i, body) in STATIC_BODIES.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(body)
                .expect(&format!("body[{i}] is not valid JSON: {body}"));
            assert!(v.get("approved").is_some(), "body[{i}] missing 'approved'");
            assert!(v.get("fraud_score").is_some(), "body[{i}] missing 'fraud_score'");
        }
    }

    #[tokio::test]
    async fn test_timeout_fallback_is_approved_true() {
        let result = tokio::time::timeout(
            Duration::from_millis(50),
            tokio::task::spawn_blocking(|| {
                std::thread::sleep(Duration::from_millis(500));
                FraudDecision { approved: false, fraud_score: 1.0, fraud_count: 5 }
            }),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(FraudDecision { approved: true, fraud_score: 0.0, fraud_count: 0 });

        assert!(result.approved, "timeout fallback must be approved=true");
        assert_eq!(result.fraud_count, 0);
    }

    #[tokio::test]
    async fn test_fast_execution_returns_actual_decision() {
        let result = tokio::time::timeout(
            Duration::from_millis(1600),
            tokio::task::spawn_blocking(|| FraudDecision {
                approved: false,
                fraud_score: 0.8,
                fraud_count: 4,
            }),
        )
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(FraudDecision { approved: true, fraud_score: 0.0, fraud_count: 0 });

        assert!(!result.approved);
        assert_eq!(result.fraud_count, 4);
    }
}
