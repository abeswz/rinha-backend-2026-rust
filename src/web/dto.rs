use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct TransactionRequest {
    pub id: String,
    pub transaction: TransactionDto,
    pub customer: CustomerDto,
    pub merchant: MerchantDto,
    pub terminal: TerminalDto,
    pub last_transaction: Option<LastTransactionDto>,
}

#[derive(Deserialize)]
pub struct TransactionDto {
    pub amount: f32,
    pub installments: u32,
    pub requested_at: DateTime<Utc>,
}

#[derive(Deserialize)]
pub struct CustomerDto {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct MerchantDto {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Deserialize)]
pub struct TerminalDto {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Deserialize)]
pub struct LastTransactionDto {
    pub timestamp: DateTime<Utc>,
    pub km_from_current: f32,
}

#[derive(Serialize)]
pub struct FraudScoreResponse {
    pub approved: bool,
    pub fraud_score: f32,
}
