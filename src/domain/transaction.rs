use chrono::{DateTime, Utc};

pub struct Transaction {
    pub id: String,
    pub amount: f32,
    pub installments: u32,
    pub requested_at: DateTime<Utc>,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

pub struct LastTransaction {
    pub timestamp: DateTime<Utc>,
    pub km_from_current: f32,
}
