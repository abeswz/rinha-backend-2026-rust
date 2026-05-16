use crate::domain::{fraud::FraudVector, transaction::Transaction};
use chrono::{Datelike, Timelike};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub struct NormalizationConstants {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

#[derive(Deserialize)]
struct NormalizationJson {
    max_amount: f32,
    max_installments: f32,
    amount_vs_avg_ratio: f32,
    max_minutes: f32,
    max_km: f32,
    max_tx_count_24h: f32,
    max_merchant_avg_amount: f32,
}

impl Default for NormalizationConstants {
    fn default() -> Self {
        Self {
            max_amount: 10_000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1_440.0,
            max_km: 1_000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10_000.0,
        }
    }
}

impl NormalizationConstants {
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let j: NormalizationJson = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self {
            max_amount: j.max_amount,
            max_installments: j.max_installments,
            amount_vs_avg_ratio: j.amount_vs_avg_ratio,
            max_minutes: j.max_minutes,
            max_km: j.max_km,
            max_tx_count_24h: j.max_tx_count_24h,
            max_merchant_avg_amount: j.max_merchant_avg_amount,
        })
    }
}

#[derive(Default)]
pub struct MccRiskMap(HashMap<String, f32>);

impl MccRiskMap {
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let map: HashMap<String, f32> = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self(map))
    }

    pub fn get(&self, mcc: &str) -> f32 {
        self.0.get(mcc).copied().unwrap_or(0.5)
    }
}

pub struct Vectorizer {
    pub norm: NormalizationConstants,
    pub mcc_risk: MccRiskMap,
}

impl Vectorizer {
    pub fn new(norm: NormalizationConstants, mcc_risk: MccRiskMap) -> Self {
        Self { norm, mcc_risk }
    }

    pub fn vectorize(&self, tx: &Transaction) -> FraudVector {
        let n = &self.norm;

        let (minutes_since_last, km_from_last) = match &tx.last_transaction {
            Some(lt) => {
                let mins = (tx.requested_at - lt.timestamp).num_minutes() as f32;
                let km = lt.km_from_current;
                (
                    (mins / n.max_minutes).clamp(0.0, 1.0),
                    (km / n.max_km).clamp(0.0, 1.0),
                )
            }
            None => (-1.0, -1.0),
        };

        let unknown_merchant = if tx.customer.known_merchants.contains(&tx.merchant.id) {
            0.0
        } else {
            1.0
        };

        let hour = tx.requested_at.hour() as f32;
        let weekday = tx.requested_at.weekday().num_days_from_monday() as f32;

        FraudVector([
            (tx.amount / n.max_amount).clamp(0.0, 1.0),
            (tx.installments as f32 / n.max_installments).clamp(0.0, 1.0),
            ((tx.amount / tx.customer.avg_amount) / n.amount_vs_avg_ratio).clamp(0.0, 1.0),
            hour / 23.0,
            weekday / 6.0,
            minutes_since_last,
            km_from_last,
            (tx.terminal.km_from_home / n.max_km).clamp(0.0, 1.0),
            (tx.customer.tx_count_24h as f32 / n.max_tx_count_24h).clamp(0.0, 1.0),
            if tx.terminal.is_online { 1.0 } else { 0.0 },
            if tx.terminal.card_present { 1.0 } else { 0.0 },
            unknown_merchant,
            self.mcc_risk.get(&tx.merchant.mcc),
            (tx.merchant.avg_amount / n.max_merchant_avg_amount).clamp(0.0, 1.0),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::transaction::*;
    use chrono::DateTime;

    fn make_vectorizer() -> Vectorizer {
        let norm =
            NormalizationConstants::from_file(std::path::Path::new("resources/normalization.json"))
                .expect("normalization.json missing — run tests from project root");
        let mcc_risk = MccRiskMap::from_file(std::path::Path::new("resources/mcc_risk.json"))
            .expect("mcc_risk.json missing");
        Vectorizer { norm, mcc_risk }
    }

    #[allow(clippy::excessive_precision)]
    fn legit_tx() -> Transaction {
        Transaction {
            id: "tx-1329056812".to_string(),
            amount: 41.12,
            installments: 2,
            requested_at: DateTime::parse_from_rfc3339("2026-03-11T18:45:53Z")
                .unwrap()
                .into(),
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".to_string(), "MERC-016".to_string()],
            },
            merchant: Merchant {
                id: "MERC-016".to_string(),
                mcc: "5411".to_string(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.2331036248,
            },
            last_transaction: None,
        }
    }

    fn fraud_tx() -> Transaction {
        Transaction {
            id: "tx-3330991687".to_string(),
            amount: 9505.97,
            installments: 10,
            requested_at: DateTime::parse_from_rfc3339("2026-03-14T05:15:12Z")
                .unwrap()
                .into(),
            customer: Customer {
                avg_amount: 81.28,
                tx_count_24h: 20,
                known_merchants: vec![
                    "MERC-008".to_string(),
                    "MERC-007".to_string(),
                    "MERC-005".to_string(),
                ],
            },
            merchant: Merchant {
                id: "MERC-068".to_string(),
                mcc: "7802".to_string(),
                avg_amount: 54.86,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 952.27,
            },
            last_transaction: None,
        }
    }

    #[test]
    fn test_vectorize_legit_example() {
        let v = make_vectorizer();
        let vec = v.vectorize(&legit_tx());
        let d = vec.0;
        assert!(
            (d[0] - 0.004112).abs() < 0.0001,
            "dim0 expected ~0.0041 got {}",
            d[0]
        );
        assert!(
            (d[1] - 0.16667).abs() < 0.0001,
            "dim1 expected ~0.1667 got {}",
            d[1]
        );
        assert!(
            (d[2] - 0.05).abs() < 0.0001,
            "dim2 expected 0.05 got {}",
            d[2]
        );
        assert!(
            (d[3] - 0.78261).abs() < 0.0001,
            "dim3 expected ~0.7826 got {}",
            d[3]
        );
        assert!(
            (d[4] - 0.33333).abs() < 0.0001,
            "dim4 expected ~0.3333 got {}",
            d[4]
        );
        assert_eq!(d[5], -1.0, "dim5 must be -1 for null last_tx");
        assert_eq!(d[6], -1.0, "dim6 must be -1 for null last_tx");
        assert!(
            (d[7] - 0.02923).abs() < 0.0001,
            "dim7 expected ~0.0292 got {}",
            d[7]
        );
        assert!(
            (d[8] - 0.15).abs() < 0.0001,
            "dim8 expected 0.15 got {}",
            d[8]
        );
        assert_eq!(d[9], 0.0, "dim9: not online");
        assert_eq!(d[10], 1.0, "dim10: card present");
        assert_eq!(d[11], 0.0, "dim11: known merchant");
        assert!((d[12] - 0.15).abs() < 0.0001, "dim12: mcc 5411 = 0.15");
        assert!((d[13] - 0.006025).abs() < 0.0001, "dim13 got {}", d[13]);
    }

    #[test]
    fn test_vectorize_fraud_example() {
        let v = make_vectorizer();
        let vec = v.vectorize(&fraud_tx());
        let d = vec.0;
        assert!((d[0] - 0.950597).abs() < 0.0001, "dim0 got {}", d[0]);
        assert!((d[1] - 0.83333).abs() < 0.0001, "dim1 got {}", d[1]);
        assert_eq!(d[2], 1.0, "dim2: clamped to 1.0");
        assert!((d[3] - 0.21739).abs() < 0.0001, "dim3 got {}", d[3]);
        assert!((d[4] - 0.83333).abs() < 0.0001, "dim4 got {}", d[4]);
        assert_eq!(d[5], -1.0, "dim5: null last_tx");
        assert_eq!(d[6], -1.0, "dim6: null last_tx");
        assert!((d[7] - 0.95227).abs() < 0.0001, "dim7 got {}", d[7]);
        assert_eq!(d[8], 1.0, "dim8: clamped to 1.0");
        assert_eq!(d[9], 0.0, "dim9: not online");
        assert_eq!(d[10], 1.0, "dim10: card present");
        assert_eq!(d[11], 1.0, "dim11: unknown merchant");
        assert!((d[12] - 0.75).abs() < 0.0001, "dim12: mcc 7802 = 0.75");
        assert!((d[13] - 0.005486).abs() < 0.0001, "dim13 got {}", d[13]);
    }

    #[test]
    fn test_vectorize_null_last_tx() {
        let v = make_vectorizer();
        let mut tx = legit_tx();
        tx.last_transaction = None;
        let vec = v.vectorize(&tx);
        assert_eq!(vec.0[5], -1.0, "dim5 must be -1.0 when last_tx is None");
        assert_eq!(vec.0[6], -1.0, "dim6 must be -1.0 when last_tx is None");
    }

    #[test]
    fn test_vectorize_clamp_high() {
        let v = make_vectorizer();
        let mut tx = legit_tx();
        tx.amount = 100_000.0;
        let vec = v.vectorize(&tx);
        assert_eq!(
            vec.0[0], 1.0,
            "dim0: clamped to 1.0 when amount >> max_amount"
        );
    }

    #[test]
    fn test_vectorize_unknown_mcc() {
        let v = make_vectorizer();
        let mut tx = legit_tx();
        tx.merchant.mcc = "9999".to_string();
        let vec = v.vectorize(&tx);
        assert!(
            (vec.0[12] - 0.5).abs() < 0.0001,
            "dim12: unknown MCC defaults to 0.5"
        );
    }

    #[test]
    fn test_vectorize_known_mcc() {
        let v = make_vectorizer();
        let mut tx = legit_tx();
        tx.merchant.mcc = "7995".to_string();
        let vec = v.vectorize(&tx);
        assert!((vec.0[12] - 0.85).abs() < 0.0001, "dim12: MCC 7995 = 0.85");
    }
}
