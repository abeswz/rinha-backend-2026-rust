use crate::domain::{fraud::FraudDecision, transaction::Transaction};
use crate::repository::reference::ReferenceRepository;
use crate::service::vectorizer::Vectorizer;

pub struct ScoreFraudUseCase {
    pub vectorizer: Vectorizer,
    pub repository: ReferenceRepository,
}

impl ScoreFraudUseCase {
    pub fn execute(&self, tx: &Transaction) -> FraudDecision {
        let vector = self.vectorizer.vectorize(tx);
        let labels = self.repository.knn_adaptive(&vector.0, 5);
        let fraud_count = labels.iter().filter(|&&l| l == 1).count();
        let fraud_score = fraud_count as f32 / 5.0;
        FraudDecision {
            approved: fraud_score < 0.6,
            fraud_score,
            fraud_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::transaction::{Customer, Merchant, Terminal, Transaction};
    use crate::repository::reference::ReferenceRepository;
    use crate::service::vectorizer::{MccRiskMap, NormalizationConstants, Vectorizer};
    use chrono::Utc;

    fn make_repo_at(name: &str) -> ReferenceRepository {
        let mut buf: Vec<u8> = Vec::new();
        // IVF2 format
        buf.extend_from_slice(b"IVF2");
        buf.extend_from_slice(&8u32.to_le_bytes()); // n=8
        buf.extend_from_slice(&2u32.to_le_bytes()); // k=2
        buf.extend_from_slice(&14u32.to_le_bytes()); // d=14

        // centroids column-major: C0=[0.0;14], C1=[10.0;14]
        for _ in 0..14 {
            buf.extend_from_slice(&0.0f32.to_le_bytes()); // C0
            buf.extend_from_slice(&10.0f32.to_le_bytes()); // C1
        }

        // offsets: [0, 1, 2]
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());

        // labels: 2 blocks × 8 slots
        buf.extend_from_slice(&[0u8; 8]); // block 0: legit
        buf.extend_from_slice(&[1u8; 8]); // block 1: fraud

        // blocks: 2 × 14 × 8 i16
        let legit_val: i16 = 1000; // ~0.1
        let fraud_val: i16 = i16::MAX; // far away
        for _ in 0..112 { buf.extend_from_slice(&legit_val.to_le_bytes()); }
        for _ in 0..112 { buf.extend_from_slice(&fraud_val.to_le_bytes()); }

        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, &buf).unwrap();
        ReferenceRepository::from_file(&path, 3, 24).unwrap()
    }

    fn make_tx(amount: f32) -> Transaction {
        Transaction {
            id: "test-tx-001".to_string(),
            amount,
            installments: 1,
            requested_at: Utc::now(),
            customer: Customer {
                avg_amount: 100.0,
                tx_count_24h: 1,
                known_merchants: vec!["m1".to_string()],
            },
            merchant: Merchant {
                id: "m1".to_string(),
                mcc: "5411".to_string(),
                avg_amount: 100.0,
            },
            terminal: Terminal {
                is_online: true,
                card_present: true,
                km_from_home: 0.5,
            },
            last_transaction: None,
        }
    }

    #[test]
    fn test_execute_sets_fraud_count() {
        let repo = make_repo_at("usecase_test_ivf_count.bin");
        let norm = NormalizationConstants::default();
        let mcc_risk = MccRiskMap::default();
        let use_case = ScoreFraudUseCase {
            vectorizer: Vectorizer::new(norm, mcc_risk),
            repository: repo,
        };
        let tx = make_tx(100.0);
        let decision = use_case.execute(&tx);
        assert!(decision.fraud_count <= 5, "fraud_count must be 0..=5");
    }

    #[test]
    fn test_execute_returns_fraud_decision() {
        let repo = make_repo_at("usecase_test_ivf_decision.bin");
        let norm = NormalizationConstants::default();
        let mcc_risk = MccRiskMap::default();
        let use_case = ScoreFraudUseCase {
            vectorizer: Vectorizer::new(norm, mcc_risk),
            repository: repo,
        };
        let tx = make_tx(100.0);
        let decision = use_case.execute(&tx);
        assert!(
            decision.fraud_score >= 0.0 && decision.fraud_score <= 1.0,
            "fraud_score out of range: {}",
            decision.fraud_score
        );
    }
}
