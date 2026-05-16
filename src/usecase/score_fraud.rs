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
        }
    }
}
