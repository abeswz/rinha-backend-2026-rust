pub struct FraudVector(pub [f32; 14]);

pub struct FraudDecision {
    pub approved: bool,
    pub fraud_score: f32,
    pub fraud_count: usize,
}
