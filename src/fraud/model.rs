#[path = "model_gen.rs"]
mod model_gen;

const LOW: f32 = 0.20;
const HIGH: f32 = 0.95;

pub enum Decision {
    Fraud,
    Legit,
    Uncertain,
}

pub fn init() {}

pub fn predict(q: &[f32; 14]) -> Decision {
    let input: Vec<f64> = q.iter().map(|&x| x as f64).collect();
    let p = model_gen::score(input)[1] as f32;
    if p >= HIGH {
        Decision::Fraud
    } else if p <= LOW {
        Decision::Legit
    } else {
        Decision::Uncertain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_fraud_for_all_ones() {
        let q = [1.0f32; 14];
        assert!(
            matches!(predict(&q), Decision::Fraud),
            "all-ones vector must classify as Fraud"
        );
    }

    #[test]
    fn predict_legit_for_all_zeros() {
        let q = [0.0f32; 14];
        assert!(
            matches!(predict(&q), Decision::Legit),
            "all-zeros vector must classify as Legit"
        );
    }

    #[test]
    fn uncertain_rate_within_bounds() {
        let boundary_vectors: Vec<[f32; 14]> = (0..20)
            .map(|i| {
                let t = i as f32 / 19.0;
                [t, t, t, t, t, t, t, t, t, 0.0, 1.0, 0.0, 0.5, t]
            })
            .collect();
        let uncertain_count = boundary_vectors
            .iter()
            .filter(|q| matches!(predict(q), Decision::Uncertain))
            .count();
        assert!(
            uncertain_count <= 10,
            "Too many uncertain predictions ({uncertain_count}/20); check thresholds"
        );
    }

    #[test]
    fn measure_inference_latency() {
        let q = [0.5f32; 14];
        for _ in 0..10 {
            predict(&q);
        }
        let n = 10_000usize;
        let start = std::time::Instant::now();
        for _ in 0..n {
            predict(&q);
        }
        let per_us = start.elapsed().as_micros() as f64 / n as f64;
        println!("\n=== m2cgen inference: {per_us:.2}µs per call ({n} iterations) ===\n");
        assert!(per_us < 1000.0, "inference {per_us:.1}µs > 1ms — too slow");
    }
}
