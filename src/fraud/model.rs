use std::io::Cursor;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

const LOW: f32 = 0.20;
const HIGH: f32 = 0.95;

pub enum Decision {
    Fraud,
    Legit,
    Uncertain,
}

pub struct FraudModel {
    model: TypedRunnableModel<TypedModel>,
}

static MODEL: OnceLock<FraudModel> = OnceLock::new();

pub fn init() {
    MODEL.get_or_init(|| FraudModel::load().expect("failed to load ONNX model"));
}

pub fn predict(q: &[f32; 14]) -> Decision {
    MODEL
        .get()
        .expect("call fraud::model::init() before predict()")
        .predict(q)
}

static MODEL_BYTES: &[u8] = include_bytes!("../../resources/model.onnx");

impl FraudModel {
    fn load() -> anyhow::Result<Self> {
        let mut cursor = Cursor::new(MODEL_BYTES);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![1, 14]))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    fn predict(&self, q: &[f32; 14]) -> Decision {
        let input = tract_ndarray::Array2::from_shape_fn((1, 14), |(_, j)| q[j]);
        let input: Tensor = input.into();
        let result = self
            .model
            .run(tvec![input.into()])
            .expect("tract inference failed");
        // result[1] = probabilities (f32 [1, 2]): [0]=P(legit), [1]=P(fraud)
        let probs = result[1].as_slice::<f32>().expect("probabilities not f32 slice");
        let p = probs[1];
        match p {
            p if p >= HIGH => Decision::Fraud,
            p if p <= LOW => Decision::Legit,
            _ => Decision::Uncertain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ensure_loaded() {
        init();
    }

    #[test]
    fn load_succeeds() {
        ensure_loaded();
    }

    #[test]
    fn predict_fraud_for_all_ones() {
        ensure_loaded();
        let q = [1.0f32; 14];
        let result = predict(&q);
        assert!(
            matches!(result, Decision::Fraud),
            "all-ones vector must classify as Fraud"
        );
    }

    #[test]
    fn predict_legit_for_all_zeros() {
        ensure_loaded();
        let q = [0.0f32; 14];
        let result = predict(&q);
        assert!(
            matches!(result, Decision::Legit),
            "all-zeros vector must classify as Legit"
        );
    }

    #[test]
    fn uncertain_rate_within_bounds() {
        ensure_loaded();
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
}
