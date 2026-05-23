# XGBoost Hybrid Fraud Classifier

**Date:** 2026-05-23
**Goal:** Replace pure IVF path with LightGBM+ONNX primary classifier + IVF fallback to reduce CPU per request ~7x, eliminate CFS throttling on competition hardware, and reduce false positives.

**Baseline:** Rust LB + IVF AVX2 → p99=127ms, 23 FP, score=3481.56 on competition.

**Target:** p99 ≤ 5ms, 0 FP, score ≥ 5800 on competition.

---

## Root Cause

Mac Mini 2014 (i5-4278U, 2 cores, 3MB L3). IVF scan at nprobe=5/24 takes ~1ms per request (memory bandwidth bound, 3M vectors, small cache). At 900 req/s split across 2 API instances: 450 req/s × 1ms = 450ms/s — against 0.475 CPU quota (47.5ms per 100ms CFS period). Any burst exhausts quota → 95ms CFS wait → p99 spikes.

23 FP = IVF approximation error: nprobe misses correct clusters for edge cases.

---

## Solution

LightGBM model (14 features → fraud probability) handles ~90% of requests in ~50μs. IVF fallback only for uncertain cases (probability 0.30–0.70).

```
[request vector: f32[14]]
        ↓
[model.rs — tract ONNX inference, ~50μs]
        ↓ p >= 0.70   → fraud  (return immediately)
        ↓ p <= 0.30   → legit  (return immediately)
        ↓ 0.30 < p < 0.70 → knn5_ivf (existing IVF, ~1ms)
```

Weighted average CPU: 0.9 × 50μs + 0.1 × 1ms = **145μs** (vs ~1ms today, ~7x reduction).

---

## Architecture

### New components

| Component | Path | Role |
|-----------|------|------|
| `FraudModel` struct | `src/fraud/model.rs` | Load ONNX, run inference |
| Training script | `tools/train_model.py` | LightGBM → ONNX (offline) |
| Model file | `resources/model.onnx` | Baked into Docker image |

### Unchanged components

IVF index, `knn5_ivf`, AVX2 scan, LB, docker-compose — no changes.

### Runtime state

`FraudModel` stored as `Arc<FraudModel>` in `AppState`. Loaded once at startup, immutable. Thread-safe concurrent inference via tract's `TypedRunnableModel`.

---

## Training Pipeline (offline)

**Script:** `tools/train_model.py`

**Input:** `resources/references.json.gz` — 3M vectors (14 f32 dimensions, labels 0/1)

**Normalization:** same as IVF — load `resources/normalization.json`, apply before training

**Model:** `LightGBMClassifier`
- `n_estimators=300`
- `max_depth=6`
- `num_leaves=63`
- `learning_rate=0.05`

**Train/validation split:** 80/20

**Export:** `skl2onnx` → `resources/model.onnx`

**Python deps (uv):**
```
lightgbm
scikit-learn
skl2onnx
onnxmltools
numpy
```

**Offline validation targets:**
- AUC-ROC ≥ 0.99
- FP rate = 0% for predictions with p ≥ 0.70
- FN rate = 0% for predictions with p ≤ 0.30
- Edge case rate (0.30–0.70): 5–15% of validation set

---

## Rust Implementation

### Cargo.toml

```toml
[dependencies]
tract-onnx = "0.21"
```

tract is pure Rust — no `.so` dependencies, no change to Dockerfile.

### src/fraud/model.rs

```rust
use tract_onnx::prelude::*;

const LOW:  f32 = 0.30;
const HIGH: f32 = 0.70;

pub enum Decision { Fraud, Legit, Uncertain }

pub struct FraudModel {
    model: TypedRunnableModel<TypedModel>,
}

impl FraudModel {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    pub fn predict(&self, q: &[f32; 14]) -> Decision {
        // build input tensor, run, extract probability
        let p = /* tract inference result */;
        match p {
            p if p >= HIGH => Decision::Fraud,
            p if p <= LOW  => Decision::Legit,
            _              => Decision::Uncertain,
        }
    }
}
```

### Integration in request handler

```rust
// Before calling knn5_ivf:
match state.model.predict(&q) {
    Decision::Fraud    => return fraud_response(),
    Decision::Legit    => return legit_response(),
    Decision::Uncertain => { /* fall through to IVF */ }
}
let label = knn5_ivf(&q, ds);
```

---

## Memory Budget (per instance)

| Item | MB |
|------|----|
| IVF index (centroids + blocks + labels) | ~115 |
| ONNX model file | ~3 |
| tract runtime + graph | ~8 |
| **Total** | **~126** |

Limit: 172MB → **46MB headroom**.

---

## Testing

### Unit tests (`src/fraud/model.rs`)

- `predict()` returns `Fraud` for all-ones vector (clearly fraudulent neighborhood)
- `predict()` returns `Legit` for all-zeros vector (clearly legitimate neighborhood)
- `predict()` returns `Uncertain` for boundary vectors (mid-range features)
- `FraudModel::load()` succeeds with `resources/model.onnx`

### Integration tests

- End-to-end: model + IVF together score 6000/6000 on `test/test-data.json`
- Edge case rate: assert ≤ 15% of queries fall through to IVF
- Zero FP/FN on local dataset

### Regression

- Local p99 ≤ 0.22ms (model adds ≤ 50μs overhead)
- `cargo clippy --all-targets --all-features -- -D warnings` passes
- `cargo test` passes

---

## Threshold Tuning

Thresholds `LOW=0.30` and `HIGH=0.70` are constants in `model.rs`. Tuning procedure:

1. Run local test → check FP/FN count and edge case rate
2. If FP > 0: raise `LOW` (e.g. 0.20) to send more borderline legit → IVF
3. If edge case rate > 15%: widen thresholds toward 0.50 to reduce IVF load
4. If IVF FPs persist: increase `FAST_NPROBE` or `FULL_NPROBE` independently

No model retraining required to tune thresholds.

---

## Docker Image Impact

| Item | Change |
|------|--------|
| `resources/model.onnx` | +~3MB |
| `tract-onnx` compiled into binary | +~5MB |
| `libxgboost.so` | none (tract is pure Rust) |
| Dockerfile | unchanged |
| docker-compose.yml | unchanged |

---

## Expected Score Impact

| Metric | Baseline | Target |
|--------|----------|--------|
| p99 (competition) | 127ms | ≤ 5ms |
| FP | 23 | 0 |
| score_p99 | 895 | 2800–3000 |
| detection_score | 2586 | 3000 |
| **final_score** | **3481** | **≥ 5800** |
