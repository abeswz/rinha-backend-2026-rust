# Fraud Detection API — Design Spec
Date: 2026-05-10

## Problem

Build a high-performance fraud detection API for Rinha de Backend.
Receives transaction payloads, computes a 14-dimension vector, searches a 3M-vector reference dataset with KNN (k=5, Euclidean distance), returns `fraud_score` and `approved`.

Success criteria: maximize `score_final = score_p99 + score_det` (each -3000 to +3000).
Detection accuracy has a hard cut at 15% failure rate → exact KNN is safer than approximate.

## Infrastructure

### Topology

```
Client → nginx:9999 (round-robin) → api1:3000
                                  → api2:3000
```

nginx is a pure pass-through load balancer. No payload inspection, no business logic.

### Resource Budget

| Service | CPUs  | RAM   |
|---------|-------|-------|
| nginx   | 0.05  | 10MB  |
| api1    | 0.475 | 170MB |
| api2    | 0.475 | 170MB |
| Total   | 1.0   | 350MB |

Each API instance uses ~87MB for reference data (84MB f16 vectors + 3MB labels),
leaving ~83MB for Rust runtime, Tokio stack, active request buffers.

### Dockerfile (multi-stage)

**Stage 1 — builder:**
- Install Rust toolchain
- Copy source + resources
- Run `cargo build --release` (produces `fraud-detection` binary)
- Run `cargo run --bin preprocess` — converts `references.json.gz` to `refs.bin`
  - Format: N × `[f16; 14]` vectors (contiguous) followed by N × `u8` labels
  - Eliminates JSON parsing + gzip decompression from startup path

**Stage 2 — runtime (debian:bookworm-slim):**
- Copy binary + `refs.bin` + `mcc_risk.json` + `normalization.json`
- Expose port 3000
- `CMD ["./fraud-detection"]`

### docker-compose.yml structure

```yaml
services:
  nginx:
    image: nginx:alpine
    ports: ["9999:9999"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf:ro"]
    depends_on: [api1, api2]
    networks: [fraud-net]
    deploy:
      resources:
        limits:
          cpus: "0.05"
          memory: "10MB"

  api1:
    build: .
    environment: [PORT=3000]
    networks: [fraud-net]
    deploy:
      resources:
        limits:
          cpus: "0.475"
          memory: "170MB"

  api2:
    build: .
    environment: [PORT=3000]
    networks: [fraud-net]
    deploy:
      resources:
        limits:
          cpus: "0.475"
          memory: "170MB"

networks:
  fraud-net:
    driver: bridge
```

### nginx.conf (relevant section)

```nginx
upstream api_backends {
    server api1:3000;
    server api2:3000;
}
server {
    listen 9999;
    location / {
        proxy_pass http://api_backends;
    }
}
```

## Code Architecture (DDD-lite)

### Module Structure

```
src/
├── main.rs              # tokio main, AppState init, Axum bind :PORT
├── config.rs            # Config { port, refs_path, mcc_path, norm_path }
├── error.rs             # AppError enum + IntoResponse (422 for bad input, 500 for internal)
│
├── web/
│   ├── mod.rs
│   ├── router.rs        # GET /ready, POST /fraud-score routes
│   ├── handlers.rs      # ready_handler, fraud_score_handler
│   └── dto.rs           # TransactionRequest, FraudScoreResponse (serde)
│
├── domain/
│   ├── mod.rs
│   ├── transaction.rs   # Transaction, Customer, Merchant, Terminal, LastTransaction
│   └── fraud.rs         # FraudVector([f32; 14]), FraudScore(f32), FraudDecision { approved, fraud_score }
│
├── usecase/
│   ├── mod.rs
│   └── score_fraud.rs   # ScoreFraudUseCase::execute(&self, tx: Transaction) -> FraudDecision
│
├── service/
│   ├── mod.rs
│   └── vectorizer.rs    # Vectorizer::vectorize(&self, tx: &Transaction) -> FraudVector
│
└── repository/
    ├── mod.rs
    └── reference.rs     # ReferenceRepository::knn(&self, v: &[f32; 14], k: usize) -> SmallVec<[u8; 5]>

bin/
└── preprocess.rs        # Build-time: references.json.gz → refs.bin
```

### Key Types

```rust
// domain/fraud.rs
pub struct FraudVector(pub [f32; 14]);
pub struct FraudDecision {
    pub approved: bool,
    pub fraud_score: f32,
}

// domain/transaction.rs
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

// repository/reference.rs
pub struct ReferenceRepository {
    vectors: Box<[[f16; 14]]>,  // 3_000_000 × 14, contiguous heap allocation
    labels: Box<[u8]>,          // 3_000_000 labels (0 = legit, 1 = fraud)
}
```

### AppState

```rust
pub struct AppState {
    pub use_case: ScoreFraudUseCase,
}
// Arc<AppState> is the Axum state type — AppState derives Clone via Arc<AppState>.
// No inner Arc needed on sub-fields: the outer Arc handles shared ownership.
// No Mutex needed — all state is read-only after startup.
```

### Data Flow

```
POST /fraud-score
  └─ dto::TransactionRequest          (serde deserialization)
  └─ domain::Transaction              (validated conversion, no allocations)
  └─ ScoreFraudUseCase::execute(tx)
       ├─ Vectorizer::vectorize(tx)   → FraudVector([f32; 14])
       ├─ ReferenceRepository::knn()  → top-5 labels
       ├─ fraud_score = fraud_count / 5.0
       └─ approved = fraud_score < 0.6
  └─ dto::FraudScoreResponse { approved, fraud_score }
  └─ HTTP 200 JSON
```

### Layer Contracts

- `web/` knows DTOs + AppState only. No domain logic.
- `domain/` has no I/O, no HTTP types, no serde derives.
- `service/` depends on domain types + NormalizationConstants + MccRiskMap. No I/O.
- `usecase/` orchestrates service + repository. No HTTP.
- `repository/` knows vectors + labels only. No transaction or fraud types.

## KNN Implementation

### Algorithm: Brute Force with f16 Storage

Rationale: exact KNN = perfect detection score. Memory budget prohibits HNSW (M=8 needs
~186MB per instance for graph alone). f16 storage = 84MB vs f32 = 168MB per instance.

### Inner Loop Design

```rust
pub fn knn(&self, query: &[f32; 14], k: usize) -> SmallVec<[u8; 5]> {
    // Stack-allocated max-heap of size k for (distance: f32, label: u8)
    // Single pass over all N vectors:
    //   for each vector chunk:
    //     convert f16 → f32 (inline, per-element)
    //     compute squared Euclidean distance (no sqrt — order preserved)
    //     update heap if new distance < heap.max()
    // Return heap labels in any order (only label counts matter for score)
}
```

Key performance properties:
- No heap allocations per query
- Sequential memory access pattern (cache-friendly)
- Squared distance avoids sqrt (same rank order)
- Query vector stays in registers (14 f32 = 56 bytes, fits in L1 cache)
- Compiler can auto-vectorize the distance accumulation with AVX2

### Preprocessing Binary (bin/preprocess.rs)

Input: `resources/references.json.gz`
Output: `resources/refs.bin`

Binary format:
```
[4 bytes: N as u32 little-endian]
[N × 14 × 2 bytes: f16 vectors, row-major]
[N × 1 byte: labels (0=legit, 1=fraud)]
```

Run during Docker build stage. Result included in runtime image.
Startup: read file, cast bytes to slices directly. Zero parsing overhead.

## Vectorization Rules (14 Dimensions)

Constants from `normalization.json`:
```
max_amount = 10000, max_installments = 12, amount_vs_avg_ratio = 10,
max_minutes = 1440, max_km = 1000, max_tx_count_24h = 20,
max_merchant_avg_amount = 10000
```

| idx | dimension              | formula |
|-----|------------------------|---------|
| 0   | amount                 | clamp(amount / 10000) |
| 1   | installments           | clamp(installments / 12) |
| 2   | amount_vs_avg          | clamp((amount / avg_amount) / 10) |
| 3   | hour_of_day            | hour_utc / 23 |
| 4   | day_of_week            | weekday (Mon=0, Sun=6) / 6 |
| 5   | minutes_since_last_tx  | clamp(minutes / 1440) OR -1.0 if last_tx is None |
| 6   | km_from_last_tx        | clamp(km_from_current / 1000) OR -1.0 if last_tx is None |
| 7   | km_from_home           | clamp(km_from_home / 1000) |
| 8   | tx_count_24h           | clamp(tx_count_24h / 20) |
| 9   | is_online              | 1.0 if online else 0.0 |
| 10  | card_present           | 1.0 if card_present else 0.0 |
| 11  | unknown_merchant       | 1.0 if merchant.id NOT in known_merchants else 0.0 |
| 12  | mcc_risk               | mcc_risk.json[mcc] OR 0.5 if not found |
| 13  | merchant_avg_amount    | clamp(merchant.avg_amount / 10000) |

clamp(x) = x.clamp(0.0, 1.0). Dims 5 and 6 use -1.0 as sentinel (not clamped).

## API Contract

### GET /ready
Response: HTTP 200, body: `"ok"`
Condition: returns 200 only after AppState is fully initialized (refs.bin loaded, indexes ready).

### POST /fraud-score
Request body: `TransactionRequest` (JSON).
Response body: `{ "approved": bool, "fraud_score": f32 }`
Error: HTTP 422 for malformed input (serde validation failure).
No HTTP 500 in normal operation — panic = restartable container.

## Testing Strategy

All tests validate real fraud scenarios. No coverage-padding tests.

### Unit Tests (inline in modules)

**service/vectorizer.rs:**
- `test_vectorize_legit_example` — exact legit payload from detection-rules.md, assert all 14 dims to 4dp
- `test_vectorize_fraud_example` — exact fraud payload, assert all 14 dims to 4dp
- `test_vectorize_null_last_tx` — dims 5 and 6 must equal -1.0 exactly
- `test_vectorize_clamp_high` — amount=100_000 → dim 0 = 1.0
- `test_vectorize_unknown_mcc` — MCC "9999" → dim 12 = 0.5 (default)
- `test_vectorize_known_mcc` — MCC "7995" → dim 12 = 0.85

**repository/reference.rs:**
- `test_knn_all_fraud` — synthetic dataset where 5 nearest are all fraud → score=1.0, approved=false
- `test_knn_all_legit` — 5 nearest all legit → score=0.0, approved=true
- `test_knn_threshold_3_of_5` — 3/5 fraud = 0.6 → approved=false (boundary)
- `test_knn_threshold_2_of_5` — 2/5 fraud = 0.4 → approved=true (boundary)

### Integration Tests (tests/integration.rs)

Loads real AppState once via `once_cell::sync::Lazy<AppState>`.

- `test_ready_endpoint` — GET /ready → HTTP 200
- `test_legit_transaction_from_docs` — exact legit payload from detection-rules.md → approved=true, fraud_score=0.0
- `test_fraud_transaction_from_docs` — exact fraud payload → approved=false, fraud_score=1.0
- `test_null_last_transaction_no_panic` — last_transaction: null → valid 200 response
- `test_malformed_missing_field` — missing `merchant` → HTTP 422
- `test_malformed_invalid_timestamp` — `requested_at: "not-a-date"` → HTTP 422

### Regression Tests (tests/regression.rs)

- `test_high_velocity_customer` — tx_count_24h=20 (max), known merchant, low amount → should be close to approved
- `test_suspicious_value_spike` — amount=9505.97 with avg_amount=81.28, unknown merchant → approved=false
- `test_unknown_merchant_far_from_home` — unknown merchant + km_from_home=952 → high fraud_score
- `test_first_time_customer` — last_transaction: null, modest amount, known merchant → likely approved
- `test_all_fraud_signals` — all dimensions at max fraud values → approved=false, fraud_score >= 0.6

## Dependencies (Cargo.toml)

```toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
half = "2"              # f16 type
smallvec = "1"          # SmallVec for KNN results (stack-allocated)
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"] }
dotenvy = "0.15"
tower-http = { version = "0.5", features = ["trace"] }

[dev-dependencies]
axum-test = "15"        # HTTP integration test helpers
once_cell = "1"         # Lazy AppState initialization in integration tests
```

## Non-Goals

- No database — reference data is read-only, loaded at startup, never written
- No caching layer — each request independently computes KNN
- No authentication or rate limiting
- No approximate search (HNSW/IVF) — exact brute force for correctness
- No dynamic configuration reload
