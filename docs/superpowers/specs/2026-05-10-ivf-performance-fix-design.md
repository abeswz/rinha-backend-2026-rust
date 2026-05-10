# Design: IVF Performance Fix

**Date:** 2026-05-10  
**Status:** Approved  
**Context:** `docs/reqs/performance-problem.md`

---

## Problem

Current brute-force KNN blocks Tokio async thread and costs ~130ms per request over 3M × 14-dim vectors. Contest target: 900 req/s within 1 CPU + 350MB budget across 2 instances + load balancer.

Two distinct bugs:

| Problem | Cause | Impact |
|---|---|---|
| Async blocking | `execute()` runs CPU-bound on Tokio thread | Serializes requests per instance |
| Algorithm | Brute-force O(N×D): 42M ops/request | ~7 req/s max on contest budget |

Current state: ~15 req/s total. Target: 900 req/s.

---

## Constraints

- 2 API instances + load balancer (round-robin, nginx)
- docker-compose, linux/amd64, bridge network, port 9999
- Total budget: 1 CPU + 350MB RAM across all services
- Per instance: ~0.475 CPU + ~170MB RAM
- **Test data is pre-labeled with exact brute-force KNN** — IVF recall errors become real FP/FN
- Reference vectors do not change during test — pre-processing in Dockerfile is valid and encouraged
- Payload lookup tables are forbidden — query must be computed from the request vector at runtime

---

## Approach: spawn_blocking + IVF

Two independent phases applied in sequence.

### Phase 1 — spawn_blocking (bug fix)

Wrap the CPU-bound use case call to offload from Tokio async thread to blocking thread pool.

**File:** `src/web/handlers.rs`

```rust
// Before
let decision = state.use_case.execute(&tx);

// After
let decision = tokio::task::spawn_blocking({
    let state = Arc::clone(&state);
    move || state.use_case.execute(&tx)
}).await.map_err(|e| AppError::Internal(e.to_string()))??;
```

This alone does not fix throughput (130ms/request remains), but eliminates request serialization and is a required correctness fix regardless of algorithm.

### Phase 2 — IVF Index (algorithmic)

Replace brute-force with Inverted File Index pre-built at Docker image build time.

**Key parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| K (clusters) | 1732 | √3_000_000 — IVF standard |
| nprobe | 8 (configurable via `IVF_NPROBE` env) | Throughput/recall balance (see trade-offs) |
| Distance metric | Squared Euclidean | Same as current brute-force |
| Vector storage | f16 | Same as current reference array |

**Throughput vs recall trade-off:**

| nprobe | time/req | throughput (0.95 CPU) | recall est. | score impact |
|---|---|---|---|---|
| 4 | ~0.37ms | ~2567 req/s | ~93% | detection penalty significant |
| **8** | **~0.67ms** | **~1418 req/s ✓** | **~97%** | **recommended** |
| 16 | ~1.3ms | ~731 req/s ✗ | ~99% | below 900 req/s → queue builds → HTTP errors |

nprobe=16 falls below the 900 req/s target. Queue growth causes p99 > 2000ms and HTTP errors (weight 5) — worse than the recall improvement from 97% → 99%.

---

## Architecture

```
Docker build:
  resources/references.json.gz
        ↓
  tools/build_ivf.py  (mini-batch k-means, scikit-learn)
        ↓
  resources/ivf_index.bin  (baked into image)

Runtime:
  HTTP request
    → vectorize (14 dims, deterministic, unchanged)
    → spawn_blocking
        → IvfIndex::knn(query, k=5, nprobe=8)
            → distance to 1732 centroids  [O(24K ops)]
            → brute-force on top-8 clusters (~13.8K vectors) [O(193K ops)]
            → top-5 labels
    → fraud_score = fraud_labels / 5
    → approved = fraud_score < 0.6
```

Operations per request: ~217K vs 42M brute-force = **~193× speedup**.  
Estimated p99 at 900 req/s: ~1ms (well within 2000ms cut).

---

## IVF Binary Format

File: `resources/ivf_index.bin`

```
[4B  u32 LE] K — number of clusters
[4B  u32 LE] D — dimensions (14)
[K × D × 4B f32] centroids — row-major
[K × 4B u32] list sizes — number of vectors per cluster
[for each cluster i: list_sizes[i] × (D × 2B f16 + 1B u8)]
  — vector (f16 × 14) followed by label (u8)
```

**Memory footprint:**

| Component | Size |
|---|---|
| Centroids | 1732 × 14 × 4B = 97KB |
| Lists (vectors + labels) | 3M × (28B + 1B) = 87MB |
| **Total** | **~87MB** |

Identical to current reference array (84MB vectors + 3MB labels = 87MB). No memory regression.

---

## Code Changes

```
src/
  repository/
    ivf.rs          NEW — IvfIndex struct, load(path) -> IvfIndex, knn(query, k, nprobe) -> SmallVec<[u8;5]>
    reference.rs    MODIFY — remove brute-force knn(), delegate to IvfIndex; keep ReferenceRepository struct
  web/
    handlers.rs     MODIFY — spawn_blocking wrap

tools/
  build_ivf.py      NEW — offline k-means builder (Python + scikit-learn)
  requirements.txt  NEW — numpy, scikit-learn

Dockerfile          MODIFY — add Python build stage for ivf_index.bin
.gitignore          MODIFY — add resources/ivf_index.bin (generated artifact)
```

No changes to: domain logic, DTOs, scoring, vectorizer, use cases, tests.

---

## Offline Build Script (tools/build_ivf.py)

```python
# Runs in Docker build stage (not in runtime image)
# Input:  resources/references.json.gz
# Output: resources/ivf_index.bin

# Algorithm:
#   1. Load 3M vectors (f32) + labels (u8)
#   2. MiniBatchKMeans(n_clusters=1732, batch_size=50000, n_init=3)
#   3. Assign each vector to nearest centroid
#   4. Write binary IVF format (see above)
```

Build time: ~3–8 minutes depending on CPU. Runs once at `docker build`, not at runtime.

---

## Dockerfile Build Stage

```dockerfile
FROM python:3.12-slim AS ivf-builder
WORKDIR /build
COPY resources/references.json.gz resources/
COPY tools/ tools/
RUN pip install --no-cache-dir numpy scikit-learn && \
    python tools/build_ivf.py

FROM rust:1.82-slim AS rust-builder
# ... existing Rust build ...
COPY --from=ivf-builder /build/resources/ivf_index.bin resources/

FROM debian:bookworm-slim
COPY --from=rust-builder /app/fraud-detection /app/
COPY --from=ivf-builder /build/resources/ivf_index.bin /app/resources/ivf_index.bin
```

---

## Error Handling

- If `ivf_index.bin` missing at startup → panic with clear message (build misconfigured)
- If `spawn_blocking` join error → return `AppError::Internal` → HTTP 500
- HTTP 500 has weight 5 in scoring — all error paths must be handled before they become 500s

Per spec guidance: if processing fails, return `approved: true, fraud_score: 0.0` (FP weight=1) rather than HTTP 500 (weight=5).

---

## Testing

- Existing unit tests for vectorizer, scoring, detection rules: unchanged
- Add integration test: load IVF index from test fixture, query known vectors, assert ≥97% of top-5 label sets match brute-force exactly (nprobe=8 recall target)
- Local load test (`make load`): validate p99 < 100ms and throughput > 900 req/s before submission

---

## Out of Scope

- Vector result cache (LRU): deferred — evaluate after IVF is profiled in load test
- SIMD F16C: not needed with IVF speedup
- HNSW: memory budget insufficient (250–500MB index)
- VP Tree: tops at ~475 req/s — below 900 req/s target
