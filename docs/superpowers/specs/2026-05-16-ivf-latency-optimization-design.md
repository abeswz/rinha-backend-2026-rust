# IVF Latency Optimization — Design Spec
**Date:** 2026-05-16
**Status:** Approved

## Context

Current remote score: 3809.43 (p99=83ms, FP=1, FN=2, detection near-perfect).
Local score: 5729 (p99=0.35ms). Gap = ~1920 pts, all latency.

Root cause: Linux CFS CPU throttling (0.45 CPU = 45ms/100ms period) + `spawn_blocking` queue (2 blocking threads). Under high load, the container exhausts CPU quota mid-period and stalls up to 55ms. Combined with blocking thread pool queue, p99 spikes to 83ms.

Detection is essentially perfect (3 errors/54100). All improvement potential is in latency.

## Goal

Push remote p99 from 83ms toward 10ms or better.

Expected score gain: +920 pts minimum (83ms→10ms), up to +1920 pts if p99 reaches 1ms.

## Changes

### 1. IVF Rebuild — `tools/build_ivf.py`

Change `K`: 1732 → 3000.

- Avg cluster size: 1730 → 1000 vectors
- `nprobe_fast=3` → scans 3×1000=3000 vectors (vs current 5×1730=8650, 3x less work)
- `nprobe_slow=12` → halved from 24 (ambiguous cases still accurate, just cheaper)
- K=3000 produces tighter clusters → better locality → nprobe_fast=3 still finds correct neighbors

Make `nprobe_fast` configurable as env var `IVF_NPROBE_FAST` (default 3) in `src/config.rs` and `src/repository/ivf.rs`.

The Python script uses `MiniBatchKMeans`. Keep `n_init=3`, `random_state=42` for reproducibility. `batch_size=50_000` unchanged.

### 2. SIMD Centroid Distance — `src/repository/ivf.rs`

`centroid_sq_dist` is currently scalar (f32 loop over 14 dims). Replace with AVX2 path (same pattern as existing `vec_sq_dist_simd`):

- Pad centroid to 16 dims on the call site (stack, zero-cost)
- Load 8 f32 × 2 iterations via `_mm256_loadu_ps`
- Compute squared differences with `_mm256_sub_ps` + `_mm256_mul_ps`
- Horizontal sum

Scalar fallback preserved for non-AVX2 hosts. `RUSTFLAGS` already enables AVX2+F16C in Dockerfile.

With K=3000, centroid scan cost doubles vs K=1732 in raw count. SIMD reduces centroid distance cost ~4x, making centroid scan cheaper than the list scan — balanced.

### 3. Remove `spawn_blocking` — `src/web/handlers.rs`

Replace:
```rust
tokio::time::timeout(
    Duration::from_millis(1600),
    tokio::task::spawn_blocking(move || state.use_case.execute(&tx)),
)
.await
.ok()
.and_then(|r| r.ok())
.unwrap_or(FraudDecision { approved: true, fraud_score: 0.0 })
```

With:
```rust
state.use_case.execute(&tx)
```

KNN with nprobe_fast=3, K=3000 scans ~3000 SIMD f16 comparisons → ~100-200μs on a throttled core. Safe on a Tokio worker thread (sub-1ms). Eliminates thread handoff latency (~100-200μs overhead per request) and blocking pool queue entirely.

The timeout + approved=true fallback is no longer needed. Removing it eliminates potential false negatives from timed-out fraud detection.

### 4. Thread + Resource Config

**`src/main.rs`:**
- `worker_threads`: 2 → 4
- Remove `max_blocking_threads` override (use Tokio default; no spawn_blocking in hot path)

**`docker-compose.yml`:**
- nginx: 0.10 CPU → 0.05 CPU (nginx only proxies, Unix socket, very lightweight)
- api1 + api2: 0.45 CPU each → 0.475 CPU each
- Total: 0.05 + 0.475 + 0.475 = 1.0 CPU ✅ (within contest limit)
- Memory unchanged: nginx 10MB, each API 170MB (within 350MB total)

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| nprobe_fast=3 increases errors | Low | K=3000 tighter clusters compensate; edge cases (797) already near-irreducible |
| Inline KNN stalls Tokio workers | Low | 3000 SIMD ops ≈ 100-200μs; measured local p99 = 0.35ms |
| K=3000 rebuild accuracy regression | Low | Run local k6 test after build, verify FP/FN unchanged |

## Out of Scope

- VP Tree / HNSW (memory constraints)
- u8 quantization (complexity vs gain unclear)
- Adaptive threshold changes (already well-tuned)
- Detection logic changes (near-perfect, YAGNI)

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| p99 remote | 83ms | ~5-15ms |
| p99_score | 1080 | ~1500-2000 |
| detection_score | 2729 | 2729 (unchanged) |
| final_score | 3809 | ~4229-4729 |
| per-request KNN cost | ~8650 SIMD ops | ~3000 SIMD ops |
