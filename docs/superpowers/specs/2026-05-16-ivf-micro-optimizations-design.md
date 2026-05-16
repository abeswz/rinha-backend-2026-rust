# IVF Micro-Optimizations Design

**Goal:** Push p99 from ~2ms to sub-0.90ms remotely via four targeted changes — no topology or memory budget changes.

---

## Problem

After IVF K=3000 + SIMD centroid distance + spawn_blocking removal, estimated p99 ~1.5-3ms due to:
- 3000 stack allocs + memcpys per request (centroid padding in hot loop)
- CENTROID_BUF undersized (capacity=2048 < K=3000, reallocs on first request per thread)
- 4 worker_threads on 0.475 CPU = high scheduling jitter at p99
- nprobe_fast=3 scans 3000 vectors on fast path (room to cut to 2000)

---

## Changes

### 1. Pre-pad centroids to [f32;16] at load time (`src/repository/ivf.rs`)

**Current:** `centroids: Vec<[f32;14]>`. Every call to `centroid_sq_dist` creates `[f32;16]` on the stack and copies 14 elements before calling SIMD.

**New:** `centroids: Vec<[f32;16]>`. Load time pads dims 14-15 to 0.0. `centroid_sq_dist` signature becomes `(&[f32;16], &[f32;16])` — calls SIMD directly, no internal copy.

**Impact:** Eliminates 3000 × (stack alloc + 56-byte copy) per request. Pure AVX2 load-load-sub-mul in the hot loop.

**Binary format unchanged** — on-disk format still 14×f32 per centroid; padding happens during `load()`.

**Affected:** `IvfIndex` struct, `load()` centroid parsing, `centroid_sq_dist`, `centroid_sq_dist_simd` (signature stays `(&[f32;16], &[f32;16])` — no change needed). Test `test_centroid_sq_dist_correctness` centroid arg changes from `[f32;14]` to `[f32;16]`.

### 2. CENTROID_BUF capacity 2048→3072 (`src/repository/ivf.rs`)

**Current:** `Vec::with_capacity(2048)`. K=3000 triggers realloc on first request per thread.

**New:** `Vec::with_capacity(3072)`. No realloc ever.

### 3. worker_threads 4→2 (`src/main.rs`)

**Current:** 4 threads on 0.475 CPU = kernel preempts each thread ~88% of the time. High p99 tail from scheduling jitter.

**New:** 2 threads → each gets ~24% CPU time, fewer preemptions, lower p99 tail.

**Throughput note:** At expected load (~450 req/s per instance), 2 threads with sub-1ms processing time is sufficient. Tokio is non-blocking; threads are not blocked waiting.

### 4. nprobe_fast=2 in docker-compose (`docker-compose.yml`)

**Current:** Default `IVF_NPROBE_FAST=3` (3 clusters × ~1000 vectors = 3000 vector comparisons on fast path).

**New:** `IVF_NPROBE_FAST=2` (2 clusters × ~1000 vectors = 2000 comparisons, ~33% less work).

**Risk:** May increase FN (ambiguous queries near cluster boundaries get fewer probes on Stage 1, triggering Stage 2 less accurately). If FN increases significantly after load test, revert to `IVF_NPROBE_FAST=3` — no recompile needed.

**Reversibility:** docker-compose env var only. Zero code change to revert.

---

## Testing

- `cargo test` — all 34 tests must pass
- `cargo clippy --all-targets --all-features -- -D warnings` — zero warnings
- `make load` — compare p99 and FP/FN vs baseline (FP=5, FN=9 @ nprobe_fast=3)
- Accept FN increase up to ~20 before reverting nprobe_fast

---

## Estimated Impact

| Change | p99 reduction |
|--------|--------------|
| Pre-padded centroids | ~0.3ms |
| CENTROID_BUF capacity | ~0.1ms (tail) |
| worker_threads=2 | ~0.5-0.8ms (p99 jitter) |
| nprobe_fast=2 | ~0.1ms |
| **Total** | **~1.0-1.3ms** |

Target: p99 remote ≤ 0.90ms. Saturates `p99_score` at 3000 if p99 ≤ 1ms.

---

## Rollback

| Change | Rollback |
|--------|---------|
| Pre-padded centroids | git revert |
| CENTROID_BUF capacity | git revert |
| worker_threads=2 | git revert |
| nprobe_fast=2 | Change `IVF_NPROBE_FAST=2→3` in docker-compose |
