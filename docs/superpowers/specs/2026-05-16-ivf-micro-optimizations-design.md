# IVF Micro-Optimizations Design

**Goal:** Restore stability (p99 < 100ms, 0 HTTP errors) AND push p99 toward sub-1ms via targeted changes.

---

## Problem

### Critical regression (remote test, score -3802.66)

Image `79e9d85` (commit `570a83b perf(handler): remove spawn_blocking`) produced:
- p99: 2001ms (k6 timeout, p99_score cut = -3000)
- http_errors: 2535 (7.47% failure rate)
- final_score: -3802.66 (was +4273.15 before)

**Root cause:** Removing `spawn_blocking` caused Tokio worker threads to run CPU-bound KNN directly. Under Docker cgroup CPU throttling (0.475 CPU quota), the kernel throttles a worker thread mid-KNN. The Tokio reactor cannot schedule new tasks. Requests queue, hit the 2001ms k6 timeout, return as HTTP errors.

This problem does NOT appear locally because the local machine runs at full CPU speed without cgroup throttling.

### Latency optimizations (carry forward)

After restoring spawn_blocking, remaining micro-optimizations reduce p99 further:
- 3000 stack allocs + memcpys per request (centroid padding in hot loop)
- CENTROID_BUF undersized (capacity=2048 < K=3000, reallocs on first request per thread)
- nprobe_fast=3 scans 3000 vectors (room to cut to 2000 with nprobe_fast=2)

---

## Changes

### 0. Restore spawn_blocking in handler (CRITICAL) (`src/web/handlers.rs`)

**Current (broken):**
```rust
let decision = state.use_case.execute(&tx);
```

**New:**
```rust
let decision = tokio::task::spawn_blocking(move || state.use_case.execute(&tx))
    .await
    .expect("KNN task panicked");
```

No timeout — KNN is now sub-1ms (K=3000 + SIMD). A panic is a real bug; let it surface.

**Why no timeout:** Original timeout (1600ms) existed to avoid HTTP errors from slow KNN. With K=3000 + SIMD, KNN takes ~0.3ms. Timeout adds complexity with no benefit. If KNN somehow hangs, the panic is the correct signal.

**worker_threads stays at 4** (NOT changed to 2 as originally planned). With spawn_blocking, worker threads handle I/O and task scheduling — 4 is correct. Blocking pool handles KNN concurrently.

### 1. Pre-pad centroids to [f32;16] at load time (`src/repository/ivf.rs`)

**Current:** `centroids: Vec<[f32;14]>`. Every call to `centroid_sq_dist` creates `[f32;16]` on stack and copies 14 elements before calling SIMD.

**New:** `centroids: Vec<[f32;16]>`. Load time pads dims 14-15 to 0.0. `centroid_sq_dist` signature becomes `(&[f32;16], &[f32;16])` — calls SIMD directly, no internal copy.

**Impact:** Eliminates 3000 × (stack alloc + 56-byte copy) per request. Pure AVX2 load-load-sub-mul in the hot loop.

**Binary format unchanged** — on-disk format still 14×f32 per centroid; padding happens during `load()`.

**Affected:** `IvfIndex` struct, `load()` centroid parsing, `centroid_sq_dist`. Test `test_centroid_sq_dist_correctness` centroid arg changes from `[f32;14]` to `[f32;16]`.

### 2. CENTROID_BUF capacity 2048→3072 (`src/repository/ivf.rs`)

**Current:** `Vec::with_capacity(2048)`. K=3000 triggers realloc on first request per thread.

**New:** `Vec::with_capacity(3072)`. No realloc ever.

### 3. nprobe_fast=2 in docker-compose (`docker-compose.yml`)

**Current:** `IVF_NPROBE_FAST` not set → default 3 → 3 clusters × ~1000 vectors = 3000 comparisons on fast path.

**New:** `IVF_NPROBE_FAST=2` → 2000 comparisons, ~33% less blocking-thread CPU time per request.

**Risk:** May increase FN. If FN > 20 in load test, revert to `IVF_NPROBE_FAST=3` — no recompile needed.

---

## Implementation Order

1. Restore spawn_blocking first — fixes the regression
2. Pre-pad centroids + CENTROID_BUF — perf improvements
3. nprobe_fast=2 in docker-compose — last, easiest to revert

---

## Testing

- `cargo test` — all 34 tests must pass after each step
- `cargo clippy --all-targets --all-features -- -D warnings` — zero warnings
- `make load` locally after all changes — compare p99 and FP/FN vs pre-regression baseline (FP=5, FN=9)
- Accept FN ≤ 20 with nprobe_fast=2; revert to 3 otherwise

---

## Estimated Impact (after all changes)

| Change | Effect |
|--------|--------|
| Restore spawn_blocking | Fixes p99 2001ms → ~5-15ms, eliminates HTTP errors |
| Pre-padded centroids | -0.3ms off KNN time in blocking pool |
| CENTROID_BUF capacity | Eliminates realloc tail latency |
| nprobe_fast=2 | -0.1ms off KNN time, ~33% less CPU per fast-path call |

Remote p99 target: 5-15ms (was 83ms before IVF). Sub-1ms is possible if the blocking pool overhead is negligible.

---

## Rollback

| Change | Rollback |
|--------|---------|
| Restore spawn_blocking | Non-negotiable — do not revert |
| Pre-padded centroids | git revert |
| CENTROID_BUF capacity | git revert |
| nprobe_fast=2 | `IVF_NPROBE_FAST=2→3` in docker-compose |
