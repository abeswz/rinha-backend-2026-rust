# Flat IVF + int16 + Block Scan — p99 Reduction Design

**Date:** 2026-05-17  
**Goal:** Reduce competition p99 from 84ms to 8–30ms (+700–1500 pts)  
**Scope:** IVF memory layout rewrite + Python index rebuild + static responses

---

## Problem

Competition p99 = 84ms despite local p99 = 0.29ms. Root cause: Linux CFS CPU throttling (0.45 CPU cgroup = 45ms quota per 100ms period). Under peak load (450 req/s per instance), the KNN computation exhausts the CPU quota in bursts, starving subsequent requests for the remainder of the period.

Current `Vec<Vec<([f16;16], u8)>>` layout causes pointer chasing between cluster allocations → cache misses under throttle amplify the problem. Each cluster scan jumps between heap allocations instead of reading sequential memory.

Reference implementation analysis (`memory/rinha-2026-rust`) confirmed: flat aligned arrays + int16 quantization + 8-vector block batching is the correct approach to reduce CPU time per KNN request and avoid quota exhaustion.

---

## Success Criteria

- Local `make bench` (Docker, cgroup-constrained): p99 < 30ms
- Competition: p99 < 30ms, zero HTTP errors
- Detection accuracy: FP+FN ≤ 20 total (no regression vs current nprobe_fast=5)
- Memory: total index ≤ 90 MB per instance (within 170 MB limit)
- All existing tests pass

---

## Architecture

### Index Structure

Replace current `IvfIndex` with flat memory layout:

```rust
pub struct IvfIndex {
    k: usize,           // number of centroids (4096)
    n: usize,           // total vectors (3M)
    nprobe_fast: usize,
    nprobe_slow: usize,

    // Column-major centroids: centroids[d * k + ci] = dim d of centroid ci
    // Contiguous per dimension → SIMD loads K centroids per dimension
    centroids: Vec<f32>,   // 14 × 4096 × 4 bytes = 229 KB

    // CSR-style offsets: cluster ci spans [offsets[ci], offsets[ci+1])
    offsets: Vec<u32>,     // (4096 + 1) × 4 bytes = 16 KB

    // Flat labels, one per vector, padded to 8-multiple
    labels: Vec<u8>,       // ~3M bytes = 3 MB

    // Dim-major blocks: blocks[(block_idx * 14 + d) * 8 .. +8] = 8 × i16
    // One block = 8 vectors, dimension d is contiguous for all 8
    blocks: Vec<i16>,      // ~375K blocks × 14 × 8 × 2 = 84 MB
}
```

**Total memory per instance: ~87 MB** (fits within 170 MB limit with ~83 MB headroom for binary + stack).

### Quantization

- Encoding: `i16 = round(f32 * 10_000)`. Range `[-1.0, 1.0]` maps to `[-10000, 10000]`, fits in i16 (max ±32767).
- The sentinel `-1.0` (null `last_transaction`) maps to `-10000` — preserved exactly.
- Decoding in SIMD: multiply by `0.0001f32` constant.

### Binary Format: IVF2

```
[0..4]   magic: b"IVF2"
[4..8]   n: u32 LE (total vectors)
[8..12]  k: u32 LE (centroids)
[12..16] d: u32 LE (dimensions = 14)
[16..]   centroids: f32 LE array [d * k]
         offsets:   u32 LE array [k + 1]
         labels:    u8 array [padded_n]  (padded to multiple of 8)
         blocks:    i16 LE array [total_blocks * d * 8]
```

Replaces current IVF1-format `ivf_index.bin`. Python builder outputs IVF2; Rust loader reads IVF2.

---

## SIMD Centroid Scan (Column-Major)

Query: `[f32; 14]`. Output: sorted top-nprobe centroid indices.

```
for d in 0..14:
    q_d = broadcast(query[d])                    // _mm256_set1_ps
    col = centroids[d*K .. d*K+K]               // contiguous slice
    for chunk of 8 in col:
        c   = load_8f32(chunk)                   // _mm256_loadu_ps
        sq  = fmadd(q_d - c, q_d - c, acc)      // accumulate sq dist
        store acc

select_nth_unstable top nprobe from dist buffer
```

**Advantage over current**: instead of 4096 structs with 14 scattered floats, each dim pass reads contiguous 16 KB — fits in L1 cache. 14 passes of 512 AVX2 loads vs 4096 iterations with scattered access.

Thread-local `CENTROID_BUF: Vec<f32>` (capacity K) replaces current `Vec<(f32, usize)>`.

---

## SIMD Block Scan with Early Termination

One block = 8 vectors. `blocks[(block_idx * 14 + d) * 8 .. +8]` = 8 × i16 for dim d.

```
for each probed cluster:
    for each block in cluster:
        partial = [0.0f32; 8]                     // AVX2 register

        for d in 0..14:
            raw    = load_8i16(blocks[block*14*8 + d*8])  // _mm_loadu_si128
            f32s   = cvt_i16_to_f32(raw) * 0.0001        // widen + scale
            diff   = broadcast(query[d]) - f32s
            partial = fmadd(diff, diff, partial)          // accumulate

            // early termination check after dim 6
            if d == 6 && all_8(partial > threshold):
                break                                      // skip rest of block

        // update top-K heap with surviving distances
        // update threshold = worst heap entry
```

**Early termination**: after 7 of 14 dims, if all 8 partial distances exceed the current worst neighbor, the block is skipped entirely. Expected: 30–60% blocks skipped in slow path once heap fills.

---

## Static Responses

With k=5, `fraud_count` ∈ {0..5} → 6 possible responses. Build at startup:

```rust
struct StaticResponses {
    bodies: [Bytes; 6],
}

// index = fraud_count
// 0 → {"approved":true,"fraud_score":0.0}
// 1 → {"approved":true,"fraud_score":0.2}
// 2 → {"approved":true,"fraud_score":0.4}
// 3 → {"approved":false,"fraud_score":0.6}
// 4 → {"approved":false,"fraud_score":0.8}
// 5 → {"approved":false,"fraud_score":1.0}
```

Handler indexes directly: `RESPONSES[fraud_count]`. Zero `serde_json` on hot path. Small win (~1–5µs per request) but eliminates heap allocation.

`FraudDecision` changes to carry `fraud_count: usize` instead of `fraud_score: f32` for the response path. Score still computed as `count as f32 / 5.0` for timeout fallback.

---

## Inline KNN Evaluation (Conditional)

After flat layout is implemented and validated locally:

1. Run `make bench` with Docker cgroup constraints
2. Check `docker stats` for CPU% per instance at peak
3. Measure KNN time: instrument with `std::time::Instant` in debug build, or observe from p99 delta

**Decision gate:**
- If Stage 2 KNN (nprobe=24, ~18K vectors) completes in < 0.5ms under cgroup → remove `spawn_blocking`, use `tokio::task::block_in_place`
- If Stage 2 > 0.5ms → keep `spawn_blocking` with `max_blocking_threads=2`

`block_in_place` eliminates spawn_blocking scheduling overhead (~100–500µs) at the cost of briefly blocking a Tokio worker. Safe only if KNN is fast enough that the reactor recovers before the next I/O event needs handling.

The `TOKIO_MAX_BLOCKING_THREADS` constant and its test remain regardless; they guard against accidental changes.

---

## K=4096 Centroids

Increasing K from 3000 to 4096:
- Average cluster size: 3M / 4096 ≈ 732 vectors (vs 1000 currently)
- Fewer vectors per cluster → faster scan per probe
- More clusters → better routing precision → fewer false cluster misses
- Centroid scan cost: +33% more centroids to evaluate, offset by column-major SIMD efficiency

Python builder uses `sklearn.cluster.MiniBatchKMeans(n_clusters=4096)` or equivalent. Build time: 3–8 min (unchanged, already done at Docker build time).

---

## Files Changed

| File | Change |
|------|--------|
| `tools/build_ivf.py` | K=4096, int16 quantization, IVF2 format output |
| `src/repository/ivf.rs` | Full rewrite: flat arrays, column-major centroid scan, block scan with early termination, IVF2 loader |
| `src/web/handlers.rs` | Static responses, index into `RESPONSES[fraud_count]` |
| `src/domain/fraud.rs` | `FraudDecision`: add `fraud_count: usize` field |
| `src/usecase/score_fraud.rs` | Return `fraud_count` alongside `fraud_score` |
| `src/lib.rs` | Wire `StaticResponses` into `AppState` |

---

## Testing Strategy

### Unit tests (TDD, before implementation)

- `test_ivf2_load_parses_header` — magic, n, k, d fields
- `test_ivf2_load_rejects_ivf1_magic` — backward compat guard
- `test_centroid_scan_column_major_matches_brute_force` — correctness vs naive impl
- `test_block_scan_8vec_matches_brute_force` — correctness for 8-vector block
- `test_early_termination_skips_high_distance_block` — early exit fires correctly
- `test_static_responses_all_6_are_valid_json` — response bodies are well-formed
- `test_knn_adaptive_accuracy_regression` — FP+FN ≤ 20 on known test vectors

### Integration tests

- Existing integration + regression tests must pass unchanged
- `make bench` locally: p99 < 30ms, ERR=0

### Validation gate

Run `make bench` before any remote submission. Only submit if local p99 < 30ms and ERR=0.

---

## Estimated Score Impact

| Scenario | p99 | p99_score | det_score | final |
|----------|-----|-----------|-----------|-------|
| Current (remote) | 84ms | 1074 | ~2616 | ~3690 |
| Flat IVF only | 20ms | 1699 | ~2616 | ~4315 |
| + inline KNN | 10ms | 2000 | ~2616 | ~4616 |
| Best case | 5ms | 2301 | ~2616 | ~4917 |

Detection score assumes FP=3, FN=5 (local measurement with nprobe_fast=5). May improve with K=4096 due to better cluster routing.
