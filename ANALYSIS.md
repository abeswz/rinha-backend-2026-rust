# Remote Test Analysis + Optimization Roadmap

**Updated**: 2026-05-30  
**Baseline remote** (clean redesign, commit `c8141fa`): score **3588** / p99 **99.44ms** / FP **23** / FN **0**  
**Baseline local**: score 6000 / p99 0.24ms / FP 0 / FN 0

---

## 1. Decomposing the score gap

```
Current:  p99_score=1002  +  detection_score=2586  =  3588
Max:      p99_score=3000  +  detection_score=3000  =  6000
Gap:                -1998                    -414      -2412
```

Two independent problems. Fix independently.

### 1.1 p99 = 99ms → p99_score = 1002

Formula: `K × log10(T_max / max(p99, p99_MIN))` where K=1000, T_max=1000ms, p99_MIN=1ms.

```
log10(1000 / 99.44) = log10(10.056) = 1.002  →  p99_score = 1002  ✓
```

Score potential at each target:

| p99 target | p99_score | gain vs current |
|---|---|---|
| 99ms (current) | 1002 | — |
| 10ms | 2000 | +998 |
| 1ms | 3000 | +1998 |

Every 10× speedup = +1000 points. The log scale means the first 10× (99→10ms) is worth as much as the next 10× (10→1ms).

**Root cause**: Mac Mini 2014 — dual-core i5 2.6GHz, 4 logical threads. Each instance gets 0.45 CPU = ~45% of one logical core. Under 250 VU burst, throughput ceiling is hit and requests queue. Local machine is much faster, so queueing never appears locally.

### 1.2 FP = 23 → detection_score = 2586

```
E = 1×FP + 3×FN + 5×Err = 1×23 = 23
ε = 23/54100 = 4.26e-4  (below ε_MIN=0.001, so rate_component saturates at 3000)
absolute_penalty = -300 × log10(1+23) = -300 × 1.3802 = -414.06
detection_score  = 3000 - 414 = 2586  ✓
```

Rate is already perfect — ε < ε_MIN means FP rate is so low it doesn't affect the rate term. The entire -414 loss comes from the absolute penalty on E=23.

Fix FP=0 → detection_score=3000 → **+414 points guaranteed**.

**Root cause**: Remote test dataset differs from local (`edge_case_count=645` remote vs `797` local). For 23 cases, brute-force exact KNN gives fraud_count≤1 (approved), but our IVF search returns wrong neighbors → fraud_count≥3 (denied = FP). This is IVF recall failure: suboptimal centroids cause the true nearest neighbors to fall outside the probed clusters.

---

## 2. Optimization roadmap

Prioritized by impact × implementation cost.

---

### 2.1 Fix FP via better centroids [HIGH IMPACT, MEDIUM COST]

**Impact**: Guaranteed +414 detection score → ~4000 total (holding p99 fixed).

**Mechanism**: `bin/build_index.rs` uses KMeans++ initialized from a 50k sample of the 3M dataset. With K=4096, 50k init vectors = ~12 candidates per cluster on average — poor coverage of cluster geometry. Centroids land suboptimally. Clusters in low-density regions (edge cases, unusual feature combinations) are impure, causing IVF to probe the wrong clusters for those 23 test points.

**Solution**: Rebuild index with `tools/build_ivf.py` (already written, produces IVF2 format):
- MiniBatchKMeans from sklearn (full 3M dataset input, batch size 10k)
- N_INIT=3 (multiple restarts, picks best inertia)
- Output: `resources/ivf_index.bin` (IVF2 format, uncompressed)

Then adapt runtime to read IVF2:
- `data.rs`: change magic check `IVF1` → `IVF2`, adjust block padding format (`i16::MAX` instead of `0`)
- IVF2 is uncompressed, so remove flate2 decompression step (simplifies loading)
- `include_bytes!` path: point to `resources/ivf_index.bin` instead of gzip file

**Validation**: run `tools/eval_recall.py` before/after. Must show recall improvement at NPROBE=5 and NPROBE=24.

---

### 2.2 Reduce p99 via i8 block scan [HIGH IMPACT, MEDIUM COST]

**Impact**: 2× throughput in block scan → more requests processed per second → less queueing under 250 VU → lower p99. Estimated 30–50% p99 reduction (from 99ms toward 50ms). Combined with 2.1, possibly toward 30ms.

**Mechanism**: Current block scan: i16 vectors (scale=10000), AVX2 loads 16 i16 per register = 8 effective f32-equivalent lanes per iter. All 14 features fit in i8 with scale=100 (range [-1.0,1.0] → [-100,100]). dim13 (merchant_avg_amount/10000) has std=0.015 → only ~5 distinct i8 values, but it's the lowest-variance feature and has minimal discriminative power.

i8 AVX2: `_mm256_cvtepi8_epi16` (extends 16×i8 → 16×i16) + `_mm256_madd_epi16` (multiply-accumulate 16 pairs) = 32 i8 per register → **2× throughput**.

**Changes**:
- `tools/build_ivf.py`: change quantize scale 10000→100, dtype int16→int8
- `data.rs`: change `AVec<i16>` to `AVec<i8>` for block storage
- `knn.rs`: rewrite `scan_blocks_avx2`:
  ```rust
  // Load 32 i8 (2 full vectors of dim=14 with 2 padding bytes)
  let v = _mm_loadu_si64(block_ptr);          // load 8 i8
  let v16 = _mm256_cvtepi8_epi16(v128);       // extend to 16 i16
  // accumulate with _mm256_madd_epi16
  ```
- Rebuild index with i8 quantization (can combine with 2.1)

**Risk**: Overflow in squared distance. With i8 values up to ±100, squared diff up to 40000 per dim, 14 dims → max sum 560000. Fits in i32 (max ~2B). Accumulate in i32 via `_mm256_add_epi32` after `_mm256_madd_epi16`.

**Validation**: `eval_recall.py` must show identical or better FP/FN before shipping.

---

### 2.3 Vectorize centroid selection [MEDIUM IMPACT, LOW COST]

**Impact**: ~20–25% reduction in fast path (NPROBE=5 cases). Estimated ~5–10ms p99 improvement on top of 2.2.

**Mechanism**: `top_n_centroids_fast` uses insertion sort over K=4096 entries to find top-NPROBE nearest centroids:

```rust
let pos = top[..nprobe].partition_point(|&(b, _)| b <= bits);
top[pos..nprobe].rotate_right(1);  // O(K × NPROBE) ≈ 98k ops
```

Replace with partial sort via `select_nth_unstable` (PDQ = O(K) expected ≈ 4096 ops):

```rust
let mut indexed: [(u32, u16); 4096] = ...;  // stack array, no alloc
// fill with (dist_bits, centroid_idx)
indexed.select_nth_unstable_by_key(nprobe - 1, |&(b, _)| b);
indexed[..nprobe].sort_unstable_by_key(|&(b, _)| b);
```

No heap allocation. Benchmark before committing — PDQ is O(N) expected but has higher constant than insertion sort for very small N. At K=4096 and NPROBE=5..24, PDQ wins.

---

### 2.4 Worker thread tuning [MEDIUM IMPACT, NO COST]

**Context**: 0.45 CPU on dual-core 2.6GHz = ~45% of one logical core. Each tokio worker thread competes for this slice. With 4 workers, context switching overhead adds up — OS scheduler gives each worker ~0.45/4 = 11% of one core.

**Hypothesis**: 2 workers may outperform 4 at 0.45 CPU. Less context switch, same max throughput (throughput is CPU-bound, not parallelism-bound at this scale).

**Test**: Two builds — `worker_threads(2)` vs `worker_threads(4)` — same image, benchmark remotely. No code change beyond runtime config.

---

## 3. What NOT to do

| Idea | Why not |
|---|---|
| Raise fraud threshold (fraud_count ≥ 4) | Converts FP→FN at 3× penalty weight. E goes up, not down |
| HNSW | N=3M, 32-NN graph ≈ 1.2GB RAM. Blows 168MB limit |
| PQ (Product Quantization) | D=14 doesn't split into meaningful sub-vectors. Marginal gain, precision loss |
| More API instances | Budget fixed at 350MB / 0.95 CPU. Can't add meaningful third instance |
| Blocking on FP during NPROBE=24 | Already doing this. The 23 FP are from wrong neighbor search, not threshold |

---

## 4. Priority order

```
1. Rebuild index with sklearn (2.1)     → +414 detection score (guaranteed)
2. Validate recall with eval_recall.py  → safety gate before shipping
3. i8 block scan (2.2)                  → ~2x block scan throughput
4. select_nth_unstable (2.3)            → ~20% fast path speedup
5. Worker thread A/B test (2.4)         → test 2 vs 4 workers remotely
```

Steps 1+2 are prerequisites for 3 (index rebuild needed for both). Steps 3+4 are independent of each other and can be done in parallel.

---

## 5. Score history

| Commit | Architecture | p99 remote | FP | FN | Score |
|---|---|---|---|---|---|
| `87ae940` | Rust LB + spawn_blocking(2) + LightGBM model | 210ms | 23 | 0 | 3263 |
| `a586fa8` | nginx + IVF pure | 84ms | — | — | 4071 |
| `c8141fa` | HAProxy TCP + inline IVF + 4 workers | **99ms** | **23** | **0** | **3588** |

**p99 improved** from 210ms to 99ms (+1 point in log scale) — spawn_blocking bottleneck eliminated. LightGBM model removed (was causing I-cache pollution). The FP count (23) is identical to 87ae940, suggesting FP comes from dataset characteristics, not model interference — it's an IVF recall problem present in both versions.

**Next target**: Fix IVF recall (path to FP=0) + reduce p99 to 10ms → projected score ~5000.
