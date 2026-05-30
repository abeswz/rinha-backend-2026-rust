# Project Progress

> **Rule:** Update this file after every remote test cycle or significant architectural change.
> Record: what changed, why, and the result (score/p99/FP/FN).

---

## Current State

**Main branch:** `main`
**Last updated:** 2026-05-30
**Active architecture:** IVF2 index (faiss KMeans, i16 blocks, scale=3000) + AVX2 scan + HAProxy TCP LB

### Request pipeline

```
JSON → vectorize → IVF KNN (FAST_NPROBE=16)
                        │
              count=0,1 │ → approved
              count=4,5 │ → denied
              count=2,3 → full probe (FULL_NPROBE=24) → count → approved/denied
```

### Latest local result

```
p99: 0.35ms | final_score: 5647 | FP: 2 | FN: 4 | E: 14
detection_score: 2647 | rate_component: 3000 (saturated) | p99_score: 3000
```

---

## Remote Test History

| Date       | Image commit | Dataset edge_cases | p99      | FP | FN | Score    | Notes |
|------------|--------------|--------------------|---------:|----|---:|---------:|-------|
| 2026-05-?  | `f2a276a`    | 645                | 127.17ms | 23 | 0  | 3481.56  | Pure IVF + custom Rust LB |
| 2026-05-?  | `bb5b85f`    | 797                | 366.90ms | 0  | 0  | 3435.45  | AVX2 scan, nginx, slow |
| 2026-05-?  | `a586fa8`    | 797                | 84.83ms  | 0  | 0  | **4071.46** | **Previous best** — worker_threads=2, max_blocking=2 |
| 2026-05-?  | `33baf8f`    | 797                | 105.39ms | 0  | 0  | 3977.21  | multi_thread flavor |
| 2026-05-23 | `c839874`    | 645                | 239.97ms | 51 | 0  | 3105.05  | m2cgen Fraud fast-path active — FPs from overfitting |
| 2026-05-23 | `1afacab`    | 645                | 98.91ms  | 23 | 0  | 3590.69  | Legit fast-path only — improvement vs `f2a276a` same dataset |

> **Note on datasets:** The test server alternates between two datasets.
> Dataset 645 (`edge_case_count=645`): IVF produces 23 intrinsic FPs, max score ~3590.
> Dataset 797 (`edge_case_count=797`): IVF zeroes FPs, max observed score 4071.

---

## Architectural Evolution

### Phase 1 — Baseline (nginx + pure IVF)
- nginx as LB, IVF for all requests
- High p99 (~366ms) due to nginx overhead + synchronous IVF

### Phase 2 — Custom Rust LB + AVX2 IVF
- Replaced nginx with TCP LB in Rust (`bin/lb.rs`)
- AVX2+FMA for centroid scan, POPCNT for label counting
- Adaptive two-stage search: NPROBE=5 fast → NPROBE=24 for ambiguous (count=2,3)
- Best score: 4071 with `a586fa8`

### Phase 3 — m2cgen Legit fast-path (removed 2026-05-23)
- LightGBM trained on `resources/references.json.gz` (3M vectors, 14 dims)
- Exported via m2cgen as inline Rust if-else (`src/fraud/model_gen.rs`, 1.65MB, 19,581 branches)
- **Fraud fast-path disabled** — model overfit local dataset → 51 remote FPs
- Legit fast-path only; all else routes to IVF

### Phase 4 — IVF2 index + i16 quantization (current, 2026-05-30)

**What changed:**

1. **IVF2 index** (Python faiss KMeans, commit `a709437`): replaced the Rust KMeans++ builder
   (`bin/build_index.rs`) with `tools/build_ivf.py` using faiss `Kmeans` (20 iterations on 1M sample).
   Better centroid quality → fewer cluster-miss errors on edge cases.

2. **Block quantization: i8 → i16** (commit `cb14480`): root cause of the local regression.

   The i8 switch introduced 100× more quantization noise (precision 0.01/dim vs 0.0001/dim in old
   i16 system). For borderline vectors with very similar distances, the noise flipped nearest-neighbor
   rankings → FP=76, FN=81, E=319, score=4477.

   Switched to i16 scale=3000 (precision 0.000333/dim, 30× better than i8):
   - Max squared-distance sum: 14 × 6000² = 504M — safely fits in i32 accumulator
   - Block size: 112 → 224 bytes per block; index: ~42 MB → ~84 MB
   - SIMD: `_mm_loadl_epi64` + `_mm_cvtepi8_epi16` → `_mm_loadu_si128` (direct i16 load)

   **Result: FP=76→2, FN=81→4, E=319→14, score=4477→5647 (+1170 pts)**

3. **NPROBE tuning** (tested 2026-05-30): tested FAST_NPROBE ∈ {2, 4, 14, 16} and FULL_NPROBE ∈ {24, 32}.
   No effect on E — the remaining 6 errors (FP=2, FN=4) are intrinsic IVF recall failures where true
   nearest neighbors fall outside the top-N clusters regardless of NPROBE. Settled on FAST=16, FULL=24.

4. **LightGBM hybrid attempt** (attempted and reverted 2026-05-30): tried replacing the full probe
   (FULL_NPROBE=24) with LightGBM for ambiguous cases (count=2,3). At T=0.25: FP exploded to 322
   (model gives P(fraud)≥0.25 for ~94% of legit ambiguous cases). At T=0.50: FN exploded to 451.
   Root cause: LightGBM trained on the full 3M population doesn't generalize to the ambiguous
   IVF subpopulation. Full probe is better here — it uses more IVF-space evidence. **Reverted.**

### Score formula (derived from observations)

```
E = FP + 3×FN
ε = E / N  (N=54100)
rate_component  = max(3000 - 1000×log10(max(ε/0.001, 1)), 0)
absolute_penalty = 300 × log10(1 + E)
detection_score  = rate_component - absolute_penalty
p99_score        = 1000 × log10(1000 / max(p99_ms, 1))
final_score      = p99_score + detection_score
```

Rate component saturates at 3000 when ε ≤ 0.001 (E ≤ 54 for N=54100).
Currently E=14 → rate_component=3000 (saturated). Remaining gap: absolute_penalty from E=14.

---

## Remaining Errors (E=14)

FP=2, FN=4 are intrinsic IVF failures. The true top-5 neighbors for these queries span clusters
ranked 25+ by centroid distance. No amount of NPROBE fixes them without full brute-force scan.

To reach E=0 (detection_score=3000, local score=6000):
- Brute-force fallback for edge cases (797 edge cases × full scan = expensive)
- K=8192 index (smaller clusters → fewer cross-cluster misses, but doubles centroid scan cost)
- Neither tested yet

---

## Next Steps

- [ ] Remote test with current code (IVF2+i16, FAST=16, FULL=24) → expected score 4500-5000 remote
- [ ] If remote FP still high: investigate whether remaining FPs are same 23 intrinsic cases or new ones
- [ ] Worker thread tuning: test 2 vs 4 workers remotely (remote p99=99ms bottleneck, p99_score=1002)
- [ ] K=8192 experiment: smaller clusters may fix the remaining FP=2, FN=4 at cost of 2× centroid scan
- [ ] Document whether dataset 645 or 797 is served on next remote test

---

## Quick Reference

```bash
make bench                           # build + docker up + k6 load test + score
make score                           # show score from last results.json
make submission                      # build image + force-push submission branch
cargo test                           # run all unit tests
uv run tools/build_ivf.py           # rebuild IVF2 index (faiss KMeans, ~1 min)
uv run tools/eval_recall.py         # benchmark IVF recall vs brute-force
uv run tools/train_model.py         # retrain LightGBM + export model_gen.rs
```
