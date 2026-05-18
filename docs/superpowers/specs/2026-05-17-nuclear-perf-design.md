# Nuclear Performance Design — AVX2 + Zero-Alloc + Positional JSON

**Date:** 2026-05-17  
**Goal:** Minimize p99 latency to approach monoio-era results (~21ms) without io_uring. Sacrifice readability where needed. Target score: 4155+.

**Baseline:** tokio multi_thread (worker=2, blocking=2) + spawn_blocking KNN → p99=84.83ms, score=4071.46.

---

## Scope

Four targeted rewrites, each independently testable:

| File | Change | Expected gain |
|------|--------|---------------|
| `src/fraud/knn.rs` | AVX2 centroid scan, zero-alloc buffer, POPCNT label count | -20 to -40ms p99 |
| `src/fraud/json.rs` | Positional parse + slow fallback | -3 to -8ms p99 |
| `src/net/http.rs` | Backward Content-Length scan | -1 to -2ms p99 |

**Already done:** `src/net/response.rs` — `FRAUD_RESPONSES: [&[u8]; 6]` precomputed static bytes already in codebase. `http_body_for` already returns `&'static [u8]`. No work needed.

**Unchanged:** `src/main.rs`, `src/env.rs`, `src/net/mod.rs`, `bin/build_index.rs`, docker-compose, nginx.

---

## Task 1: AVX2 Centroid Scan + Zero-Alloc Top-N Selection

### Problem

`top_n_centroids` is called on every fraud-score request (inside `spawn_blocking`). Current implementation:

```rust
let mut dists = vec![0.0f32; k];          // 16KB heap alloc
let mut indices: Vec<usize> = (0..k).collect();  // 32KB heap alloc
// scalar loop: 14 × 4096 = 57,344 multiply-adds
// select_nth_unstable_by → heap sort internals
```

Total per call: 2 heap allocations (48KB), 57,344 scalar multiply-adds, O(k log k) partial sort.

### Solution

**Thread-local dist buffer** (zero alloc):

```rust
use std::cell::UnsafeCell;

const K: usize = 4096;

thread_local! {
    static DISTS: UnsafeCell<[f32; K]> = const { UnsafeCell::new([0.0f32; K]) };
}
```

Safety: `spawn_blocking` gives each KNN call its own OS thread from the blocking pool. Thread-locals are per-thread. No concurrent access within one thread.

**AVX2 centroid distance loop:**

Centroids stored column-major: `centroids[d * K + ci]` for dim `d`, centroid `ci`. This layout lets us load 8 contiguous centroids for the same dimension in one `_mm256_loadu_ps`.

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn centroid_dists_avx2(q: &[f32; 14], centroids: *const f32, dists: *mut f32) {
    // zero dists
    for i in (0..K).step_by(8) {
        _mm256_storeu_ps(dists.add(i), _mm256_setzero_ps());
    }
    for d in 0..14usize {
        let qd = _mm256_set1_ps(q[d]);
        let base = d * K;
        for ci in (0..K).step_by(8) {
            let v = _mm256_loadu_ps(centroids.add(base + ci));
            let diff = _mm256_sub_ps(qd, v);
            let acc = _mm256_loadu_ps(dists.add(ci));
            let acc = _mm256_fmadd_ps(diff, diff, acc);
            _mm256_storeu_ps(dists.add(ci), acc);
        }
    }
}
```

512 AVX2 iterations × 14 dims = 7,168 FMA ops. ~8x vs 57,344 scalar ops.

**Zero-alloc top-nprobe selection:**

Replace `select_nth_unstable_by` + Vec<usize> with a single linear scan maintaining a fixed `[(f32, u16); FULL_NPROBE]` array (max nprobe=24, stack-allocated):

```rust
fn top_n_centroids_fast(dists: &[f32; K], nprobe: usize) -> [u16; 24] {
    // [dist_bits, centroid_idx] pair — sort by dist ascending
    let mut top = [(u32::MAX, 0u16); 24];
    let mut worst = u32::MAX;
    for (ci, &d) in dists.iter().enumerate() {
        let bits = d.to_bits();
        if bits < worst {
            // linear insert into sorted [24] array
            let pos = top.partition_point(|&(b, _)| b <= bits);
            if pos < nprobe {
                top[pos..nprobe].rotate_right(1);
                top[pos] = (bits, ci as u16);
                worst = top[nprobe - 1].0;
            }
        }
    }
    top.map(|(_, idx)| idx)
}
```

Return type: `[u16; 24]` stack array. No allocation, no sort library.

### POPCNT label count

`knn5_ivf` counts fraud labels in `[u8; 5]` where each label ∈ {0, 1}:

```rust
// current
fast.iter().filter(|&&l| l == 1).count()

// new — pack into u64, one POPCNT instruction
#[inline(always)]
fn count_fraud(labels: [u8; 5]) -> usize {
    let packed = u64::from_le_bytes([
        labels[0], labels[1], labels[2], labels[3], labels[4], 0, 0, 0,
    ]);
    // labels are 0 or 1 — bit 0 of each byte is the value.
    // mask = bit 0 of bytes 0-4: 0x01|0x0100|0x010000|0x01000000|0x0100000000
    (packed & 0x0000_0001_0101_0101).count_ones() as usize
    // compiler emits POPCNT on x86_64
}
```

---

## Task 2: Positional JSON Parsing

### Problem

`json::parse` matches field names via `memchr(b':')` + branch per field. For 15 Payload fields, this is ~15-30 memchr calls + case comparisons.

### Solution

Two-mode parser:

```rust
pub fn parse(buf: &[u8]) -> Option<Payload> {
    parse_positional(buf).or_else(|| parse_full(buf))
}
```

**`parse_positional`**: Assumes fixed field order as sent by competition. Walks forward using only `memchr(b':')` to skip to values and `memchr(b',')` / `memchr(b'}')` to end values. Zero field-name comparison.

Field extraction order (must be verified against actual competition request schema in `src/fraud/json.rs` tests):
1. Skip `{` → find `transaction_id` value (string, skip)
2. → `amount` (f32)
3. → `installments` (u8)
4. → `transaction_date` (string → extract hour byte at offset 11, weekday from date)
5. → `customer.avg_amount`, `tx_count_24h`
6. → `merchant.id` (hash for unknown check), `mcc`, `avg_amount`
7. → `terminal.is_online`, `card_present`, `km_from_home`
8. → `last_transaction` (optional → has_last_tx, minutes_since_last, km_from_current)

**Fallback:** If positional parse returns `None` (unexpected format, field order changed), `parse_full` is the current implementation renamed. No silent wrong result.

**Date/time extraction (byte arithmetic, no chrono):**

```rust
// transaction_date = "2024-01-15T14:30:00Z"
// offset 0123456789012345678
// hour at [11..13], minute at [14..16]
// weekday: compute from date bytes arithmetically (Zeller's formula on raw digits)
fn extract_datetime(s: &[u8]) -> Option<(u8, u8)> {
    if s.len() < 19 { return None; }
    let hour = (s[11] - b'0') * 10 + (s[12] - b'0');
    let weekday = zellers_from_bytes(&s[0..10])?;  // pure arithmetic, no alloc
    Some((hour, weekday))
}
```

---

## Task 3: Backward Content-Length Scan

### Problem

`parse_content_length` scans every header line from the top. NGINX always places `Content-Length` as the last header.

### Solution

```rust
pub fn parse_content_length(headers: &[u8]) -> Option<usize> {
    // fast path: Content-Length is last header line
    let last_nl = memchr::memrchr(b'\n', headers)?;
    let last_line = &headers[last_nl + 1..];
    if last_line.len() > 16
        && last_line[..16].eq_ignore_ascii_case(b"content-length: ")
    {
        return parse_digits(&last_line[16..]);
    }
    // slow path: scan from top (existing logic, renamed parse_content_length_slow)
    parse_content_length_slow(headers)
}

#[inline(always)]
fn parse_digits(b: &[u8]) -> Option<usize> {
    let mut n = 0usize;
    for &c in b {
        if c < b'0' || c > b'9' { break; }
        n = n * 10 + (c - b'0') as usize;
    }
    Some(n).filter(|&x| x > 0)
}
```

`memchr::memrchr` — already available, no new dep.

---

## Verification

Each task must pass before the next:

```bash
cargo test 2>&1
cargo clippy --all-targets --all-features -- -D warnings 2>&1
cargo build --release --bin fraud-detection 2>&1
```

Tests that must keep passing:
- `find_header_end`, `parse_content_length`, `detect_route` (http.rs)
- `parse_legit_no_last_tx`, `parse_tx_with_last_transaction`, `parse_unknown_merchant`, `parse_returns_none_on_garbage` (json.rs)
- `smoke_warmup_and_query`, `smoke_fraud_heavy_query` (knn.rs)

New tests required:
- `parse_content_length` still works when Content-Length is NOT last header (slow path triggered)
- `parse_positional` returns same result as `parse_full` for all existing test fixtures
- `resp_for_count(0..=5)` produces valid HTTP responses with correct Content-Length
- `count_fraud([1,0,1,0,1])` == 3, `count_fraud([0,0,0,0,0])` == 0

---

## Commit Plan

```
feat: precompute static HTTP responses for all fraud counts
perf: backward Content-Length scan in HTTP parser
perf: positional JSON parser with full fallback
perf: AVX2 centroid scan + zero-alloc buffer + POPCNT label count
```

Atomic per task. Task 1 last (highest risk, most lines).
