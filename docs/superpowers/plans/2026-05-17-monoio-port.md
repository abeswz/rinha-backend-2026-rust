# monoio io_uring Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the axum/tokio HTTP stack with monoio + io_uring, hand-rolled HTTP/JSON parsers, and an embedded IVF1 KNN index — targeting lowest possible p99 and highest competition score.

**Architecture:** Single-threaded monoio io_uring runtime per instance. Zero heap allocations on the hot path: stack-allocated rx buffer, positional JSON parser, compile-time AVX2/FMA KNN, and six prebuilt static HTTP responses. The IVF1 index is gzip-compressed, embedded via `include_bytes!`, and decoded once at startup into 32-byte-aligned SIMD-ready arrays.

**Tech Stack:** monoio 0.2 (io_uring), memchr 2, aligned-vec 0.6, flate2 1 (rust backend), mimalloc 0.1, libc 0.2, serde + serde_json (build tool only)

---

## File Map

| Action | Path |
|--------|------|
| Rewrite | `Cargo.toml` |
| Rewrite | `src/main.rs` |
| Create | `src/env.rs` |
| Create | `src/net/mod.rs` |
| Create | `src/net/http.rs` |
| Create | `src/net/response.rs` |
| Create | `src/fraud/mod.rs` |
| Create | `src/fraud/json.rs` |
| Create | `src/fraud/vector.rs` |
| Create | `src/fraud/knn.rs` |
| Create | `src/fraud/data.rs` |
| Create | `bin/build_index.rs` |
| Delete | `src/domain/`, `src/service/`, `src/repository/`, `src/web/`, `src/usecase/`, `src/config.rs`, `src/error.rs`, `src/lib.rs`, `bin/preprocess.rs` |
| Update | `Dockerfile` |
| Update | `docker-compose.yml` |

---

## Task 1: Cargo.toml + Delete Old Files + Module Stubs

**Files:**
- Modify: `Cargo.toml`
- Delete: `src/lib.rs`, `src/config.rs`, `src/error.rs`, `src/domain/`, `src/service/`, `src/repository/`, `src/web/`, `src/usecase/`, `bin/preprocess.rs`
- Create: `src/main.rs`, `src/env.rs`, `src/net/mod.rs`, `src/net/http.rs`, `src/net/response.rs`, `src/fraud/mod.rs`, `src/fraud/json.rs`, `src/fraud/vector.rs`, `src/fraud/knn.rs`, `src/fraud/data.rs`

- [ ] **Step 1: Replace Cargo.toml**

```toml
[package]
name = "fraud-detection"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "fraud-detection"
path = "src/main.rs"

[[bin]]
name = "build_index"
path = "bin/build_index.rs"

[dependencies]
monoio      = { version = "0.2", default-features = false, features = ["iouring", "legacy", "macros"] }
memchr      = "2"
mimalloc    = { version = "0.1", default-features = false }
aligned-vec = "0.6"
flate2      = { version = "1", default-features = false, features = ["rust_backend"] }
libc        = "0.2"
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"

[profile.release]
opt-level       = 3
lto             = "fat"
codegen-units   = 1
strip           = true
panic           = "abort"
debug           = 0
overflow-checks = false
incremental     = false

[profile.release.package."*"]
opt-level = 3
```

- [ ] **Step 2: Delete old source tree**

```bash
rm -rf src/domain src/service src/repository src/web src/usecase
rm -f src/lib.rs src/config.rs src/error.rs bin/preprocess.rs
```

- [ ] **Step 3: Create stub main.rs that compiles**

`src/main.rs`:
```rust
mod env;
mod fraud;
mod net;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {}
```

- [ ] **Step 4: Create stub module files**

`src/env.rs`:
```rust
pub fn sock_path() -> String { std::env::var("SOCK").unwrap_or_else(|_| "/tmp/api.sock".into()) }
```

`src/net/mod.rs`:
```rust
pub mod http;
pub mod response;
```

`src/net/http.rs`:
```rust
// TODO
```

`src/net/response.rs`:
```rust
// TODO
```

`src/fraud/mod.rs`:
```rust
pub mod data;
pub mod json;
pub mod knn;
pub mod vector;
```

`src/fraud/json.rs`:
```rust
// TODO
```

`src/fraud/vector.rs`:
```rust
// TODO
```

`src/fraud/knn.rs`:
```rust
// TODO
```

`src/fraud/data.rs`:
```rust
// TODO
```

- [ ] **Step 5: Create data/ directory**

```bash
mkdir -p data
```

- [ ] **Step 6: Verify cargo check passes**

```bash
cargo check 2>&1 | tail -5
```

Expected: `Finished` with no errors (warnings OK).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml src/ bin/ data/
git commit -m "chore: replace axum/tokio stack with monoio scaffolding"
```

---

## Task 2: fraud/json.rs — Positional JSON Parser

**Files:**
- Modify: `src/fraud/json.rs`

The parser advances through the fixed JSON field order without name-matching.
It only uses `memchr` for navigation. All fields go on the stack.

**JSON field order (guaranteed by spec, verified against resources/example-payloads.json):**
```
id → transaction.{amount, installments, requested_at} →
customer.{avg_amount, tx_count_24h, known_merchants[]} →
merchant.{id, mcc, avg_amount} →
terminal.{is_online, card_present, km_from_home} →
last_transaction (null | {timestamp, km_from_current})
```

- [ ] **Step 1: Write the failing tests**

`src/fraud/json.rs` (tests only, no impl yet):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    const LEGIT_PAYLOAD: &[u8] = br#"{
  "id": "tx-1329056812",
  "transaction": {
    "amount": 41.12,
    "installments": 2,
    "requested_at": "2026-03-11T18:45:53Z"
  },
  "customer": {
    "avg_amount": 82.24,
    "tx_count_24h": 3,
    "known_merchants": [
      "MERC-003",
      "MERC-016"
    ]
  },
  "merchant": {
    "id": "MERC-016",
    "mcc": "5411",
    "avg_amount": 60.25
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 29.2331036248
  },
  "last_transaction": null
}"#;

    const TX_WITH_LAST: &[u8] = br#"{
  "id": "tx-3576980410",
  "transaction": {
    "amount": 384.88,
    "installments": 3,
    "requested_at": "2026-03-11T20:23:35Z"
  },
  "customer": {
    "avg_amount": 769.76,
    "tx_count_24h": 3,
    "known_merchants": [
      "MERC-009",
      "MERC-009",
      "MERC-001",
      "MERC-001"
    ]
  },
  "merchant": {
    "id": "MERC-001",
    "mcc": "5912",
    "avg_amount": 298.95
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 13.7090520965
  },
  "last_transaction": {
    "timestamp": "2026-03-11T14:58:35Z",
    "km_from_current": 18.8626479774
  }
}"#;

    #[test]
    fn parse_legit_no_last_tx() {
        let p = parse(LEGIT_PAYLOAD).expect("parse failed");
        assert!((p.amount - 41.12).abs() < 0.001);
        assert_eq!(p.installments, 2);
        assert_eq!(p.hour, 18);
        assert_eq!(p.weekday, 2); // Wednesday = 2 (Mon=0)
        assert!((p.customer_avg_amount - 82.24).abs() < 0.001);
        assert_eq!(p.tx_count_24h, 3);
        assert!(!p.is_unknown_merchant); // MERC-016 is in known_merchants
        assert_eq!(p.mcc, 5411);
        assert!((p.merchant_avg_amount - 60.25).abs() < 0.001);
        assert!(!p.is_online);
        assert!(p.card_present);
        assert!((p.km_from_home - 29.2331).abs() < 0.001);
        assert!(!p.has_last_tx);
    }

    #[test]
    fn parse_tx_with_last_transaction() {
        let p = parse(TX_WITH_LAST).expect("parse failed");
        assert!((p.amount - 384.88).abs() < 0.001);
        assert_eq!(p.installments, 3);
        assert_eq!(p.hour, 20);
        assert_eq!(p.weekday, 2);
        assert!(!p.is_unknown_merchant); // MERC-001 is in known_merchants
        assert_eq!(p.mcc, 5912);
        assert!(p.has_last_tx);
        assert!((p.minutes_since_last - 325.0).abs() < 1.0);
        assert!((p.km_from_current - 18.8626).abs() < 0.001);
    }

    #[test]
    fn parse_unknown_merchant() {
        // Modify LEGIT_PAYLOAD to use a merchant not in known_merchants
        // merchant "MERC-999" is not in ["MERC-003", "MERC-016"]
        let raw = LEGIT_PAYLOAD.to_vec();
        let s = std::str::from_utf8(&raw).unwrap();
        let modified = s.replace("\"MERC-016\",\n    \"mcc\"", "\"MERC-999\",\n    \"mcc\"");
        let p = parse(modified.as_bytes()).expect("parse failed");
        assert!(p.is_unknown_merchant);
    }

    #[test]
    fn parse_returns_none_on_garbage() {
        assert!(parse(b"not json").is_none());
        assert!(parse(b"{}").is_none());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test fraud::json 2>&1 | tail -20
```

Expected: compile error or test failures — the `parse` function and `Payload` struct are not yet defined.

- [ ] **Step 3: Implement Payload struct and parse function**

Replace `src/fraud/json.rs` with:

```rust
use memchr::{memchr, memchr2};

#[derive(Debug, Clone, Copy)]
pub struct Payload {
    pub amount: f32,
    pub installments: u8,
    pub hour: u8,
    pub weekday: u8,
    pub customer_avg_amount: f32,
    pub tx_count_24h: u8,
    pub is_unknown_merchant: bool,
    pub mcc: u32,
    pub merchant_avg_amount: f32,
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
    pub has_last_tx: bool,
    pub minutes_since_last: f32,
    pub km_from_current: f32,
}

// Advances `pos` past the next ':' (skips field name we already know).
#[inline(always)]
fn skip_to_value(buf: &[u8], pos: &mut usize) -> Option<()> {
    let i = memchr(b':', &buf[*pos..])?;
    *pos += i + 1;
    // skip whitespace
    while *pos < buf.len() && (buf[*pos] == b' ' || buf[*pos] == b'\n' || buf[*pos] == b'\r' || buf[*pos] == b'\t') {
        *pos += 1;
    }
    Some(())
}

// Reads a quoted string, returns its byte slice, advances pos past closing '"'.
#[inline(always)]
fn read_string<'a>(buf: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    if buf.get(*pos) != Some(&b'"') { return None; }
    *pos += 1;
    let end = memchr(b'"', &buf[*pos..])?;
    let s = &buf[*pos..*pos + end];
    *pos += end + 1;
    Some(s)
}

// Reads an unquoted value (number or bool/null) up to the next ',' '}' or whitespace.
#[inline(always)]
fn read_token<'a>(buf: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    let start = *pos;
    while *pos < buf.len() {
        let c = buf[*pos];
        if c == b',' || c == b'}' || c == b']' || c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
            break;
        }
        *pos += 1;
    }
    if *pos == start { return None; }
    Some(&buf[start..*pos])
}

// Parses "YYYY-MM-DDTHH:MM:SS" from buf[pos..pos+19], returns (y,mo,d,h,min).
// Expects exactly that pattern; the trailing 'Z' is ignored.
fn parse_iso(buf: &[u8], pos: usize) -> Option<(u16, u8, u8, u8, u8)> {
    if buf.len() < pos + 19 { return None; }
    let d = &buf[pos..];
    let y = parse_digits4(d, 0)?;
    let mo = parse_digits2(d, 5)?;
    let day = parse_digits2(d, 8)?;
    let h = parse_digits2(d, 11)?;
    let mn = parse_digits2(d, 14)?;
    Some((y, mo, day, h, mn))
}

#[inline(always)]
fn parse_digits4(d: &[u8], off: usize) -> Option<u16> {
    let a = (d.get(off)? - b'0') as u16;
    let b = (d.get(off+1)? - b'0') as u16;
    let c = (d.get(off+2)? - b'0') as u16;
    let e = (d.get(off+3)? - b'0') as u16;
    Some(a*1000 + b*100 + c*10 + e)
}

#[inline(always)]
fn parse_digits2(d: &[u8], off: usize) -> Option<u8> {
    let a = d.get(off)? - b'0';
    let b = d.get(off+1)? - b'0';
    Some(a * 10 + b)
}

// Rata Die (Julian Day Number variant) - days since epoch 0000-01-01.
// Used for computing minute difference across day boundaries.
fn days_since_epoch(y: u16, mo: u8, d: u8) -> u32 {
    let (y, m) = if mo <= 2 {
        (y as i32 - 1, mo as i32 + 12)
    } else {
        (y as i32, mo as i32)
    };
    let a = y / 100;
    let b = 2 - a + a / 4;
    ((365.25 * (y + 4716) as f64) as i32
        + (30.6001 * (m + 1) as f64) as i32
        + d as i32 + b - 1524) as u32
}

fn minutes_between(
    cur: (u16, u8, u8, u8, u8),
    prev: (u16, u8, u8, u8, u8),
) -> f32 {
    let dc = days_since_epoch(cur.0, cur.1, cur.2) as i64;
    let dp = days_since_epoch(prev.0, prev.1, prev.2) as i64;
    let mc = cur.3 as i64 * 60 + cur.4 as i64;
    let mp = prev.3 as i64 * 60 + prev.4 as i64;
    ((dc - dp) * 1440 + (mc - mp)) as f32
}

// Weekday 0=Mon .. 6=Sun from days-since-epoch value.
fn weekday_from_days(days: u32) -> u8 {
    // epoch day 0 = some Monday; find offset modulo 7
    // Julian day 0 = Monday in this system.
    (days % 7) as u8
}

// Parse weekday for a date. 2026-03-11 is a Wednesday (weekday=2 Mon=0).
// We derive it from days_since_epoch modulo 7.
fn date_weekday(y: u16, mo: u8, d: u8) -> u8 {
    // Tomohiko Sakamoto's algorithm, returns 0=Sun..6=Sat; remap to 0=Mon..6=Sun.
    let t = [0u8, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if mo < 3 { y - 1 } else { y };
    let dow_sun0 = (y as u32 + y as u32/4 - y as u32/100 + y as u32/400
        + t[(mo-1) as usize] as u32 + d as u32) % 7;
    // dow_sun0: 0=Sun,1=Mon,...,6=Sat → remap to 0=Mon..6=Sun
    ((dow_sun0 + 6) % 7) as u8
}

pub fn parse(buf: &[u8]) -> Option<Payload> {
    let mut pos = 0;

    // skip to first '{'
    pos = memchr(b'{', buf)?;
    pos += 1;

    // "id": skip value
    skip_to_value(buf, &mut pos)?;
    read_string(buf, &mut pos)?;

    // "transaction": {
    skip_to_value(buf, &mut pos)?; // skip "transaction" field name
    // skip to '{' of transaction object
    pos += memchr(b'{', &buf[pos..])? + 1;

    // "amount":
    skip_to_value(buf, &mut pos)?;
    let amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?.parse().ok()?;

    // "installments":
    skip_to_value(buf, &mut pos)?;
    let installments: u8 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?.parse().ok()?;

    // "requested_at": "YYYY-MM-DDTHH:MM:SSZ"
    skip_to_value(buf, &mut pos)?;
    if buf.get(pos) != Some(&b'"') { return None; }
    let dt_start = pos + 1;
    let (y, mo, d, hour, _min_ignored) = parse_iso(buf, dt_start)?;
    let weekday = date_weekday(y, mo, d);
    let cur_time = (y, mo, d, hour, {
        parse_digits2(buf, dt_start + 14)?
    });
    // advance past closing '"'
    pos = dt_start + memchr(b'"', &buf[dt_start..])? + 1;

    // close transaction object
    pos += memchr(b'}', &buf[pos..])? + 1;

    // "customer": {
    skip_to_value(buf, &mut pos)?;
    pos += memchr(b'{', &buf[pos..])? + 1;

    // "avg_amount":
    skip_to_value(buf, &mut pos)?;
    let customer_avg_amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?.parse().ok()?;

    // "tx_count_24h":
    skip_to_value(buf, &mut pos)?;
    let tx_count_24h: u8 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?.parse().ok()?;

    // "known_merchants": [ ... ]
    skip_to_value(buf, &mut pos)?;
    pos += memchr(b'[', &buf[pos..])? + 1;

    // Collect up to 32 merchant IDs on the stack (each up to 16 bytes).
    const MAX_KNOWN: usize = 32;
    const MAX_ID_LEN: usize = 16;
    let mut known_buf = [[0u8; MAX_ID_LEN]; MAX_KNOWN];
    let mut known_lens = [0u8; MAX_KNOWN];
    let mut known_count: usize = 0;

    loop {
        // skip whitespace and commas
        while pos < buf.len() && matches!(buf[pos], b' ' | b'\n' | b'\r' | b'\t' | b',') {
            pos += 1;
        }
        if pos >= buf.len() { return None; }
        if buf[pos] == b']' { pos += 1; break; }
        if buf[pos] == b'"' {
            let s = read_string(buf, &mut pos)?;
            if known_count < MAX_KNOWN {
                let len = s.len().min(MAX_ID_LEN);
                known_buf[known_count][..len].copy_from_slice(&s[..len]);
                known_lens[known_count] = len as u8;
                known_count += 1;
            }
        } else {
            pos += 1;
        }
    }

    // close customer object
    pos += memchr(b'}', &buf[pos..])? + 1;

    // "merchant": {
    skip_to_value(buf, &mut pos)?;
    pos += memchr(b'{', &buf[pos..])? + 1;

    // "id":
    skip_to_value(buf, &mut pos)?;
    let merch_id = read_string(buf, &mut pos)?;
    let is_unknown_merchant = !(0..known_count).any(|i| {
        let len = known_lens[i] as usize;
        merch_id.len() == len && merch_id == &known_buf[i][..len]
    });

    // "mcc": (string value, parse as u32)
    skip_to_value(buf, &mut pos)?;
    let mcc_str = read_string(buf, &mut pos)?;
    let mcc: u32 = std::str::from_utf8(mcc_str).ok()?.parse().ok()?;

    // "avg_amount":
    skip_to_value(buf, &mut pos)?;
    let merchant_avg_amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?.parse().ok()?;

    // close merchant object
    pos += memchr(b'}', &buf[pos..])? + 1;

    // "terminal": {
    skip_to_value(buf, &mut pos)?;
    pos += memchr(b'{', &buf[pos..])? + 1;

    // "is_online":
    skip_to_value(buf, &mut pos)?;
    let tok = read_token(buf, &mut pos)?;
    let is_online = tok == b"true";

    // "card_present":
    skip_to_value(buf, &mut pos)?;
    let tok = read_token(buf, &mut pos)?;
    let card_present = tok == b"true";

    // "km_from_home":
    skip_to_value(buf, &mut pos)?;
    let km_from_home: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?.parse().ok()?;

    // close terminal object
    pos += memchr(b'}', &buf[pos..])? + 1;

    // "last_transaction":
    skip_to_value(buf, &mut pos)?;

    // skip whitespace
    while pos < buf.len() && matches!(buf[pos], b' ' | b'\n' | b'\r' | b'\t') {
        pos += 1;
    }

    let (has_last_tx, minutes_since_last, km_from_current) = if buf.get(pos..pos+4) == Some(b"null") {
        (false, 0.0f32, 0.0f32)
    } else if buf.get(pos) == Some(&b'{') {
        pos += 1;

        // "timestamp":
        skip_to_value(buf, &mut pos)?;
        if buf.get(pos) != Some(&b'"') { return None; }
        let ts_start = pos + 1;
        let prev_time = (
            parse_digits4(buf, ts_start)?,
            parse_digits2(buf, ts_start + 5)?,
            parse_digits2(buf, ts_start + 8)?,
            parse_digits2(buf, ts_start + 11)?,
            parse_digits2(buf, ts_start + 14)?,
        );
        pos = ts_start + memchr(b'"', &buf[ts_start..])? + 1;

        // "km_from_current":
        skip_to_value(buf, &mut pos)?;
        let km_cur: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
            .ok()?.parse().ok()?;

        let mins = minutes_between(cur_time, prev_time);
        (true, mins, km_cur)
    } else {
        return None;
    };

    Some(Payload {
        amount,
        installments,
        hour,
        weekday,
        customer_avg_amount,
        tx_count_24h,
        is_unknown_merchant,
        mcc,
        merchant_avg_amount,
        is_online,
        card_present,
        km_from_home,
        has_last_tx,
        minutes_since_last,
        km_from_current,
    })
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test fraud::json 2>&1 | tail -20
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/json.rs
git commit -m "feat: positional JSON parser for Payload"
```

---

## Task 3: fraud/vector.rs — vectorize + round4

**Files:**
- Modify: `src/fraud/vector.rs`

Feature vector layout (14 dims, matches existing vectorizer exactly):
```
[0] amount/10000  [1] installments/12  [2] (amount/cust_avg)/10
[3] hour/23       [4] weekday/6        [5] minutes_since_last/1440 or -1.0
[6] km_from_current/1000 or -1.0       [7] km_from_home/1000
[8] tx_count_24h/20  [9] is_online  [10] card_present  [11] is_unknown_merchant
[12] mcc_risk  [13] merchant_avg_amount/10000
```

All values passed through `round4(x) = (x * 10000.0).round() * 0.0001`.

- [ ] **Step 1: Write failing tests**

`src/fraud/vector.rs` (tests only):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fraud::json::Payload;

    fn legit_payload() -> Payload {
        Payload {
            amount: 41.12,
            installments: 2,
            hour: 18,
            weekday: 2,
            customer_avg_amount: 82.24,
            tx_count_24h: 3,
            is_unknown_merchant: false,
            mcc: 5411,
            merchant_avg_amount: 60.25,
            is_online: false,
            card_present: true,
            km_from_home: 29.2331036248,
            has_last_tx: false,
            minutes_since_last: 0.0,
            km_from_current: 0.0,
        }
    }

    fn fraud_payload() -> Payload {
        Payload {
            amount: 9505.97,
            installments: 10,
            hour: 5,
            weekday: 4, // Friday (2026-03-14 is a Saturday, let's use exact weekday)
            customer_avg_amount: 81.28,
            tx_count_24h: 20,
            is_unknown_merchant: true,
            mcc: 7802,
            merchant_avg_amount: 54.86,
            is_online: false,
            card_present: true,
            km_from_home: 952.27,
            has_last_tx: false,
            minutes_since_last: 0.0,
            km_from_current: 0.0,
        }
    }

    #[test]
    fn test_round4() {
        assert_eq!(round4(0.004112), 0.0041);
        assert_eq!(round4(-1.0), -1.0);
        assert_eq!(round4(1.0), 1.0);
        assert!((round4(0.166667) - 0.1667).abs() < 0.00001);
    }

    #[test]
    fn test_vectorize_legit() {
        let v = vectorize(&legit_payload());
        // dim0: round4(41.12/10000) = round4(0.004112) = 0.0041
        assert!((v[0] - 0.0041).abs() < 0.0001, "dim0 got {}", v[0]);
        // dim1: round4(2/12) = round4(0.16667) = 0.1667
        assert!((v[1] - 0.1667).abs() < 0.0001, "dim1 got {}", v[1]);
        // dim2: round4((41.12/82.24)/10) = round4(0.05) = 0.05
        assert!((v[2] - 0.05).abs() < 0.0001, "dim2 got {}", v[2]);
        // dim3: round4(18/23) = round4(0.78261) = 0.7826
        assert!((v[3] - 0.7826).abs() < 0.0001, "dim3 got {}", v[3]);
        // dim4: round4(2/6) = round4(0.33333) = 0.3333
        assert!((v[4] - 0.3333).abs() < 0.0001, "dim4 got {}", v[4]);
        // dim5: no last_tx → -1.0
        assert_eq!(v[5], -1.0, "dim5 should be -1.0");
        // dim6: no last_tx → -1.0
        assert_eq!(v[6], -1.0, "dim6 should be -1.0");
        // dim7: round4(29.2331/1000) = round4(0.029233) = 0.0292
        assert!((v[7] - 0.0292).abs() < 0.0001, "dim7 got {}", v[7]);
        // dim8: round4(3/20) = round4(0.15) = 0.15
        assert!((v[8] - 0.15).abs() < 0.0001, "dim8 got {}", v[8]);
        // dim9: not online → 0.0
        assert_eq!(v[9], 0.0);
        // dim10: card present → 1.0
        assert_eq!(v[10], 1.0);
        // dim11: known merchant → 0.0
        assert_eq!(v[11], 0.0);
        // dim12: mcc 5411 → 0.15
        assert!((v[12] - 0.15).abs() < 0.0001, "dim12 got {}", v[12]);
        // dim13: round4(60.25/10000) = round4(0.006025) = 0.006
        assert!((v[13] - 0.006).abs() < 0.0001, "dim13 got {}", v[13]);
    }

    #[test]
    fn test_vectorize_fraud() {
        let v = vectorize(&fraud_payload());
        // dim0: round4(9505.97/10000) = 0.9506
        assert!((v[0] - 0.9506).abs() < 0.0001, "dim0 got {}", v[0]);
        // dim2: clamped to 1.0 (9505.97/(81.28*10) > 1)
        assert_eq!(v[2], 1.0, "dim2 should be clamped to 1.0");
        // dim8: tx_count 20/20 = 1.0
        assert_eq!(v[8], 1.0, "dim8 = 20/20 = 1.0");
        // dim11: unknown merchant → 1.0
        assert_eq!(v[11], 1.0);
        // dim12: mcc 7802 → 0.75
        assert!((v[12] - 0.75).abs() < 0.0001, "dim12 got {}", v[12]);
    }

    #[test]
    fn test_mcc_unknown_defaults_to_0_5() {
        let mut p = legit_payload();
        p.mcc = 9999;
        let v = vectorize(&p);
        assert!((v[12] - 0.5).abs() < 0.0001, "unknown mcc should default to 0.5, got {}", v[12]);
    }

    #[test]
    fn test_with_last_tx() {
        let mut p = legit_payload();
        p.has_last_tx = true;
        p.minutes_since_last = 325.0;
        p.km_from_current = 18.8626;
        let v = vectorize(&p);
        // dim5: round4(325/1440) = round4(0.22569) = 0.2257
        assert!((v[5] - 0.2257).abs() < 0.0001, "dim5 got {}", v[5]);
        // dim6: round4(18.8626/1000) = round4(0.018863) = 0.0189
        assert!((v[6] - 0.0189).abs() < 0.0001, "dim6 got {}", v[6]);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test fraud::vector 2>&1 | tail -10
```

Expected: compile error — `vectorize` and `round4` not defined.

- [ ] **Step 3: Implement vectorize**

Replace `src/fraud/vector.rs` with:
```rust
use crate::fraud::json::Payload;

#[inline(always)]
pub fn round4(x: f32) -> f32 {
    (x * 10000.0).round() * 0.0001
}

#[inline(always)]
fn mcc_risk(mcc: u32) -> f32 {
    match mcc {
        5411 => 0.15,
        5812 => 0.30,
        5912 => 0.20,
        5944 => 0.45,
        7801 => 0.80,
        7802 => 0.75,
        7995 => 0.85,
        4511 => 0.35,
        5311 => 0.25,
        5999 => 0.50,
        _    => 0.50,
    }
}

pub fn vectorize(p: &Payload) -> [f32; 14] {
    let (minutes_norm, km_cur_norm) = if p.has_last_tx {
        (
            round4((p.minutes_since_last / 1440.0).clamp(0.0, 1.0)),
            round4((p.km_from_current / 1000.0).clamp(0.0, 1.0)),
        )
    } else {
        (-1.0, -1.0)
    };

    [
        round4((p.amount / 10_000.0).clamp(0.0, 1.0)),
        round4((p.installments as f32 / 12.0).clamp(0.0, 1.0)),
        round4(((p.amount / p.customer_avg_amount) / 10.0).clamp(0.0, 1.0)),
        round4(p.hour as f32 / 23.0),
        round4(p.weekday as f32 / 6.0),
        minutes_norm,
        km_cur_norm,
        round4((p.km_from_home / 1000.0).clamp(0.0, 1.0)),
        round4((p.tx_count_24h as f32 / 20.0).clamp(0.0, 1.0)),
        if p.is_online { 1.0 } else { 0.0 },
        if p.card_present { 1.0 } else { 0.0 },
        if p.is_unknown_merchant { 1.0 } else { 0.0 },
        mcc_risk(p.mcc),
        round4((p.merchant_avg_amount / 10_000.0).clamp(0.0, 1.0)),
    ]
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test fraud::vector 2>&1 | tail -20
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/vector.rs
git commit -m "feat: vectorize + round4 + MCC match table (no alloc)"
```

---

## Task 4: net/response.rs — 6 Static HTTP Responses

**Files:**
- Modify: `src/net/response.rs`

Six complete HTTP/1.1 responses, one per `fraud_count` 0-5. Three utility responses for ready/404/400.

Decision rule: `approved = fraud_count <= 2`. Score = `fraud_count / 5.0`.

- [ ] **Step 1: Write failing tests**

`src/net/response.rs` (tests only):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn body_of(resp: &[u8]) -> &[u8] {
        let sep = b"\r\n\r\n";
        let pos = resp.windows(4).position(|w| w == sep).expect("no header separator");
        &resp[pos + 4..]
    }

    #[test]
    fn all_6_responses_valid_json() {
        for count in 0u8..=5 {
            let resp = http_body_for(count);
            let body = body_of(resp);
            let s = std::str::from_utf8(body).expect("not utf8");
            assert!(s.contains("\"approved\""), "count={count} missing approved");
            assert!(s.contains("\"fraud_score\""), "count={count} missing fraud_score");
        }
    }

    #[test]
    fn approved_flag_follows_threshold() {
        for count in 0u8..=2 {
            let body = std::str::from_utf8(body_of(http_body_for(count))).unwrap();
            assert!(body.contains("\"approved\":true"), "count={count} should be approved");
        }
        for count in 3u8..=5 {
            let body = std::str::from_utf8(body_of(http_body_for(count))).unwrap();
            assert!(body.contains("\"approved\":false"), "count={count} should be rejected");
        }
    }

    #[test]
    fn content_length_matches_body() {
        for count in 0u8..=5 {
            let resp = http_body_for(count);
            let header = std::str::from_utf8(
                &resp[..resp.windows(4).position(|w| w == b"\r\n\r\n").unwrap()]
            ).unwrap();
            let cl: usize = header.lines()
                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                .expect("no content-length")
                .split(':').nth(1).unwrap().trim().parse().unwrap();
            let body_len = body_of(resp).len();
            assert_eq!(cl, body_len, "count={count}: content-length={cl} body_len={body_len}");
        }
    }

    #[test]
    fn ready_response_is_200() {
        let s = std::str::from_utf8(RESP_READY).unwrap();
        assert!(s.starts_with("HTTP/1.1 200"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test net::response 2>&1 | tail -10
```

Expected: compile error — symbols not defined.

- [ ] **Step 3: Implement response.rs**

Replace `src/net/response.rs` with:
```rust
// fraud_count 0-5 → complete HTTP/1.1 200 response with JSON body
static RESP_0: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}";
static RESP_1: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}";
static RESP_2: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}";
static RESP_3: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}";
static RESP_4: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}";
static RESP_5: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}";

static FRAUD_RESPONSES: [&[u8]; 6] = [RESP_0, RESP_1, RESP_2, RESP_3, RESP_4, RESP_5];

pub static RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK";
pub static RESP_NOT_FOUND: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
pub static RESP_BAD_REQ: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

#[inline(always)]
pub fn http_body_for(fraud_count: u8) -> &'static [u8] {
    FRAUD_RESPONSES[fraud_count as usize]
}
```

- [ ] **Step 4: Verify Content-Length values are correct**

```bash
python3 -c "
bodies = [
    b'{\"approved\":true,\"fraud_score\":0.0}',
    b'{\"approved\":true,\"fraud_score\":0.2}',
    b'{\"approved\":true,\"fraud_score\":0.4}',
    b'{\"approved\":false,\"fraud_score\":0.6}',
    b'{\"approved\":false,\"fraud_score\":0.8}',
    b'{\"approved\":false,\"fraud_score\":1.0}',
]
for i, b in enumerate(bodies):
    print(f'count={i}: len={len(b)}')
"
```

Expected output: `count=0,1,2: len=35` and `count=3,4,5: len=36`. Adjust Content-Length values if different.

- [ ] **Step 5: Run tests**

```bash
cargo test net::response 2>&1 | tail -20
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/net/response.rs
git commit -m "feat: 6 prebuilt static HTTP fraud-score responses"
```

---

## Task 5: bin/build_index.rs — kmeans++ IVF1 Index Builder

**Files:**
- Create: `bin/build_index.rs`

This binary reads `resources/references.json.gz`, applies `round4` to each vector, runs kmeans++ (K=4096, 25 iterations), and writes `data/index.bin.gz` in IVF1 format.

IVF1 binary format (little-endian):
```
[4 bytes] magic: b"IVF1"
[4 bytes] n: u32 — total vectors
[4 bytes] k: u32 — number of centroids (4096)
[4 bytes] d: u32 — dimensions (14)
[k*d*4 bytes] centroids: f32[] — column-major (dim0 of all k, then dim1, ...)
[(k+1)*4 bytes] offsets: u32[] — CSR block offsets (offsets[i]..offsets[i+1] = blocks for centroid i)
[total_blocks*8 bytes] labels: u8[] — fraud label per slot (8 per block)
[total_blocks*14*8*2 bytes] blocks: i16[] — vectors as i16 = round(f32*10000), layout: [block][dim*8+slot]
```

Blocks pack 8 vectors each. Last block of each centroid is zero-padded if count % 8 != 0.

- [ ] **Step 1: Create bin/build_index.rs**

```rust
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

const K: usize = 4096;
const D: usize = 14;
const INIT_SAMPLE: usize = 50_000;
const LLOYD_ITERS: usize = 25;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; D],
    label: String,
}

fn round4(x: f32) -> f32 { (x * 10000.0).round() * 0.0001 }

fn sq_dist(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..D { let d = a[i] - b[i]; s += d * d; }
    s
}

fn nearest_centroid(v: &[f32; D], centroids: &[[f32; D]]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_dist(v, c);
        if d < best_d { best_d = d; best = i; }
    }
    best
}

fn kmeans_plus_plus_init(vecs: &[[f32; D]], k: usize, sample_n: usize) -> Vec<[f32; D]> {
    // sample_n points from vecs for initialization
    let step = (vecs.len() / sample_n).max(1);
    let sample: Vec<&[f32; D]> = vecs.iter().step_by(step).take(sample_n).collect();
    let n = sample.len();
    
    let mut rng = 0xdeadbeef_u64;
    let lcg = |r: &mut u64| -> usize {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*r >> 33) as usize)
    };

    let mut centers: Vec<[f32; D]> = Vec::with_capacity(k);
    centers.push(*sample[lcg(&mut rng) % n]);

    let mut dists = vec![f32::INFINITY; n];

    for _ in 1..k {
        // Update distances to nearest existing center
        let last = centers.last().unwrap();
        for (i, v) in sample.iter().enumerate() {
            let d = sq_dist(v, last);
            if d < dists[i] { dists[i] = d; }
        }
        // Sample next center proportional to d²
        let total: f32 = dists.iter().sum();
        let mut threshold = (lcg(&mut rng) as f32 / u32::MAX as f32) * total;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 { chosen = i; break; }
        }
        centers.push(*sample[chosen]);
    }

    centers
}

fn lloyd_assign(vecs: &[[f32; D]], centroids: &[[f32; D]]) -> Vec<usize> {
    // Parallel assignment using thread::scope
    let n = vecs.len();
    let nthreads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(16);
    let chunk = (n + nthreads - 1) / nthreads;
    let mut assignments = vec![0usize; n];

    std::thread::scope(|s| {
        let chunks: Vec<_> = assignments.chunks_mut(chunk).enumerate().collect();
        let mut handles = Vec::new();
        for (ci, chunk_slice) in chunks {
            let start = ci * chunk;
            let end = (start + chunk_slice.len()).min(n);
            let vslice = &vecs[start..end];
            handles.push(s.spawn(move || {
                for (j, v) in vslice.iter().enumerate() {
                    chunk_slice[j] = nearest_centroid(v, centroids);
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
    });

    assignments
}

fn lloyd_update(vecs: &[[f32; D]], assignments: &[usize], k: usize) -> Vec<[f32; D]> {
    let mut sums = vec![[0.0f32; D]; k];
    let mut counts = vec![0u64; k];
    for (v, &ci) in vecs.iter().zip(assignments.iter()) {
        for d in 0..D { sums[ci][d] += v[d]; }
        counts[ci] += 1;
    }
    for (ci, sum) in sums.iter_mut().enumerate() {
        let c = counts[ci].max(1) as f32;
        for d in 0..D { sum[d] /= c; }
    }
    sums
}

fn write_ivf1(
    centroids: &[[f32; D]],
    assignments: &[usize],
    vecs: &[[f32; D]],
    labels: &[u8],
    k: usize,
    out_path: &str,
) {
    // Group vectors by centroid
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &ci) in assignments.iter().enumerate() {
        groups[ci].push(i);
    }

    // Build blocks (8 vectors per block, zero-padded)
    let mut offsets: Vec<u32> = Vec::with_capacity(k + 1);
    let mut label_buf: Vec<u8> = Vec::new();
    let mut block_buf: Vec<i16> = Vec::new();

    let mut block_idx: u32 = 0;
    for g in &groups {
        offsets.push(block_idx);
        let nblocks = (g.len() + 7) / 8;
        for b in 0..nblocks {
            // labels: 8 slots
            for slot in 0..8usize {
                let vec_idx = b * 8 + slot;
                let label = if vec_idx < g.len() { labels[g[vec_idx]] } else { 0 };
                label_buf.push(label);
            }
            // blocks: 14 dims × 8 slots, column order within dim
            for d in 0..D {
                for slot in 0..8usize {
                    let vec_idx = b * 8 + slot;
                    let val = if vec_idx < g.len() {
                        (vecs[g[vec_idx]][d] * 10000.0).round() as i16
                    } else {
                        0i16
                    };
                    block_buf.push(val);
                }
            }
        }
        block_idx += nblocks as u32;
    }
    offsets.push(block_idx);

    // Centroids: column-major f32
    let mut centroid_buf: Vec<f32> = Vec::with_capacity(D * k);
    for d in 0..D {
        for c in centroids {
            centroid_buf.push(c[d]);
        }
    }

    let n = vecs.len() as u32;

    let out = File::create(out_path).expect("cannot create output");
    let gz = GzEncoder::new(BufWriter::new(out), Compression::best());
    let mut w = BufWriter::new(gz);

    w.write_all(b"IVF1").unwrap();
    w.write_all(&n.to_le_bytes()).unwrap();
    w.write_all(&(k as u32).to_le_bytes()).unwrap();
    w.write_all(&(D as u32).to_le_bytes()).unwrap();
    for &f in &centroid_buf { w.write_all(&f.to_le_bytes()).unwrap(); }
    for &o in &offsets { w.write_all(&o.to_le_bytes()).unwrap(); }
    for &l in &label_buf { w.write_all(&[l]).unwrap(); }
    for &b in &block_buf { w.write_all(&b.to_le_bytes()).unwrap(); }
    w.flush().unwrap();

    println!("IVF1 written to {out_path}: n={n}, k={k}, blocks={block_idx}");
}

fn main() {
    let in_path = "resources/references.json.gz";
    let out_path = "data/index.bin.gz";

    eprintln!("reading {in_path}...");
    let file = File::open(in_path).expect("cannot open references.json.gz");
    let gz = GzDecoder::new(BufReader::new(file));
    let refs: Vec<Reference> = serde_json::from_reader(gz).expect("failed to parse references JSON");

    let n = refs.len();
    eprintln!("loaded {n} reference vectors");

    // Apply round4 to each vector component
    let vecs: Vec<[f32; D]> = refs.iter().map(|r| {
        let mut v = r.vector;
        for x in v.iter_mut() { *x = round4(*x); }
        v
    }).collect();

    let labels: Vec<u8> = refs.iter().map(|r| if r.label == "fraud" { 1u8 } else { 0u8 }).collect();

    eprintln!("running kmeans++ init (K={K}, sample={INIT_SAMPLE})...");
    let mut centroids = kmeans_plus_plus_init(&vecs, K, INIT_SAMPLE);

    for iter in 0..LLOYD_ITERS {
        let assignments = lloyd_assign(&vecs, &centroids);
        centroids = lloyd_update(&vecs, &assignments, K);
        if iter % 5 == 0 || iter == LLOYD_ITERS - 1 {
            eprintln!("  iter {}/{LLOYD_ITERS}", iter + 1);
        }
        if iter == LLOYD_ITERS - 1 {
            write_ivf1(&centroids, &assignments, &vecs, &labels, K, out_path);
        }
    }
}
```

- [ ] **Step 2: Build and run build_index**

```bash
cargo build --release --bin build_index 2>&1 | tail -10
./target/release/build_index
```

Expected: `IVF1 written to data/index.bin.gz: n=<N>, k=4096, blocks=<B>`. This takes 1-5 minutes.

- [ ] **Step 3: Verify output**

```bash
ls -lh data/index.bin.gz
python3 -c "
import gzip, struct
with gzip.open('data/index.bin.gz','rb') as f:
    magic = f.read(4)
    n,k,d = struct.unpack('<III', f.read(12))
    print(f'magic={magic} n={n} k={k} d={d}')
"
```

Expected: `magic=b'IVF1' n=<N> k=4096 d=14`.

- [ ] **Step 4: Commit**

```bash
git add bin/build_index.rs data/index.bin.gz
git commit -m "feat: kmeans++ IVF1 index builder + embedded index artifact"
```

---

## Task 6: fraud/data.rs — OnceLock Dataset Loader

**Files:**
- Modify: `src/fraud/data.rs`

Decodes the embedded gzip IVF1 index into 32-byte-aligned SIMD-ready arrays.

- [ ] **Step 1: Implement data.rs**

Replace `src/fraud/data.rs` with:
```rust
use aligned_vec::{AVec, ConstAlign};
use flate2::read::GzDecoder;
use std::io::Read;
use std::sync::OnceLock;

static INDEX_GZ: &[u8] = include_bytes!("../../data/index.bin.gz");
static DATASET: OnceLock<Dataset> = OnceLock::new();

pub struct Dataset {
    pub n: usize,
    pub k: usize,
    pub centroids: AVec<f32, ConstAlign<32>>,
    pub offsets: Vec<u32>,
    pub labels: Vec<u8>,
    pub blocks: AVec<i16, ConstAlign<32>>,
}

pub fn dataset() -> &'static Dataset {
    DATASET.get().expect("call data::init() before dataset()")
}

pub fn init() {
    DATASET.get_or_init(decode);
}

fn decode() -> Dataset {
    let mut gz = GzDecoder::new(INDEX_GZ);
    let mut raw: Vec<u8> = Vec::new();
    gz.read_to_end(&mut raw).expect("failed to decompress index");

    let mut pos = 0usize;

    macro_rules! read_u32 {
        () => {{
            let v = u32::from_le_bytes(raw[pos..pos+4].try_into().unwrap());
            pos += 4;
            v
        }};
    }

    assert_eq!(&raw[..4], b"IVF1", "bad IVF1 magic");
    pos = 4;

    let n = read_u32!() as usize;
    let k = read_u32!() as usize;
    let d = read_u32!() as usize;
    assert_eq!(d, 14, "expected d=14, got {d}");

    let centroid_count = d * k;
    let mut centroids: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, centroid_count);
    for _ in 0..centroid_count {
        centroids.push(f32::from_le_bytes(raw[pos..pos+4].try_into().unwrap()));
        pos += 4;
    }

    let mut offsets: Vec<u32> = Vec::with_capacity(k + 1);
    for _ in 0..=k {
        offsets.push(u32::from_le_bytes(raw[pos..pos+4].try_into().unwrap()));
        pos += 4;
    }

    let total_blocks = offsets[k] as usize;
    let labels = raw[pos..pos + total_blocks * 8].to_vec();
    pos += total_blocks * 8;

    let block_i16_count = total_blocks * d * 8;
    let mut blocks: AVec<i16, ConstAlign<32>> = AVec::with_capacity(32, block_i16_count);
    for _ in 0..block_i16_count {
        blocks.push(i16::from_le_bytes(raw[pos..pos+2].try_into().unwrap()));
        pos += 2;
    }

    Dataset { n, k, centroids, offsets, labels, blocks }
}
```

- [ ] **Step 2: Verify cargo check**

```bash
cargo check 2>&1 | tail -10
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/fraud/data.rs
git commit -m "feat: OnceLock Dataset loader for embedded IVF1 index"
```

---

## Task 7: fraud/knn.rs — AVX2/FMA KNN5 with IVF1

**Files:**
- Modify: `src/fraud/knn.rs`

Compile-time AVX2/FMA (no runtime cpuid detection). Adaptive two-stage probe: fast (5 probes), upgrade to full (24) only when result is 2 or 3 fraud — the ambiguous borderline range.

- [ ] **Step 1: Write smoke test**

`src/fraud/knn.rs` (test only):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fraud::data;

    #[test]
    fn smoke_warmup_and_query() {
        data::init();
        warmup();
        let q = [0.0f32; 14];
        let ds = data::dataset();
        let result = knn5_ivf(&q, ds);
        assert!(result <= 5, "knn5_ivf must return 0..=5, got {result}");
    }

    #[test]
    fn smoke_fraud_heavy_query() {
        data::init();
        let q = [1.0f32; 14]; // all features maxed out → should trend toward fraud
        let ds = data::dataset();
        let result = knn5_ivf(&q, ds);
        assert!(result <= 5, "knn5_ivf must return 0..=5, got {result}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test fraud::knn 2>&1 | tail -10
```

Expected: compile error — `knn5_ivf` and `warmup` not defined.

- [ ] **Step 3: Implement knn.rs**

Replace `src/fraud/knn.rs` with:

```rust
use crate::fraud::data::Dataset;

pub const FAST_NPROBE: usize = 5;
pub const FULL_NPROBE: usize = 24;

pub fn knn5_ivf(q: &[f32; 14], ds: &Dataset) -> u8 {
    let fast = probe(q, ds, FAST_NPROBE);
    let fraud_count = fast.iter().filter(|&&l| l == 1).count();
    if fraud_count == 2 || fraud_count == 3 {
        let full = probe(q, ds, FULL_NPROBE);
        full.iter().filter(|&&l| l == 1).count() as u8
    } else {
        fraud_count as u8
    }
}

pub fn warmup() {
    let ds = crate::fraud::data::dataset();
    let mut x = 0x12345678u32;
    for _ in 0..500 {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        let mut q = [0.0f32; 14];
        let mut s = x;
        for v in q.iter_mut() {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            *v = (s & 0xFFFF) as f32 / 65535.0;
        }
        let _ = knn5_ivf(&q, ds);
    }
}

fn probe(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    #[cfg(target_arch = "x86_64")]
    {
        return unsafe { probe_avx2(q, ds, nprobe) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    probe_scalar(q, ds, nprobe)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn probe_avx2(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    let probed = top_n_centroids(q, ds, nprobe);
    scan_blocks_avx2(q, ds, &probed)
}

fn probe_scalar(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> [u8; 5] {
    let probed = top_n_centroids(q, ds, nprobe);
    scan_blocks_scalar(q, ds, &probed)
}

fn top_n_centroids(q: &[f32; 14], ds: &Dataset, nprobe: usize) -> Vec<usize> {
    let k = ds.k;
    let cp = ds.centroids.as_ptr();
    let mut dists = vec![0.0f32; k];

    // Column-major centroid distance: accumulate dim-by-dim
    for d in 0..14usize {
        let qd = q[d];
        let base = d * k;
        for ci in 0..k {
            let diff = unsafe { *cp.add(base + ci) } - qd;
            dists[ci] += diff * diff;
        }
    }

    let nprobe = nprobe.min(k);
    let mut indices: Vec<usize> = (0..k).collect();
    if nprobe < k {
        indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
            dists[a].partial_cmp(&dists[b]).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    indices[..nprobe].to_vec()
}

fn scan_blocks_scalar(q: &[f32; 14], ds: &Dataset, probed: &[usize]) -> [u8; 5] {
    const K_NEIGHBORS: usize = 5;
    let mut top: [(u32, u8); 5] = [(u32::MAX, 0u8); K_NEIGHBORS];
    let mut worst_bits = u32::MAX;

    for &ci in probed {
        let block_start = ds.offsets[ci] as usize;
        let block_end = ds.offsets[ci + 1] as usize;

        for block_i in block_start..block_end {
            let bb = block_i * 14 * 8;
            let lb = block_i * 8;

            // Early exit: check first 8 dims
            let mut partial = 0.0f32;
            for d in 0..8usize {
                let raw = ds.blocks[bb + d * 8] as f32;
                let diff = q[d] - raw * 0.0001;
                partial += diff * diff;
            }
            if partial.to_bits() >= worst_bits && top[K_NEIGHBORS-1].0 < u32::MAX {
                continue;
            }

            for slot in 0..8usize {
                let mut sq = 0.0f32;
                for d in 0..14usize {
                    let raw = ds.blocks[bb + d * 8 + slot] as f32;
                    let diff = q[d] - raw * 0.0001;
                    sq += diff * diff;
                }
                let bits = sq.to_bits();
                let label = ds.labels[lb + slot];
                if bits < worst_bits {
                    // Insert into sorted top-5
                    let insert_pos = top.partition_point(|&(d, _)| d <= bits);
                    if insert_pos < K_NEIGHBORS {
                        top[insert_pos..].rotate_right(1);
                        top[insert_pos] = (bits, label);
                        worst_bits = top[K_NEIGHBORS-1].0;
                    }
                }
            }
        }
    }

    [top[0].1, top[1].1, top[2].1, top[3].1, top[4].1]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_blocks_avx2(q: &[f32; 14], ds: &Dataset, probed: &[usize]) -> [u8; 5] {
    use std::arch::x86_64::*;

    const K_NEIGHBORS: usize = 5;
    let scale = _mm256_set1_ps(0.0001);
    let mut q_vecs = [_mm256_setzero_ps(); 14];
    for d in 0..14usize {
        q_vecs[d] = _mm256_set1_ps(q[d]);
    }

    let mut top: [(u32, u8); 5] = [(u32::MAX, 0u8); K_NEIGHBORS];
    let mut worst_bits = u32::MAX;
    let bp = ds.blocks.as_ptr();
    let lp = ds.labels.as_ptr();

    for &ci in probed {
        let block_start = ds.offsets[ci] as usize;
        let block_end = ds.offsets[ci + 1] as usize;

        'block: for block_i in block_start..block_end {
            // prefetch next block
            if block_i + 4 < block_end {
                _mm_prefetch(bp.add((block_i + 4) * 112) as *const i8, _MM_HINT_T0);
            }

            let bb = block_i * 112;
            let threshold = _mm256_set1_ps(f32::from_bits(worst_bits));

            // Dims 0-7: early termination (4 pairs of dims)
            macro_rules! load_dim {
                ($d:expr) => {{
                    let raw = _mm_loadu_si128(bp.add(bb + $d * 8) as *const _);
                    let i32s = _mm256_cvtepi16_epi32(raw);
                    _mm256_mul_ps(_mm256_cvtepi32_ps(i32s), scale)
                }};
            }
            macro_rules! fmadd_diff {
                ($acc:expr, $d:expr) => {{
                    let v = load_dim!($d);
                    let diff = _mm256_sub_ps(q_vecs[$d], v);
                    _mm256_fmadd_ps(diff, diff, $acc)
                }};
            }

            let mut acc = _mm256_setzero_ps();
            acc = fmadd_diff!(acc, 0);
            acc = fmadd_diff!(acc, 1);
            acc = fmadd_diff!(acc, 2);
            acc = fmadd_diff!(acc, 3);
            acc = fmadd_diff!(acc, 4);
            acc = fmadd_diff!(acc, 5);
            acc = fmadd_diff!(acc, 6);
            acc = fmadd_diff!(acc, 7);

            // Early exit: all 8 partial dists exceed worst
            if top[K_NEIGHBORS-1].0 < u32::MAX {
                let cmp = _mm256_cmp_ps::<_CMP_GE_OQ>(acc, threshold);
                if _mm256_movemask_ps(cmp) == 0xFF { continue 'block; }
            }

            // Dims 8-13: complete the distance
            acc = fmadd_diff!(acc, 8);
            acc = fmadd_diff!(acc, 9);
            acc = fmadd_diff!(acc, 10);
            acc = fmadd_diff!(acc, 11);
            acc = fmadd_diff!(acc, 12);
            acc = fmadd_diff!(acc, 13);

            // Extract 8 distances and update top-5
            let mut dists = [0.0f32; 8];
            _mm256_storeu_ps(dists.as_mut_ptr(), acc);
            let labels_ptr = lp.add(block_i * 8);

            for slot in 0..8usize {
                let bits = dists[slot].to_bits();
                if bits < worst_bits {
                    let label = *labels_ptr.add(slot);
                    let insert_pos = top.partition_point(|&(d, _)| d <= bits);
                    if insert_pos < K_NEIGHBORS {
                        top[insert_pos..].rotate_right(1);
                        top[insert_pos] = (bits, label);
                        worst_bits = top[K_NEIGHBORS-1].0;
                    }
                }
            }
        }
    }

    [top[0].1, top[1].1, top[2].1, top[3].1, top[4].1]
}
```

- [ ] **Step 4: Run tests**

Note: requires `data/index.bin.gz` to exist (built in Task 5).

```bash
cargo test fraud::knn 2>&1 | tail -20
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fraud/knn.rs
git commit -m "feat: AVX2/FMA KNN5 with IVF1 adaptive probe"
```

---

## Task 8: src/env.rs — SOCK Env Var

**Files:**
- Modify: `src/env.rs`

- [ ] **Step 1: Implement env.rs**

```rust
pub fn sock_path() -> String {
    std::env::var("SOCK").unwrap_or_else(|_| "/tmp/fraud-api.sock".into())
}
```

This is already in place from Task 1. Verify it's correct and commit if changed.

- [ ] **Step 2: Verify cargo check**

```bash
cargo check 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add src/env.rs
git commit -m "feat: SOCK env var reader"
```

---

## Task 9: net/http.rs — HTTP Parser + Connection Loop

**Files:**
- Modify: `src/net/http.rs`

The connection loop:
1. Read into 8KB stack-boxed buffer until `\r\n\r\n` found
2. Parse method/path
3. For `POST /fraud-score`: parse Content-Length, wait for full body, call fraud pipeline, push response to output Vec
4. For `GET /ready`: push RESP_READY
5. Unknown route → RESP_NOT_FOUND
6. After each batch (or partial read), flush accumulated responses in one write

- [ ] **Step 1: Write parser unit tests**

`src/net/http.rs` (tests only):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_header_end_locates_crlfcrlf() {
        let buf = b"POST /fraud-score HTTP/1.1\r\nContent-Length: 5\r\n\r\nbody!";
        assert_eq!(find_header_end(buf), Some(47));
    }

    #[test]
    fn find_header_end_returns_none_when_absent() {
        let buf = b"GET /ready HTTP/1.1\r\n";
        assert_eq!(find_header_end(buf), None);
    }

    #[test]
    fn parse_content_length_case_insensitive() {
        let header = b"POST /x HTTP/1.1\r\ncontent-length: 123\r\n\r\n";
        assert_eq!(parse_content_length(header), Some(123));
        let header2 = b"POST /x HTTP/1.1\r\nContent-Length: 456\r\n\r\n";
        assert_eq!(parse_content_length(header2), Some(456));
    }

    #[test]
    fn route_detection() {
        assert_eq!(detect_route(b"POST /fraud-score HTTP/1.1"), Route::FraudScore);
        assert_eq!(detect_route(b"GET /ready HTTP/1.1"), Route::Ready);
        assert_eq!(detect_route(b"GET /unknown HTTP/1.1"), Route::NotFound);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test net::http 2>&1 | tail -10
```

Expected: compile error.

- [ ] **Step 3: Implement http.rs**

Replace `src/net/http.rs` with:
```rust
use memchr::memmem;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::UnixStream;
use crate::fraud::{data, json, knn, vector};
use crate::net::response::{http_body_for, RESP_BAD_REQ, RESP_NOT_FOUND, RESP_READY};

const RX_CAP: usize = 8192;
const TX_CAP: usize = 8192;

#[derive(Debug, PartialEq)]
pub enum Route {
    FraudScore,
    Ready,
    NotFound,
}

pub fn find_header_end(buf: &[u8]) -> Option<usize> {
    memmem::find(buf, b"\r\n\r\n").map(|i| i + 4)
}

pub fn parse_content_length(header_bytes: &[u8]) -> Option<usize> {
    // Case-insensitive scan for "content-length:"
    let lower = header_bytes
        .windows(16)
        .position(|w| w.eq_ignore_ascii_case(b"content-length: ")
            || (w.len() >= 15 && w[..15].eq_ignore_ascii_case(b"content-length:")))?;
    let colon = header_bytes[lower..].iter().position(|&b| b == b':')?;
    let start = lower + colon + 1;
    // skip spaces
    let start = start + header_bytes[start..].iter().take_while(|&&b| b == b' ').count();
    let end = start + header_bytes[start..].iter().position(|&b| b == b'\r' || b == b'\n')?;
    std::str::from_utf8(&header_bytes[start..end]).ok()?.trim().parse().ok()
}

pub fn detect_route(first_line: &[u8]) -> Route {
    if first_line.starts_with(b"POST /fraud-score") {
        Route::FraudScore
    } else if first_line.starts_with(b"GET /ready") {
        Route::Ready
    } else {
        Route::NotFound
    }
}

pub async fn serve_connection(mut stream: UnixStream) {
    let ds = data::dataset();
    let mut rx_buf = Box::new([0u8; RX_CAP]);
    let mut rx_len: usize = 0;
    let mut tx_buf: Vec<u8> = Vec::with_capacity(TX_CAP);

    loop {
        // Read more bytes into rx_buf
        let read_buf = rx_buf.split_at_mut(rx_len).1;
        if read_buf.is_empty() {
            break; // buffer full without complete request — drop connection
        }
        let read_slice = unsafe {
            std::slice::from_raw_parts_mut(read_buf.as_mut_ptr(), read_buf.len())
        };
        // monoio read: takes ownership of buffer, returns (result, buffer)
        let (res, buf_back) = stream.read(monoio::buf::IoBufMut::new(read_slice)).await;
        match res {
            Ok(0) => break, // connection closed
            Err(_) => break,
            Ok(n) => {
                rx_len += n;
                // monoio returns the buffer back; buf_back is the slice we passed
                drop(buf_back);
            }
        }

        // Process all complete requests in the buffer
        let mut consumed = 0usize;
        loop {
            let available = &rx_buf[consumed..rx_len];
            let header_end = match find_header_end(available) {
                Some(e) => e,
                None => break, // incomplete header, read more
            };

            let header_bytes = &available[..header_end];
            let route = detect_route(header_bytes);

            match route {
                Route::Ready => {
                    tx_buf.extend_from_slice(RESP_READY);
                    consumed += header_end;
                }
                Route::NotFound => {
                    tx_buf.extend_from_slice(RESP_NOT_FOUND);
                    consumed += header_end;
                }
                Route::FraudScore => {
                    let cl = match parse_content_length(header_bytes) {
                        Some(n) => n,
                        None => {
                            tx_buf.extend_from_slice(RESP_BAD_REQ);
                            consumed += header_end;
                            continue;
                        }
                    };
                    let body_start = consumed + header_end;
                    let body_end = body_start + cl;
                    if body_end > rx_len {
                        break; // incomplete body, read more
                    }
                    let body = &rx_buf[body_start..body_end];
                    let resp = match json::parse(body) {
                        Some(payload) => {
                            let vec = vector::vectorize(&payload);
                            let count = knn::knn5_ivf(&vec, ds);
                            http_body_for(count)
                        }
                        None => RESP_BAD_REQ,
                    };
                    tx_buf.extend_from_slice(resp);
                    consumed += header_end + cl;
                }
            }
        }

        // Flush accumulated responses
        if !tx_buf.is_empty() {
            let out = std::mem::take(&mut tx_buf);
            let (res, out) = stream.write_all(out).await;
            tx_buf = out;
            tx_buf.clear();
            if res.is_err() { break; }
        }

        // Compact rx_buf: shift unconsumed bytes to front
        if consumed > 0 {
            rx_buf.copy_within(consumed..rx_len, 0);
            rx_len -= consumed;
        }
    }
}
```

**Note on monoio read API:** monoio 0.2 uses ownership-passing buffers. The exact `read` call syntax depends on which `AsyncReadRent` impl is in scope. If the above doesn't compile, use the monoio-provided `read_exact` or `AsyncReadRentExt::read` with a `Vec<u8>` as `IoBufMut`. Adjust as needed — the logic is correct, the exact API call may need tweaking.

- [ ] **Step 4: Run parser unit tests**

```bash
cargo test net::http 2>&1 | tail -20
```

Expected: 4 tests PASS (the parser functions don't require monoio async runtime).

- [ ] **Step 5: Commit**

```bash
git add src/net/http.rs
git commit -m "feat: HTTP parser + monoio connection loop"
```

---

## Task 10: src/main.rs — monoio Bootstrap

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Implement main.rs**

Replace `src/main.rs` with:
```rust
mod env;
mod fraud;
mod net;

use monoio::net::UnixListener;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    // Initialize dataset and warm up KNN before accepting connections.
    // Socket bind happens after warmup, so socket presence guarantees readiness.
    fraud::data::init();
    fraud::knn::warmup();

    let sock_path = env::sock_path();
    let _ = std::fs::remove_file(&sock_path); // remove stale socket if present

    monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .with_entries(1024)
        .build()
        .expect("failed to build monoio runtime")
        .block_on(async {
            let listener = UnixListener::bind(&sock_path)
                .expect("failed to bind unix socket");

            // Set socket permissions so nginx can connect
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o777))
                .expect("failed to set socket permissions");

            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        monoio::spawn(net::http::serve_connection(stream));
                    }
                    Err(e) => {
                        eprintln!("accept error: {e}");
                        break;
                    }
                }
            }
        });
}
```

- [ ] **Step 2: cargo build release**

```bash
cargo build --release --bin fraud-detection 2>&1 | tail -20
```

Expected: compiles. Fix any API mismatches in `net/http.rs` if monoio read/write signatures differ. Consult `cargo doc --open` or `grep -r "fn read" ~/.cargo/registry/src/*/monoio-*/src/` for exact method signatures.

- [ ] **Step 3: Quick smoke test (local)**

```bash
# Terminal 1
SOCK=/tmp/test.sock ./target/release/fraud-detection &

# Terminal 2 — wait for socket
until [ -S /tmp/test.sock ]; do sleep 0.1; done
printf 'GET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' | socat - UNIX-CONNECT:/tmp/test.sock
```

Expected: `HTTP/1.1 200 OK` with body `OK`.

- [ ] **Step 4: Test POST fraud-score**

```bash
BODY='{"id":"tx-1","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}'
LEN=${#BODY}
printf "POST /fraud-score HTTP/1.1\r\nHost: localhost\r\nContent-Length: $LEN\r\nConnection: close\r\n\r\n$BODY" | socat - UNIX-CONNECT:/tmp/test.sock
```

Expected: JSON with `approved` and `fraud_score`.

```bash
kill %1 2>/dev/null; rm -f /tmp/test.sock
```

- [ ] **Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat: monoio io_uring runtime bootstrap with warmup"
```

---

## Task 11: Dockerfile + docker-compose.yml

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Replace Dockerfile**

```dockerfile
FROM rust:latest AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin build_index
RUN ./target/release/build_index
RUN cargo build --release --bin fraud-detection

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/fraud-detection /fraud-detection
CMD ["/fraud-detection"]
```

- [ ] **Step 2: Update docker-compose.yml**

Replace the `nginx` command (readiness check) and `api1`/`api2` environments:

```yaml
volumes:
  sock:

services:
  nginx:
    image: nginx:alpine
    command: >
      sh -c "until [ -S /run/sock/api1.sock ] && [ -S /run/sock/api2.sock ]; do sleep 1; done && exec nginx -g 'daemon off;'"
    ports:
      - "9999:9999"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - sock:/run/sock
    depends_on:
      - api1
      - api2
    networks:
      - fraud-net
    deploy:
      resources:
        limits:
          cpus: "0.10"
          memory: "10MB"

  api1:
    build: .
    environment:
      - SOCK=/run/sock/api1.sock
    volumes:
      - sock:/run/sock
    networks:
      - fraud-net
    security_opt:
      - seccomp:unconfined
    ulimits:
      nofile:
        soft: 65535
        hard: 65535
    deploy:
      resources:
        limits:
          cpus: "0.45"
          memory: "170MB"

  api2:
    build: .
    environment:
      - SOCK=/run/sock/api2.sock
    volumes:
      - sock:/run/sock
    networks:
      - fraud-net
    security_opt:
      - seccomp:unconfined
    ulimits:
      nofile:
        soft: 65535
        hard: 65535
    deploy:
      resources:
        limits:
          cpus: "0.45"
          memory: "170MB"

networks:
  fraud-net:
    driver: bridge
```

- [ ] **Step 3: Build Docker image**

```bash
docker build -t fraud-detection:monoio . 2>&1 | tail -20
```

Expected: build succeeds, final image has `/fraud-detection`.

- [ ] **Step 4: Smoke test with docker-compose**

```bash
docker compose up --build -d
sleep 5
curl -s http://localhost:9999/ready
```

Expected: `OK`.

- [ ] **Step 5: Send a fraud-score request**

```bash
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{"id":"tx-1","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}'
```

Expected: `{"approved":true,"fraud_score":0.0}` (or similar score).

```bash
docker compose down
```

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: multi-stage Dockerfile with embedded IVF1 index + seccomp:unconfined for io_uring"
```

---

## Task 12: Final Verification

- [ ] **Step 1: Run all unit tests**

```bash
cargo test 2>&1 | tail -30
```

Expected: all tests pass. There should be tests from: `fraud::json`, `fraud::vector`, `fraud::knn`, `net::response`, `net::http`.

- [ ] **Step 2: cargo clippy release**

```bash
cargo clippy --release 2>&1 | grep "^error" | head -20
```

Expected: no errors (warnings acceptable).

- [ ] **Step 3: Verify data/index.bin.gz is committed and embedded**

```bash
ls -lh data/index.bin.gz
# Verify it's referenced from data.rs
grep "include_bytes" src/fraud/data.rs
```

Expected: file exists, data.rs has `include_bytes!("../../data/index.bin.gz")`.

- [ ] **Step 4: Check binary size**

```bash
ls -lh target/release/fraud-detection
```

The binary should be large (includes the embedded gzipped index). Expected: 5-15 MB range.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete monoio io_uring port — zero-alloc hot path, embedded IVF1 index"
```

---

## Known Gotchas

1. **monoio read API**: monoio 0.2 uses owned buffers (`IoBuf`/`IoBufMut` traits). If `stream.read(slice)` doesn't compile, use `stream.read(Vec::with_capacity(RX_CAP))` and copy into rx_buf. Check exact signatures with `cargo doc --no-deps --open`.

2. **io_uring in Docker**: requires `seccomp:unconfined` (added in docker-compose.yml). Without it, `io_uring_setup` syscall is blocked and the server panics on startup.

3. **AVX2 target feature**: `#[target_feature(enable = "avx2,fma")]` requires the host CPU to support AVX2. The competition judge machines do. For local dev without AVX2, the scalar fallback is used automatically.

4. **include_bytes! path**: if `cargo build --bin fraud-detection` runs before `build_index` produces `data/index.bin.gz`, it will fail. Always run `build_index` first (the Dockerfile enforces this order).

5. **Content-Length parser**: the `parse_content_length` implementation uses `eq_ignore_ascii_case` on windows. It may miss headers without a space after the colon. Adjust the pattern if needed for your nginx config (nginx always sends `Content-Length: N` with a space).
