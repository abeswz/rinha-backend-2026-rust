use memchr::memchr;

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
    while *pos < buf.len()
        && (buf[*pos] == b' ' || buf[*pos] == b'\n' || buf[*pos] == b'\r' || buf[*pos] == b'\t')
    {
        *pos += 1;
    }
    Some(())
}

// Reads a quoted string, returns its byte slice, advances pos past closing '"'.
#[inline(always)]
fn read_string<'a>(buf: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    if buf.get(*pos) != Some(&b'"') {
        return None;
    }
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
        if c == b','
            || c == b'}'
            || c == b']'
            || c == b' '
            || c == b'\n'
            || c == b'\r'
            || c == b'\t'
        {
            break;
        }
        *pos += 1;
    }
    if *pos == start {
        return None;
    }
    Some(&buf[start..*pos])
}

// Parses "YYYY-MM-DDTHH:MM:SS" from buf[pos..pos+19], returns (y,mo,d,h,min).
fn parse_iso(buf: &[u8], pos: usize) -> Option<(u16, u8, u8, u8, u8)> {
    if buf.len() < pos + 19 {
        return None;
    }
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
    let b = (d.get(off + 1)? - b'0') as u16;
    let c = (d.get(off + 2)? - b'0') as u16;
    let e = (d.get(off + 3)? - b'0') as u16;
    Some(a * 1000 + b * 100 + c * 10 + e)
}

#[inline(always)]
fn parse_digits2(d: &[u8], off: usize) -> Option<u8> {
    let a = d.get(off)? - b'0';
    let b = d.get(off + 1)? - b'0';
    Some(a * 10 + b)
}

fn days_since_epoch(y: u16, mo: u8, d: u8) -> u32 {
    let (y, m) = if mo <= 2 {
        (y as i32 - 1, mo as i32 + 12)
    } else {
        (y as i32, mo as i32)
    };
    let a = y / 100;
    let b = 2 - a + a / 4;
    ((365.25 * (y + 4716) as f64) as i32 + (30.6001 * (m + 1) as f64) as i32 + d as i32 + b - 1524)
        as u32
}

fn minutes_between(cur: (u16, u8, u8, u8, u8), prev: (u16, u8, u8, u8, u8)) -> f32 {
    let dc = days_since_epoch(cur.0, cur.1, cur.2) as i64;
    let dp = days_since_epoch(prev.0, prev.1, prev.2) as i64;
    let mc = cur.3 as i64 * 60 + cur.4 as i64;
    let mp = prev.3 as i64 * 60 + prev.4 as i64;
    ((dc - dp) * 1440 + (mc - mp)) as f32
}

fn date_weekday(y: u16, mo: u8, d: u8) -> u8 {
    let t = [0u8, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if mo < 3 { y - 1 } else { y };
    let dow_sun0 = (y as u32 + y as u32 / 4 - y as u32 / 100
        + y as u32 / 400
        + t[(mo - 1) as usize] as u32
        + d as u32)
        % 7;
    ((dow_sun0 + 6) % 7) as u8
}

pub fn parse(buf: &[u8]) -> Option<Payload> {
    let mut pos = memchr(b'{', buf)?;
    pos += 1;

    // "id": skip value
    skip_to_value(buf, &mut pos)?;
    read_string(buf, &mut pos)?;

    // "transaction" key, then "amount" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    skip_to_value(buf, &mut pos)?;
    let installments: u8 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "requested_at"
    skip_to_value(buf, &mut pos)?;
    if buf.get(pos) != Some(&b'"') {
        return None;
    }
    let dt_start = pos + 1;
    let (y, mo, d, hour, _min_ignored) = parse_iso(buf, dt_start)?;
    let weekday = date_weekday(y, mo, d);
    let cur_time = (y, mo, d, hour, parse_digits2(buf, dt_start + 14)?);
    pos = dt_start + memchr(b'"', &buf[dt_start..])? + 1;

    // "customer" key, then "avg_amount" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let customer_avg_amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    skip_to_value(buf, &mut pos)?;
    let tx_count_24h: u8 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "known_merchants": array
    skip_to_value(buf, &mut pos)?;
    pos += memchr(b'[', &buf[pos..])? + 1;

    const MAX_KNOWN: usize = 32;
    const MAX_ID_LEN: usize = 16;
    let mut known_buf = [[0u8; MAX_ID_LEN]; MAX_KNOWN];
    let mut known_lens = [0u8; MAX_KNOWN];
    let mut known_count: usize = 0;

    loop {
        while pos < buf.len() && matches!(buf[pos], b' ' | b'\n' | b'\r' | b'\t' | b',') {
            pos += 1;
        }
        if pos >= buf.len() {
            return None;
        }
        if buf[pos] == b']' {
            pos += 1;
            break;
        }
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

    // "merchant" key, then "id" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let merch_id = read_string(buf, &mut pos)?;
    let is_unknown_merchant = !(0..known_count).any(|i| {
        let len = known_lens[i] as usize;
        merch_id.len() == len && merch_id == &known_buf[i][..len]
    });

    skip_to_value(buf, &mut pos)?;
    let mcc_str = read_string(buf, &mut pos)?;
    let mcc: u32 = std::str::from_utf8(mcc_str).ok()?.parse().ok()?;

    skip_to_value(buf, &mut pos)?;
    let merchant_avg_amount: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "terminal" key, then "is_online" key
    skip_to_value(buf, &mut pos)?;
    skip_to_value(buf, &mut pos)?;
    let tok = read_token(buf, &mut pos)?;
    let is_online = tok == b"true";

    skip_to_value(buf, &mut pos)?;
    let tok = read_token(buf, &mut pos)?;
    let card_present = tok == b"true";

    skip_to_value(buf, &mut pos)?;
    let km_from_home: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
        .ok()?
        .parse()
        .ok()?;

    // "last_transaction": null | { ... }
    skip_to_value(buf, &mut pos)?;

    while pos < buf.len() && matches!(buf[pos], b' ' | b'\n' | b'\r' | b'\t') {
        pos += 1;
    }

    let (has_last_tx, minutes_since_last, km_from_current) =
        if buf.get(pos..pos + 4) == Some(b"null") {
            (false, 0.0f32, 0.0f32)
        } else if buf.get(pos) == Some(&b'{') {
            pos += 1;

            skip_to_value(buf, &mut pos)?;
            if buf.get(pos) != Some(&b'"') {
                return None;
            }
            let ts_start = pos + 1;
            let prev_time = (
                parse_digits4(buf, ts_start)?,
                parse_digits2(buf, ts_start + 5)?,
                parse_digits2(buf, ts_start + 8)?,
                parse_digits2(buf, ts_start + 11)?,
                parse_digits2(buf, ts_start + 14)?,
            );
            pos = ts_start + memchr(b'"', &buf[ts_start..])? + 1;

            skip_to_value(buf, &mut pos)?;
            let km_cur: f32 = std::str::from_utf8(read_token(buf, &mut pos)?)
                .ok()?
                .parse()
                .ok()?;

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
