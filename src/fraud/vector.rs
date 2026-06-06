use crate::fraud::json::Payload;

#[inline(always)]
fn clamp01_i16(x: f32) -> i16 {
    (x.clamp(0.0, 1.0) * 10000.0).round() as i16
}

#[inline(always)]
fn mcc_risk(mcc: u32) -> i16 {
    match mcc {
        5411 => 1500,
        5812 => 3000,
        5912 => 2000,
        5944 => 4500,
        7801 => 8000,
        7802 => 7500,
        7995 => 8500,
        4511 => 3500,
        5311 => 2500,
        5999 => 5000,
        _ => 5000,
    }
}

pub fn vectorize(p: &Payload) -> [i16; 16] {
    let mut v = [0i16; 16];
    v[0] = clamp01_i16(p.amount / 10_000.0);
    v[1] = clamp01_i16(p.installments as f32 / 12.0);
    v[2] = if p.customer_avg_amount > 0.0 {
        clamp01_i16((p.amount / p.customer_avg_amount) / 10.0)
    } else {
        0
    };
    v[3] = clamp01_i16(p.hour as f32 / 23.0);
    v[4] = clamp01_i16(p.weekday as f32 / 6.0);
    if p.has_last_tx {
        v[5] = clamp01_i16(p.minutes_since_last / 1440.0);
        v[6] = clamp01_i16(p.km_from_current / 1000.0);
    } else {
        v[5] = -10000;
        v[6] = -10000;
    }
    v[7] = clamp01_i16(p.km_from_home / 1000.0);
    v[8] = clamp01_i16(p.tx_count_24h as f32 / 20.0);
    v[9] = if p.is_online { 10000 } else { 0 };
    v[10] = if p.card_present { 10000 } else { 0 };
    v[11] = if p.is_unknown_merchant { 10000 } else { 0 };
    v[12] = mcc_risk(p.mcc);
    v[13] = clamp01_i16(p.merchant_avg_amount / 10_000.0);
    // v[14], v[15] = 0 (SIMD padding)
    v
}

pub fn tag_from_request(p: &Payload) -> usize {
    (p.card_present as usize) << 3
        | (p.is_online as usize) << 2
        | (p.is_unknown_merchant as usize) << 1
        | (p.has_last_tx as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            km_from_home: 29.233_103_f32,
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
            weekday: 4,
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
    fn test_vectorize_legit_i16() {
        let v = vectorize(&legit_payload());
        assert_eq!(v[0], 41, "dim0: 41.12/10000=0.004112 → 41");
        assert_eq!(v[1], 1667, "dim1: 2/12=0.1667 → 1667");
        assert_eq!(v[2], 500, "dim2: (41.12/82.24)/10=0.05 → 500");
        assert_eq!(v[3], 7826, "dim3: 18/23=0.7826 → 7826");
        assert_eq!(v[4], 3333, "dim4: 2/6=0.3333 → 3333");
        assert_eq!(v[5], -10000, "dim5: no last_tx → sentinel");
        assert_eq!(v[6], -10000, "dim6: no last_tx → sentinel");
        assert_eq!(v[7], 292, "dim7: 29.233/1000=0.02923 → 292");
        assert_eq!(v[8], 1500, "dim8: 3/20=0.15 → 1500");
        assert_eq!(v[9], 0, "dim9: is_online=false");
        assert_eq!(v[10], 10000, "dim10: card_present=true");
        assert_eq!(v[11], 0, "dim11: unknown=false");
        assert_eq!(v[12], 1500, "dim12: mcc 5411 → 0.15 → 1500");
        assert_eq!(v[13], 60, "dim13: 60.25/10000=0.006025 → 60 (rounded)");
        assert_eq!(v[14], 0, "dim14: padding");
        assert_eq!(v[15], 0, "dim15: padding");
    }

    #[test]
    fn test_vectorize_fraud_i16() {
        let v = vectorize(&fraud_payload());
        assert_eq!(v[0], 9506, "dim0: 9505.97/10000 clamped → 9506");
        assert_eq!(v[2], 10000, "dim2: ratio > 10 → clamped to 10000");
        assert_eq!(v[8], 10000, "dim8: 20/20 = 1.0 → 10000");
        assert_eq!(v[11], 10000, "dim11: unknown=true → 10000");
        assert_eq!(v[12], 7500, "dim12: mcc 7802 → 0.75 → 7500");
    }

    #[test]
    fn test_vectorize_with_last_tx() {
        let mut p = legit_payload();
        p.has_last_tx = true;
        p.minutes_since_last = 325.0;
        p.km_from_current = 18.8626;
        let v = vectorize(&p);
        assert_eq!(v[5], 2257, "dim5: 325/1440=0.2257 → 2257");
        assert_eq!(v[6], 189, "dim6: 18.8626/1000=0.01886 → 189");
    }

    #[test]
    fn test_mcc_unknown_defaults() {
        let mut p = legit_payload();
        p.mcc = 9999;
        let v = vectorize(&p);
        assert_eq!(v[12], 5000, "unknown mcc → 0.5 → 5000");
    }

    #[test]
    fn test_tag_all_16_combinations() {
        let mut p = legit_payload();
        // card_present=1, online=0, unknown=0, has_last=0 → tag=8
        p.card_present = true; p.is_online = false; p.is_unknown_merchant = false; p.has_last_tx = false;
        assert_eq!(tag_from_request(&p), 8);
        // card_present=0, online=1, unknown=1, has_last=1 → tag=7
        p.card_present = false; p.is_online = true; p.is_unknown_merchant = true; p.has_last_tx = true;
        assert_eq!(tag_from_request(&p), 7);
        // all zero → tag=0
        p.card_present = false; p.is_online = false; p.is_unknown_merchant = false; p.has_last_tx = false;
        assert_eq!(tag_from_request(&p), 0);
        // all bits → tag=15
        p.card_present = true; p.is_online = true; p.is_unknown_merchant = true; p.has_last_tx = true;
        assert_eq!(tag_from_request(&p), 15);
    }

    #[test]
    fn test_padding_dims_always_zero() {
        let v = vectorize(&legit_payload());
        assert_eq!(v[14], 0);
        assert_eq!(v[15], 0);
        let v2 = vectorize(&fraud_payload());
        assert_eq!(v2[14], 0);
        assert_eq!(v2[15], 0);
    }
}
