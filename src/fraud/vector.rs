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
        _ => 0.50,
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
    fn test_round4() {
        assert_eq!(round4(0.004112), 0.0041);
        assert_eq!(round4(-1.0), -1.0);
        assert_eq!(round4(1.0), 1.0);
        assert!((round4(0.166667) - 0.1667).abs() < 0.00001);
    }

    #[test]
    fn test_vectorize_legit() {
        let v = vectorize(&legit_payload());
        assert!((v[0] - 0.0041).abs() < 0.0001, "dim0 got {}", v[0]);
        assert!((v[1] - 0.1667).abs() < 0.0001, "dim1 got {}", v[1]);
        assert!((v[2] - 0.05).abs() < 0.0001, "dim2 got {}", v[2]);
        assert!((v[3] - 0.7826).abs() < 0.0001, "dim3 got {}", v[3]);
        assert!((v[4] - 0.3333).abs() < 0.0001, "dim4 got {}", v[4]);
        assert_eq!(v[5], -1.0, "dim5 should be -1.0");
        assert_eq!(v[6], -1.0, "dim6 should be -1.0");
        assert!((v[7] - 0.0292).abs() < 0.0001, "dim7 got {}", v[7]);
        assert!((v[8] - 0.15).abs() < 0.0001, "dim8 got {}", v[8]);
        assert_eq!(v[9], 0.0);
        assert_eq!(v[10], 1.0);
        assert_eq!(v[11], 0.0);
        assert!((v[12] - 0.15).abs() < 0.0001, "dim12 got {}", v[12]);
        assert!((v[13] - 0.006).abs() < 0.0001, "dim13 got {}", v[13]);
    }

    #[test]
    fn test_vectorize_fraud() {
        let v = vectorize(&fraud_payload());
        assert!((v[0] - 0.9506).abs() < 0.0001, "dim0 got {}", v[0]);
        assert_eq!(v[2], 1.0, "dim2 should be clamped to 1.0");
        assert_eq!(v[8], 1.0, "dim8 = 20/20 = 1.0");
        assert_eq!(v[11], 1.0);
        assert!((v[12] - 0.75).abs() < 0.0001, "dim12 got {}", v[12]);
    }

    #[test]
    fn test_mcc_unknown_defaults_to_0_5() {
        let mut p = legit_payload();
        p.mcc = 9999;
        let v = vectorize(&p);
        assert!(
            (v[12] - 0.5).abs() < 0.0001,
            "unknown mcc should default to 0.5, got {}",
            v[12]
        );
    }

    #[test]
    fn test_with_last_tx() {
        let mut p = legit_payload();
        p.has_last_tx = true;
        p.minutes_since_last = 325.0;
        p.km_from_current = 18.8626;
        let v = vectorize(&p);
        assert!((v[5] - 0.2257).abs() < 0.0001, "dim5 got {}", v[5]);
        assert!((v[6] - 0.0189).abs() < 0.0001, "dim6 got {}", v[6]);
    }
}
