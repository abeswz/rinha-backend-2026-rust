use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub static FAST_PATH_COUNT: AtomicU64 = AtomicU64::new(0);
pub static IVF_COUNT: AtomicU64 = AtomicU64::new(0);
pub static FAST_PROBE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static FULL_PROBE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static FAST_PROBE_TOTAL_US: AtomicU64 = AtomicU64::new(0);
pub static FULL_PROBE_TOTAL_US: AtomicU64 = AtomicU64::new(0);
pub static REQUEST_TOTAL: AtomicU64 = AtomicU64::new(0);

pub struct Snapshot {
    pub fast_path_count: u64,
    pub ivf_count: u64,
    pub fast_probe_count: u64,
    pub full_probe_count: u64,
    pub fast_probe_total_us: u64,
    pub full_probe_total_us: u64,
    pub request_total: u64,
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        fast_path_count: FAST_PATH_COUNT.load(Relaxed),
        ivf_count: IVF_COUNT.load(Relaxed),
        fast_probe_count: FAST_PROBE_COUNT.load(Relaxed),
        full_probe_count: FULL_PROBE_COUNT.load(Relaxed),
        fast_probe_total_us: FAST_PROBE_TOTAL_US.load(Relaxed),
        full_probe_total_us: FULL_PROBE_TOTAL_US.load(Relaxed),
        request_total: REQUEST_TOTAL.load(Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reads_all_seven_counters() {
        FAST_PATH_COUNT.store(1, Relaxed);
        IVF_COUNT.store(2, Relaxed);
        FAST_PROBE_COUNT.store(3, Relaxed);
        FULL_PROBE_COUNT.store(4, Relaxed);
        FAST_PROBE_TOTAL_US.store(500, Relaxed);
        FULL_PROBE_TOTAL_US.store(600, Relaxed);
        REQUEST_TOTAL.store(7, Relaxed);

        let s = snapshot();
        assert_eq!(s.fast_path_count, 1);
        assert_eq!(s.ivf_count, 2);
        assert_eq!(s.fast_probe_count, 3);
        assert_eq!(s.full_probe_count, 4);
        assert_eq!(s.fast_probe_total_us, 500);
        assert_eq!(s.full_probe_total_us, 600);
        assert_eq!(s.request_total, 7);

        FAST_PATH_COUNT.store(0, Relaxed);
        IVF_COUNT.store(0, Relaxed);
        FAST_PROBE_COUNT.store(0, Relaxed);
        FULL_PROBE_COUNT.store(0, Relaxed);
        FAST_PROBE_TOTAL_US.store(0, Relaxed);
        FULL_PROBE_TOTAL_US.store(0, Relaxed);
        REQUEST_TOTAL.store(0, Relaxed);
    }
}
