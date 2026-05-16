use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub ivf_path: PathBuf,
    pub mcc_path: PathBuf,
    pub norm_path: PathBuf,
    pub nprobe_fast: usize,
    pub nprobe_slow: usize,
}

impl Config {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        Self {
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .expect("PORT must be a valid port number"),
            ivf_path: PathBuf::from(
                std::env::var("IVF_PATH").unwrap_or_else(|_| "resources/ivf_index.bin".to_string()),
            ),
            mcc_path: PathBuf::from(
                std::env::var("MCC_PATH").unwrap_or_else(|_| "resources/mcc_risk.json".to_string()),
            ),
            norm_path: PathBuf::from(
                std::env::var("NORM_PATH")
                    .unwrap_or_else(|_| "resources/normalization.json".to_string()),
            ),
            nprobe_fast: std::env::var("IVF_NPROBE_FAST")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .expect("IVF_NPROBE_FAST must be a valid integer"),
            nprobe_slow: std::env::var("IVF_NPROBE")
                .unwrap_or_else(|_| "24".to_string())
                .parse()
                .expect("IVF_NPROBE must be a valid integer"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-var tests to avoid races between parallel test threads
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_config_nprobe_fast_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("IVF_NPROBE_FAST");
        let config = Config::from_env();
        assert_eq!(config.nprobe_fast, 3);
    }

    #[test]
    fn test_config_nprobe_fast_from_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("IVF_NPROBE_FAST", "5");
        let config = Config::from_env();
        assert_eq!(config.nprobe_fast, 5);
        std::env::remove_var("IVF_NPROBE_FAST");
    }
}
