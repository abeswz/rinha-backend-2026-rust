use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub ivf_path: PathBuf,
    pub mcc_path: PathBuf,
    pub norm_path: PathBuf,
    pub nprobe: usize,
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
                std::env::var("IVF_PATH")
                    .unwrap_or_else(|_| "resources/ivf_index.bin".to_string()),
            ),
            mcc_path: PathBuf::from(
                std::env::var("MCC_PATH")
                    .unwrap_or_else(|_| "resources/mcc_risk.json".to_string()),
            ),
            norm_path: PathBuf::from(
                std::env::var("NORM_PATH")
                    .unwrap_or_else(|_| "resources/normalization.json".to_string()),
            ),
            nprobe: std::env::var("IVF_NPROBE")
                .unwrap_or_else(|_| "8".to_string())
                .parse()
                .expect("IVF_NPROBE must be a valid integer"),
        }
    }
}
