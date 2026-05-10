use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub refs_path: PathBuf,
    pub mcc_path: PathBuf,
    pub norm_path: PathBuf,
}

impl Config {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        Self {
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .expect("PORT must be a valid port number"),
            refs_path: PathBuf::from(
                std::env::var("REFS_PATH").unwrap_or_else(|_| "resources/refs.bin".to_string()),
            ),
            mcc_path: PathBuf::from(
                std::env::var("MCC_PATH").unwrap_or_else(|_| "resources/mcc_risk.json".to_string()),
            ),
            norm_path: PathBuf::from(
                std::env::var("NORM_PATH")
                    .unwrap_or_else(|_| "resources/normalization.json".to_string()),
            ),
        }
    }
}
