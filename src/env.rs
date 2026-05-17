pub fn sock_path() -> String { std::env::var("SOCK").unwrap_or_else(|_| "/tmp/api.sock".into()) }
