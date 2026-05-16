use fraud_detection::{config::Config, web::router::build_router, AppState};
use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let config = Config::from_env();
    let state = Arc::new(AppState::build(&config).expect("failed to initialize AppState"));
    let addr = format!("0.0.0.0:{}", config.port);
    let router = build_router(state);

    tracing::info!("listening on {addr}");

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(2)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("failed to bind listener");
            axum::serve(listener, router).await.expect("server error");
        });
}
