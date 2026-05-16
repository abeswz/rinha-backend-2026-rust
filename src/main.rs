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
    let socket_path = std::env::var("SOCKET_PATH").ok();
    let router = build_router(state);

    tracing::info!("listening on {addr}");

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
        .block_on(async {
            let tcp = tokio::net::TcpListener::bind(&addr)
                .await
                .expect("failed to bind tcp listener");

            if let Some(path) = socket_path {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::remove_file(&path);
                let unix =
                    tokio::net::UnixListener::bind(&path).expect("failed to bind unix socket");
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o777))
                    .expect("failed to set socket permissions");
                tracing::info!("also listening on unix:{path}");
                let (r1, r2) =
                    tokio::join!(axum::serve(tcp, router.clone()), axum::serve(unix, router),);
                r1.expect("tcp server error");
                r2.expect("unix server error");
            } else {
                axum::serve(tcp, router).await.expect("server error");
            }
        });
}
