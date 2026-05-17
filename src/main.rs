use fraud_detection::{config::Config, web::router::build_router, AppState};
use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Under 0.475 CPU cgroup (competition constraint): more threads = cgroup throttling = reactor starvation.
// These constants caused p99=2001ms + 1986 HTTP errors when WORKER=4 + unbounded blocking pool.
// Under 0.475 CPU cgroup per instance: WORKER=4 + unbounded blocking caused p99=2001ms + 1986 HTTP errors.
// KNN runs inline on async workers (no spawn_blocking). Only startup I/O (AppState::build) uses blocking threads.
const TOKIO_WORKER_THREADS: usize = 2;
const TOKIO_MAX_BLOCKING_THREADS: usize = 1;

fn main() {
    fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let config = Config::from_env();
    let state = Arc::new(AppState::build(&config).expect("failed to initialize AppState"));
    let addr = format!("0.0.0.0:{}", config.port);
    let socket_path = std::env::var("SOCKET_PATH").ok();
    let router = build_router(state);

    tracing::info!("listening on {addr}");

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(TOKIO_WORKER_THREADS)
        .max_blocking_threads(TOKIO_MAX_BLOCKING_THREADS)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokio_runtime_config_bounded_for_cgroup_throttling() {
        // 0.475 CPU cgroup per instance: each extra thread steals from reactor.
        // WORKER=4 + unbounded blocking → p99=2001ms + 1986 HTTP errors in competition.
        assert_eq!(TOKIO_WORKER_THREADS, 2, "worker_threads must be 2 on 0.475 CPU cgroup");
        assert_eq!(
            TOKIO_MAX_BLOCKING_THREADS, 1,
            "max_blocking_threads must be 1: KNN runs inline, only startup I/O needs a blocking thread"
        );
    }
}
