mod env;
mod fraud;
mod metrics;
mod net;

use tokio::net::UnixListener;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(2)
        .enable_all()
        .build()
        .expect("failed to build runtime")
        .block_on(run());
}

async fn run() {
    fraud::data::init();
    fraud::model::init();
    fraud::knn::warmup();

    let sock_path = env::sock_path();
    let _ = std::fs::remove_file(&sock_path);

    let listener = UnixListener::bind(&sock_path).expect("failed to bind unix socket");

    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o777))
        .expect("failed to set socket permissions");

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                tokio::spawn(net::http::serve_connection(stream));
            }
            Err(e) => {
                eprintln!("accept error: {e}");
                break;
            }
        }
    }
}
