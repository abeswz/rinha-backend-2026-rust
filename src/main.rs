mod env;
mod fraud;
mod net;

use monoio::net::{ListenerOpts, UnixListener};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    // Initialize dataset and warm up KNN before accepting connections.
    // Socket bind happens after warmup, so socket presence guarantees readiness.
    fraud::data::init();
    fraud::knn::warmup();

    let sock_path = env::sock_path();
    let _ = std::fs::remove_file(&sock_path); // remove stale socket if present

    monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .with_entries(1024)
        .build()
        .expect("failed to build monoio runtime")
        .block_on(async {
            let opts = ListenerOpts::new()
                .reuse_port(false)
                .reuse_addr(false);
            let listener = UnixListener::bind_with_config(&sock_path, &opts)
                .expect("failed to bind unix socket");

            // Set socket permissions so nginx can connect
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o777))
                .expect("failed to set socket permissions");

            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        monoio::spawn(net::http::serve_connection(stream));
                    }
                    Err(e) => {
                        eprintln!("accept error: {e}");
                        break;
                    }
                }
            }
        });
}
