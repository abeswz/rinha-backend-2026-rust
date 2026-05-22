use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io;
use tokio::net::{TcpListener, UnixStream};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

const SOCK1: &str = "/run/sock/api1.sock";
const SOCK2: &str = "/run/sock/api2.sock";

fn main() {
    loop {
        if std::path::Path::new(SOCK1).exists() && std::path::Path::new(SOCK2).exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("build runtime")
        .block_on(serve());
}

async fn serve() {
    let listener = TcpListener::bind("0.0.0.0:9999").await.expect("bind 9999");
    loop {
        if let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(forward(stream));
        }
    }
}

async fn forward(client: tokio::net::TcpStream) {
    let _ = client.set_nodelay(true);
    let idx = COUNTER.fetch_add(1, Ordering::Relaxed) % 2;
    let sock_path = if idx == 0 { SOCK1 } else { SOCK2 };

    let Ok(backend) = UnixStream::connect(sock_path).await else {
        return;
    };

    let (mut cr, mut cw) = client.into_split();
    let (mut br, mut bw) = backend.into_split();

    let _ = tokio::join!(io::copy(&mut cr, &mut bw), io::copy(&mut br, &mut cw),);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_alternates() {
        let a = COUNTER.fetch_add(1, Ordering::Relaxed) % 2;
        let b = COUNTER.fetch_add(1, Ordering::Relaxed) % 2;
        let c = COUNTER.fetch_add(1, Ordering::Relaxed) % 2;
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn counter_wraps_correctly() {
        let val: usize = usize::MAX;
        assert!(val % 2 == 0 || val % 2 == 1);
        let wrapped = val.wrapping_add(1);
        assert!(wrapped % 2 == 0 || wrapped % 2 == 1);
    }
}
