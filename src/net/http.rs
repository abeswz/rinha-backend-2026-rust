use crate::fraud::{data, json, knn, vector};
use crate::net::response::{http_body_for, RESP_BAD_REQ, RESP_NOT_FOUND, RESP_READY};
use memchr::memmem;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

const RX_CAP: usize = 8192;

#[derive(Debug, PartialEq)]
pub enum Route {
    FraudScore,
    Ready,
    NotFound,
}

pub fn find_header_end(buf: &[u8]) -> Option<usize> {
    memmem::find(buf, b"\r\n\r\n").map(|i| i + 4)
}

pub fn parse_content_length(header_bytes: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < header_bytes.len() {
        let line_end = memchr::memchr(b'\n', &header_bytes[i..])
            .map(|e| i + e)
            .unwrap_or(header_bytes.len());
        let line = &header_bytes[i..line_end];
        if line.len() > 15 && line[..15].eq_ignore_ascii_case(b"content-length:") {
            let val = &line[15..];
            let start = val.iter().position(|&c| c != b' ').unwrap_or(val.len());
            let val = &val[start..];
            let end = val
                .iter()
                .rposition(|&c| c != b' ' && c != b'\r')
                .map(|i| i + 1)
                .unwrap_or(val.len());
            let val = &val[..end];
            return std::str::from_utf8(val).ok()?.trim().parse().ok();
        }
        i = line_end + 1;
    }
    None
}

pub fn detect_route(first_line: &[u8]) -> Route {
    if first_line.starts_with(b"POST /fraud-score") {
        Route::FraudScore
    } else if first_line.starts_with(b"GET /ready") {
        Route::Ready
    } else {
        Route::NotFound
    }
}

pub async fn serve_connection(mut stream: UnixStream) {
    let ds = data::dataset();
    let mut rx_buf: Vec<u8> = Vec::with_capacity(RX_CAP);
    let mut tx_buf: Vec<u8> = Vec::with_capacity(RX_CAP);

    loop {
        if rx_buf.len() >= RX_CAP {
            break;
        }
        match stream.read_buf(&mut rx_buf).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }

        let mut consumed = 0usize;
        loop {
            let available = &rx_buf[consumed..];
            let header_end = match find_header_end(available) {
                Some(e) => e,
                None => break,
            };

            let header_bytes = &available[..header_end];
            let first_line_end = memchr::memchr(b'\r', header_bytes).unwrap_or(header_bytes.len());
            let route = detect_route(&header_bytes[..first_line_end]);

            match route {
                Route::Ready => {
                    tx_buf.extend_from_slice(RESP_READY);
                    consumed += header_end;
                }
                Route::NotFound => {
                    tx_buf.extend_from_slice(RESP_NOT_FOUND);
                    consumed += header_end;
                }
                Route::FraudScore => {
                    let cl = match parse_content_length(header_bytes) {
                        Some(n) => n,
                        None => {
                            tx_buf.extend_from_slice(RESP_BAD_REQ);
                            consumed += header_end;
                            continue;
                        }
                    };
                    let body_end = consumed + header_end + cl;
                    if header_end + cl > RX_CAP {
                        tx_buf.extend_from_slice(RESP_BAD_REQ);
                        consumed += header_end;
                        continue;
                    }
                    if body_end > rx_buf.len() {
                        break; // need more data
                    }
                    let body = &rx_buf[consumed + header_end..body_end];
                    let resp = match json::parse(body) {
                        Some(payload) => {
                            let vec = vector::vectorize(&payload);
                            let count = knn::knn5_ivf(&vec, ds);
                            http_body_for(count)
                        }
                        None => RESP_BAD_REQ,
                    };
                    tx_buf.extend_from_slice(resp);
                    consumed = body_end;
                }
            }
        }

        if !tx_buf.is_empty() {
            if stream.write_all(&tx_buf).await.is_err() {
                break;
            }
            tx_buf.clear();
        }

        if consumed > 0 {
            if consumed < rx_buf.len() {
                let remaining = rx_buf.len() - consumed;
                rx_buf.copy_within(consumed.., 0);
                rx_buf.truncate(remaining);
            } else {
                rx_buf.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_header_end_locates_crlfcrlf() {
        let buf = b"POST /fraud-score HTTP/1.1\r\nContent-Length: 5\r\n\r\nbody!";
        assert_eq!(find_header_end(buf), Some(49));
    }

    #[test]
    fn find_header_end_returns_none_when_absent() {
        let buf = b"GET /ready HTTP/1.1\r\n";
        assert_eq!(find_header_end(buf), None);
    }

    #[test]
    fn parse_content_length_case_insensitive() {
        let header = b"POST /x HTTP/1.1\r\ncontent-length: 123\r\n\r\n";
        assert_eq!(parse_content_length(header), Some(123));
        let header2 = b"POST /x HTTP/1.1\r\nContent-Length: 456\r\n\r\n";
        assert_eq!(parse_content_length(header2), Some(456));
    }

    #[test]
    fn route_detection() {
        assert_eq!(
            detect_route(b"POST /fraud-score HTTP/1.1"),
            Route::FraudScore
        );
        assert_eq!(detect_route(b"GET /ready HTTP/1.1"), Route::Ready);
        assert_eq!(detect_route(b"GET /unknown HTTP/1.1"), Route::NotFound);
    }
}
