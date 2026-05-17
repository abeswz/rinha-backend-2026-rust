static RESP_0: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}";
static RESP_1: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}";
static RESP_2: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}";
static RESP_3: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}";
static RESP_4: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}";
static RESP_5: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}";

static FRAUD_RESPONSES: [&[u8]; 6] = [RESP_0, RESP_1, RESP_2, RESP_3, RESP_4, RESP_5];

pub static RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK";
pub static RESP_NOT_FOUND: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
pub static RESP_BAD_REQ: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

#[inline(always)]
pub fn http_body_for(fraud_count: u8) -> &'static [u8] {
    FRAUD_RESPONSES[(fraud_count as usize).min(5)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_of(resp: &[u8]) -> &[u8] {
        let sep = b"\r\n\r\n";
        let pos = resp.windows(4).position(|w| w == sep).expect("no header separator");
        &resp[pos + 4..]
    }

    #[test]
    fn all_6_responses_valid_json() {
        for count in 0u8..=5 {
            let resp = http_body_for(count);
            let body = body_of(resp);
            let s = std::str::from_utf8(body).expect("not utf8");
            assert!(s.contains("\"approved\""), "count={count} missing approved");
            assert!(s.contains("\"fraud_score\""), "count={count} missing fraud_score");
        }
    }

    #[test]
    fn approved_flag_follows_threshold() {
        for count in 0u8..=2 {
            let body = std::str::from_utf8(body_of(http_body_for(count))).unwrap();
            assert!(body.contains("\"approved\":true"), "count={count} should be approved");
        }
        for count in 3u8..=5 {
            let body = std::str::from_utf8(body_of(http_body_for(count))).unwrap();
            assert!(body.contains("\"approved\":false"), "count={count} should be rejected");
        }
    }

    #[test]
    fn content_length_matches_body() {
        for count in 0u8..=5 {
            let resp = http_body_for(count);
            let header = std::str::from_utf8(
                &resp[..resp.windows(4).position(|w| w == b"\r\n\r\n").unwrap()]
            ).unwrap();
            let cl: usize = header.lines()
                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                .expect("no content-length")
                .split(':').nth(1).unwrap().trim().parse().unwrap();
            let body_len = body_of(resp).len();
            assert_eq!(cl, body_len, "count={count}: content-length={cl} body_len={body_len}");
        }
    }

    #[test]
    fn ready_response_is_200() {
        let s = std::str::from_utf8(RESP_READY).unwrap();
        assert!(s.starts_with("HTTP/1.1 200"));
    }
}
