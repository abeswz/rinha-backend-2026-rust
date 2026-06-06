mod env;
mod fraud;

use memchr::memmem;
use std::os::unix::io::RawFd;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ── Pre-rendered HTTP responses ───────────────────────────────────────────────

static RESP_0: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}";
static RESP_1: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}";
static RESP_2: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}";
static RESP_3: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}";
static RESP_4: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}";
static RESP_5: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}";
static FRAUD_RESPONSES: [&[u8]; 6] = [RESP_0, RESP_1, RESP_2, RESP_3, RESP_4, RESP_5];
static RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
static RESP_BAD: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";

#[inline(always)]
fn fraud_response(count: u8) -> &'static [u8] {
    FRAUD_RESPONSES[(count as usize).min(5)]
}

// ── Connection state ──────────────────────────────────────────────────────────

const BUF_CAP: usize = 4096;

struct ConnState {
    buf: [u8; BUF_CAP],
    pos: usize,
}

impl ConnState {
    const ZERO: Self = Self { buf: [0u8; BUF_CAP], pos: 0 };
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

fn find_header_end(buf: &[u8]) -> Option<usize> {
    memmem::find(buf, b"\r\n\r\n").map(|i| i + 4)
}

fn parse_content_length(headers: &[u8]) -> Option<usize> {
    if let Some(scan_len) = headers.len().checked_sub(4) {
        let scan = &headers[..scan_len];
        if let Some(last_nl) = memchr::memrchr(b'\n', scan) {
            let last_line = &headers[last_nl + 1..];
            if last_line.len() > 16 && last_line[..16].eq_ignore_ascii_case(b"content-length: ") {
                if let Some(n) = parse_digits(&last_line[16..]) {
                    return Some(n);
                }
            }
        }
    }
    parse_content_length_slow(headers)
}

fn parse_digits(b: &[u8]) -> Option<usize> {
    let mut n = 0usize;
    for &c in b {
        if !c.is_ascii_digit() { break; }
        n = n.checked_mul(10)?.checked_add((c - b'0') as usize)?;
    }
    Some(n).filter(|&x| x > 0)
}

fn parse_content_length_slow(header_bytes: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < header_bytes.len() {
        let line_end = memchr::memchr(b'\n', &header_bytes[i..])
            .map(|e| i + e).unwrap_or(header_bytes.len());
        let line = &header_bytes[i..line_end];
        if line.len() > 15 && line[..15].eq_ignore_ascii_case(b"content-length:") {
            let val = &line[15..];
            let start = val.iter().position(|&c| c != b' ').unwrap_or(val.len());
            let val = &val[start..];
            let end = val.iter().rposition(|&c| c != b' ' && c != b'\r')
                .map(|i| i + 1).unwrap_or(val.len());
            return std::str::from_utf8(&val[..end]).ok()?.trim().parse().ok();
        }
        i = line_end + 1;
    }
    None
}

// ── Per-fd handler ────────────────────────────────────────────────────────────

fn handle_client(fd: RawFd, state: &mut ConnState) -> bool {
    let n = unsafe {
        libc::read(
            fd,
            state.buf.as_mut_ptr().add(state.pos) as *mut libc::c_void,
            BUF_CAP - state.pos,
        )
    };
    if n <= 0 {
        return false;
    }
    state.pos += n as usize;

    let mut consumed = 0usize;
    loop {
        let available = &state.buf[consumed..state.pos];
        let header_end = match find_header_end(available) {
            Some(e) => e,
            None => break,
        };
        let header_bytes = &available[..header_end];
        let first_line_end = memchr::memchr(b'\r', header_bytes).unwrap_or(header_bytes.len());
        let first_line = &header_bytes[..first_line_end];

        let response: &[u8] = if first_line.starts_with(b"GET /ready") {
            consumed += header_end;
            RESP_READY
        } else if first_line.starts_with(b"POST /fraud-score") {
            let cl = match parse_content_length(header_bytes) {
                Some(n) => n,
                None => {
                    consumed += header_end;
                    let mut written = 0usize;
                    while written < RESP_BAD.len() {
                        let n = unsafe {
                            libc::write(
                                fd,
                                RESP_BAD.as_ptr().add(written) as *const libc::c_void,
                                RESP_BAD.len() - written,
                            )
                        };
                        if n <= 0 { return false; }
                        written += n as usize;
                    }
                    continue;
                }
            };
            let body_end = consumed + header_end + cl;
            if body_end > state.pos {
                break;
            }
            let body = &state.buf[consumed + header_end..body_end];
            consumed = body_end;
            match fraud::json::parse(body) {
                Some(payload) => {
                    let q = fraud::vector::vectorize(&payload);
                    let tag = fraud::vector::tag_from_request(&payload);
                    let count = fraud::search::search_with_fallback(tag, &q);
                    fraud_response(count)
                }
                None => RESP_BAD,
            }
        } else {
            consumed += header_end;
            RESP_BAD
        };

        let mut written = 0usize;
        while written < response.len() {
            let n = unsafe {
                libc::write(
                    fd,
                    response.as_ptr().add(written) as *const libc::c_void,
                    response.len() - written,
                )
            };
            if n <= 0 { return false; }
            written += n as usize;
        }
    }

    if consumed > 0 {
        let remaining = state.pos - consumed;
        if remaining > 0 {
            state.buf.copy_within(consumed..state.pos, 0);
        }
        state.pos = remaining;
    }
    true
}

// ── SCM_RIGHTS receiver ───────────────────────────────────────────────────────

fn recv_fds(listen_fd: RawFd, epfd: RawFd, states: &mut Vec<ConnState>) {
    const MAX_FDS_PER_MSG: usize = 16;
    let cmsg_buf_size = unsafe { libc::CMSG_SPACE((MAX_FDS_PER_MSG * 4) as u32) } as usize;
    let mut cmsg_buf = vec![0u8; cmsg_buf_size];
    let mut iov_buf = [0u8; 1];

    loop {
        let mut iov = libc::iovec {
            iov_base: iov_buf.as_mut_ptr() as *mut libc::c_void,
            iov_len: 1,
        };
        let mut mhdr = libc::msghdr {
            msg_name: std::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: &mut iov,
            msg_iovlen: 1,
            msg_control: cmsg_buf.as_mut_ptr() as *mut libc::c_void,
            msg_controllen: cmsg_buf_size,
            msg_flags: 0,
        };

        let n = unsafe { libc::recvmsg(listen_fd, &mut mhdr, libc::MSG_DONTWAIT) };
        if n < 0 {
            break;
        }

        let mut cmsg = unsafe { libc::CMSG_FIRSTHDR(&mhdr) };
        while !cmsg.is_null() {
            let cmsg_ref = unsafe { &*cmsg };
            if cmsg_ref.cmsg_level == libc::SOL_SOCKET && cmsg_ref.cmsg_type == libc::SCM_RIGHTS {
                let data_ptr = unsafe { libc::CMSG_DATA(cmsg) } as *const libc::c_int;
                let n_fds = (cmsg_ref.cmsg_len as usize - unsafe { libc::CMSG_LEN(0) } as usize) / 4;
                for i in 0..n_fds {
                    let cfd = unsafe { *data_ptr.add(i) };
                    unsafe {
                        libc::setsockopt(
                            cfd, libc::IPPROTO_TCP, libc::TCP_QUICKACK,
                            &1i32 as *const _ as *const libc::c_void, 4,
                        );
                    }
                    if cfd as usize >= states.len() {
                        let new_len = ((cfd as usize + 1).next_power_of_two()).max(1024);
                        states.resize_with(new_len, || ConnState::ZERO);
                    }
                    states[cfd as usize].pos = 0;
                    let ev = libc::epoll_event {
                        events: (libc::EPOLLIN | libc::EPOLLRDHUP) as u32,
                        u64: cfd as u64,
                    };
                    unsafe {
                        libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, cfd, &ev as *const _ as *mut _);
                    }
                }
            }
            cmsg = unsafe { libc::CMSG_NXTHDR(&mhdr, cmsg) };
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    unsafe {
        libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
        libc::prctl(libc::PR_SET_TIMERSLACK, 1usize, 0usize, 0usize, 0usize);
        let param = libc::sched_param { sched_priority: 10 };
        libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
    }

    fraud::data::init_indices();

    let sock_path = env::sock_path();
    let _ = std::fs::remove_file(&sock_path);

    let listen_fd = unsafe {
        let fd = libc::socket(
            libc::AF_UNIX,
            libc::SOCK_DGRAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        );
        assert!(fd >= 0, "socket() failed");

        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as u16;
        let path_bytes = sock_path.as_bytes();
        assert!(path_bytes.len() < addr.sun_path.len(), "sock path too long");
        std::ptr::copy_nonoverlapping(
            path_bytes.as_ptr() as *const libc::c_char,
            addr.sun_path.as_mut_ptr(),
            path_bytes.len(),
        );
        let rc = libc::bind(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of_val(&addr) as u32,
        );
        assert!(rc == 0, "bind() failed: {}", *libc::__errno_location());
        libc::chmod(sock_path.as_ptr() as *const libc::c_char, 0o666);
        fd
    };

    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    assert!(epfd >= 0, "epoll_create1 failed");

    unsafe {
        let ev = libc::epoll_event {
            events: libc::EPOLLIN as u32,
            u64: listen_fd as u64,
        };
        libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, listen_fd, &ev as *const _ as *mut _);
    }

    let mut states: Vec<ConnState> = Vec::with_capacity(1024);
    states.resize_with(1024, || ConnState::ZERO);
    let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; 256];

    loop {
        let n = unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), events.len() as i32, 1) };
        if n <= 0 { continue; }

        for ev in &events[..n as usize] {
            let fd = ev.u64 as RawFd;
            if fd == listen_fd {
                recv_fds(listen_fd, epfd, &mut states);
            } else {
                let keep = handle_client(fd, &mut states[fd as usize]);
                if !keep {
                    unsafe {
                        libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut());
                        libc::close(fd);
                    }
                    states[fd as usize].pos = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_header_end_basic() {
        let buf = b"POST /fraud-score HTTP/1.1\r\nContent-Length: 5\r\n\r\nbody!";
        assert_eq!(find_header_end(buf), Some(49));
    }

    #[test]
    fn parse_content_length_last_header() {
        let header = b"POST /x HTTP/1.1\r\nContent-Length: 123\r\n\r\n";
        assert_eq!(parse_content_length(header), Some(123));
    }

    #[test]
    fn parse_content_length_case_insensitive() {
        let header = b"POST /x HTTP/1.1\r\ncontent-length: 456\r\n\r\n";
        assert_eq!(parse_content_length(header), Some(456));
    }

    #[test]
    fn parse_content_length_not_last() {
        let header = b"POST /x HTTP/1.1\r\nContent-Length: 42\r\nX-Custom: foo\r\n\r\n";
        assert_eq!(parse_content_length(header), Some(42));
    }

    #[test]
    fn fraud_response_mapping() {
        for i in 0u8..=5 {
            let r = fraud_response(i);
            let body = &r[r.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4..];
            let s = std::str::from_utf8(body).unwrap();
            assert!(s.contains("fraud_score"), "count={i}");
        }
    }
}
