use std::os::unix::io::RawFd;

fn create_tcp_listener(port: u16) -> RawFd {
    unsafe {
        let fd = libc::socket(
            libc::AF_INET,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        );
        assert!(fd >= 0, "socket failed");
        libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, &1i32 as *const _ as _, 4);
        libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT, &1i32 as *const _ as _, 4);
        libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_DEFER_ACCEPT, &1i32 as *const _ as _, 4);

        let mut addr: libc::sockaddr_in = std::mem::zeroed();
        addr.sin_family = libc::AF_INET as u16;
        addr.sin_port = port.to_be();
        addr.sin_addr.s_addr = 0; // 0.0.0.0

        let rc = libc::bind(fd, &addr as *const _ as *const libc::sockaddr, std::mem::size_of_val(&addr) as u32);
        assert!(rc == 0, "bind failed: {}", *libc::__errno_location());
        libc::listen(fd, 8192);
        fd
    }
}

fn create_uds_sender() -> RawFd {
    unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0);
        assert!(fd >= 0, "uds socket failed");
        let sndbuf = 16 * 1024 * 1024i32;
        libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_SNDBUF, &sndbuf as *const _ as _, 4);
        fd
    }
}

fn send_fds_via_scm(sender_fd: RawFd, dest_path: &str, fds: &[RawFd]) {
    unsafe {
        let rights_len = fds.len() * 4;
        let cmsg_space = libc::CMSG_SPACE(rights_len as u32) as usize;
        let mut cmsg_buf = vec![0u8; cmsg_space];

        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as u16;
        let path_bytes = dest_path.as_bytes();
        std::ptr::copy_nonoverlapping(
            path_bytes.as_ptr() as *const libc::c_char,
            addr.sun_path.as_mut_ptr(),
            path_bytes.len().min(addr.sun_path.len() - 1),
        );

        let mut payload = 0u8;
        let mut iov = libc::iovec {
            iov_base: &mut payload as *mut _ as *mut libc::c_void,
            iov_len: 1,
        };
        let mhdr = libc::msghdr {
            msg_name: &addr as *const _ as *mut libc::c_void,
            msg_namelen: std::mem::size_of_val(&addr) as u32,
            msg_iov: &mut iov,
            msg_iovlen: 1,
            msg_control: cmsg_buf.as_mut_ptr() as *mut libc::c_void,
            msg_controllen: cmsg_space,
            msg_flags: 0,
        };

        let cmsg = libc::CMSG_FIRSTHDR(&mhdr) as *mut libc::cmsghdr;
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(rights_len as u32) as usize;
        let data_ptr = libc::CMSG_DATA(cmsg) as *mut libc::c_int;
        for (i, &fd) in fds.iter().enumerate() {
            *data_ptr.add(i) = fd;
        }

        libc::sendmsg(sender_fd, &mhdr, 0);
    }
}

fn main() {
    unsafe {
        libc::prctl(libc::PR_SET_TIMERSLACK, 1usize, 0usize, 0usize, 0usize);
    }

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999);

    let upstreams_raw = std::env::var("UPSTREAMS")
        .unwrap_or_else(|_| panic!("UPSTREAMS env var required"));
    let upstreams: Vec<String> = upstreams_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert!(!upstreams.is_empty(), "UPSTREAMS must contain at least one path");

    let listen_fd = create_tcp_listener(port);
    let sender_fd = create_uds_sender();

    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    assert!(epfd >= 0);
    unsafe {
        let ev = libc::epoll_event { events: libc::EPOLLIN as u32, u64: listen_fd as u64 };
        libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, listen_fd, &ev as *const _ as *mut _);
    }

    let n_up = upstreams.len();
    let mut batches: Vec<Vec<RawFd>> = vec![Vec::with_capacity(64); n_up];
    let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; 1024];
    let mut rr: usize = 0;

    eprintln!("lb started: port={port} upstreams={upstreams:?}");

    loop {
        let n = unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), events.len() as i32, -1) };
        if n <= 0 {
            if n < 0 {
                let e = unsafe { *libc::__errno_location() };
                if e == libc::EINTR { continue; }
                panic!("epoll_wait: errno={e}");
            }
            continue;
        }

        for b in &mut batches { b.clear(); }

        for ev in &events[..n as usize] {
            if ev.u64 as RawFd != listen_fd { continue; }
            loop {
                let cfd = unsafe {
                    libc::accept4(listen_fd, std::ptr::null_mut(), std::ptr::null_mut(),
                                  libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC)
                };
                if cfd < 0 { break; }
                unsafe {
                    libc::setsockopt(cfd, libc::IPPROTO_TCP, libc::TCP_NODELAY, &1i32 as *const _ as _, 4);
                }
                batches[rr % n_up].push(cfd);
                rr += 1;
            }
        }

        for (i, batch) in batches.iter().enumerate() {
            if batch.is_empty() { continue; }
            for chunk in batch.chunks(16) {
                send_fds_via_scm(sender_fd, &upstreams[i], chunk);
                for &fd in chunk {
                    unsafe { libc::close(fd); }
                }
            }
        }
    }
}
