//! Milestone 2: `httpsrc ! filesink` downloads a file over HTTP and writes it to
//! disk byte-identically (spec: Milestone applications §2).
//!
//! Hermetic — no real internet. Each test stands up a `std::net::TcpListener` on
//! `127.0.0.1:0`, serves one canned response from a background thread, and points the
//! pipeline at it.

use std::io::{Read, Write};
use std::net::TcpListener;

use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSink;
use streamcraft_elements::net::HttpSrc;

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sc_m2_{}_{}.bin", tag, std::process::id()));
    p
}

/// Read a full HTTP request head (up to and including the blank line) off a stream.
fn read_request(stream: &mut std::net::TcpStream) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => break,
        }
    }
}

/// Spawn a one-shot HTTP server: accept a single connection, read the request, send
/// `head` immediately followed by `body`, then close. Returns the bound port.
fn serve_once(head: &'static str, body: Vec<u8>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        // Write the head, then the body. `write_all` blocks until the client drains,
        // which is exactly the backpressure path we want to exercise for a large body.
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(&body);
        // Dropping `stream` (and the listener) closes the connection → client sees EOF.
    });
    port
}

#[test]
fn downloads_file_byte_identically() {
    // ~500 KiB of non-trivial content: larger than the 128 KiB pool slot, so several
    // buffers/batches flow, and larger than a socket buffer, so the server blocks on
    // write until the pipeline reads.
    let body: Vec<u8> = (0..512 * 1024u32).map(|i| (i % 251) as u8).collect();
    let port = serve_once("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n", body.clone());

    let outp = temp_path("ok_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(
        got.len(),
        body.len(),
        "downloaded length matches served body"
    );
    assert_eq!(got, body, "downloaded bytes match served body");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn body_that_spills_into_header_read_is_preserved() {
    // Small body that the server sends in the *same* write as the headers, so it
    // arrives in the client's header read and must be replayed from `leftover`.
    let body = b"hello, streamcraft body".to_vec();
    let port = serve_once("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n", body.clone());

    let outp = temp_path("spill_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/small")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got, body, "small body read alongside headers is preserved");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn non_2xx_status_errors() {
    // A 404 response must fail the run (the error rides the bus and is returned).
    let port = serve_once(
        "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n",
        b"nope".to_vec(),
    );

    let outp = temp_path("404_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/missing")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");

    assert!(p.run().is_err(), "404 response must make run() error");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn connection_refused_errors() {
    // Bind then immediately drop the listener to obtain a port nothing is listening
    // on; connecting there should be refused and make the run error.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let outp = temp_path("refused_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");

    assert!(p.run().is_err(), "connecting to a closed port must error");

    let _ = std::fs::remove_file(&outp);
}
