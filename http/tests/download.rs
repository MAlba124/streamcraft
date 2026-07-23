//! Milestone 2: `httpsrc ! filesink` downloads a file over HTTP and writes it to
//! disk byte-identically (spec: Milestone applications §2).
//!
//! Hermetic — no real internet. Each test stands up a `std::net::TcpListener` on
//! `127.0.0.1:0`, serves one canned response from a background thread, and points the
//! pipeline at it.

use std::io::{Read, Write};
use std::net::TcpListener;

use sc_http::HttpSrc;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSink;

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

/// Like [`serve_once`], but the server does **not** close after writing: it holds the
/// connection open until the returned [`Sender`] is dropped (or signalled). This proves
/// a length- or chunk-framed download terminates on its own framing rather than on the
/// socket closing — if the client wrongly waited for EOF, the test would hang.
///
/// The test must keep the returned guard alive until *after* `run()` returns, then drop
/// it to let the server thread close.
fn serve_once_no_close(head: &'static str, body: Vec<u8>) -> (u16, std::sync::mpsc::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(&body);
        // Do NOT drop `stream` yet — keep the connection open so the client cannot be
        // relying on EOF to terminate. Block until the test signals it is done.
        let _ = rx.recv();
        // Now the connection (and listener) drop, closing the socket.
    });
    (port, tx)
}

/// Encode `body` as an HTTP/1.1 `chunked` message body (RFC 9112 §7.1): a series of
/// `<hex-size>\r\n<data>\r\n` chunks split at `sizes`, then the `0\r\n\r\n` terminator.
fn chunked_encode(body: &[u8], sizes: &[usize]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut off = 0;
    for &n in sizes {
        let end = (off + n).min(body.len());
        let piece = &body[off..end];
        out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
        out.extend_from_slice(piece);
        out.extend_from_slice(b"\r\n");
        off = end;
    }
    // Any remaining bytes go in a final chunk, so `sizes` need not cover the whole body.
    if off < body.len() {
        let piece = &body[off..];
        out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
        out.extend_from_slice(piece);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"0\r\n\r\n"); // last-chunk + terminating CRLF
    out
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

#[test]
fn content_length_terminates_without_close() {
    // A `Content-Length` response (RFC 9112 §6.2). The server sends exactly N bytes and
    // then keeps the connection open — the client must stop after N bytes on the length
    // alone, not wait for EOF. ~300 KiB so several pooled buffers flow.
    let body: Vec<u8> = (0..300 * 1024u32).map(|i| (i.wrapping_mul(31) % 253) as u8).collect();
    let head: &'static str = Box::leak(
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_boxed_str(),
    );
    let (port, done) = serve_once_no_close(head, body.clone());

    let outp = temp_path("clen_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");
    drop(done); // let the server thread close now that the download finished

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got.len(), body.len(), "Content-Length body length matches");
    assert_eq!(got, body, "Content-Length body is byte-identical");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn chunked_transfer_encoding_is_decoded() {
    // A `Transfer-Encoding: chunked` response (RFC 9112 §6.1, §7.1) with chunks of
    // varied sizes plus the zero terminator. The decoded stream must equal the original
    // body. The server also holds the connection open, so termination comes from the
    // last-chunk, not from EOF.
    let body: Vec<u8> = (0..250 * 1024u32).map(|i| (i % 251) as u8).collect();
    // Deliberately irregular chunk sizes, including one crossing the 128 KiB pool slot;
    // `chunked_encode` puts the tail (~35 KiB) in a final chunk.
    let sizes = [1usize, 5, 200, 4096, 130 * 1024, 7, 63 * 1024, 0xFED];
    let encoded = chunked_encode(&body, &sizes);
    let (port, done) = serve_once_no_close(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        encoded,
    );

    let outp = temp_path("chunked_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");
    drop(done);

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got.len(), body.len(), "decoded chunked length matches body");
    assert_eq!(got, body, "decoded chunked bytes are byte-identical");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn chunked_with_extensions_and_trailer_is_decoded() {
    // Exercise chunk-extensions on the size line and a trailer section after the zero
    // chunk (RFC 9112 §7.1.1, §7.1.2) — both must be ignored, data decoded intact.
    let mut encoded = Vec::new();
    encoded.extend_from_slice(b"5;ext=1\r\nhello\r\n"); // chunk-ext after size
    encoded.extend_from_slice(b"6;a;b=c\r\n world\r\n"); // multiple exts
    encoded.extend_from_slice(b"0\r\n"); // last-chunk
    encoded.extend_from_slice(b"X-Trailer: value\r\n"); // a trailer field (ignored)
    encoded.extend_from_slice(b"\r\n"); // terminating CRLF

    let (port, done) = serve_once_no_close(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        encoded,
    );

    let outp = temp_path("chunked_ext_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");
    drop(done);

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got, b"hello world", "chunk-ext/trailer ignored, data intact");

    let _ = std::fs::remove_file(&outp);
}
