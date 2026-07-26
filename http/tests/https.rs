//! HTTPS: `httpsrc` over TLS — rustls on the graviola provider (see `sc_http::tls`).
//!
//! Hermetic — no real internet, no real CAs. Each test stands up a **rustls server**
//! on `127.0.0.1:0` using the self-signed fixtures in `tests/tls/` (identity A
//! trusted via `SSL_CERT_FILE`, identity B deliberately untrusted), serves one
//! canned response, and points the pipeline at `https://127.0.0.1:<port>/`.
//!
//! Beyond byte-identical downloads, the TLS-specific behavior under test:
//! - certificate verification actually happens (untrusted identity → run errors);
//! - a close-delimited body requires the peer's `close_notify` (RFC 8446 §6.1) —
//!   a bare TCP FIN is an unauthenticated truncation and must error;
//! - length/chunk-framed bodies terminate on their own framing over TLS;
//! - plaintext decrypted alongside the response head is not lost;
//! - TLS records reassemble across trickled reactor reads.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::{ServerConnection, StreamOwned};
use sc_http::HttpSrc;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSink;

/// Identity A: what `httpsrc` is told to trust (as a one-cert PEM "CA bundle").
const CERT_A_PEM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tls/cert_a.pem");
const CERT_A_DER: &[u8] = include_bytes!("tls/cert_a.der");
const KEY_A_DER: &[u8] = include_bytes!("tls/key_a.der");
/// Identity B: valid TLS material that is *not* in the trusted bundle.
const CERT_B_DER: &[u8] = include_bytes!("tls/cert_b.der");
const KEY_B_DER: &[u8] = include_bytes!("tls/key_b.der");

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sc_https_{}_{}.bin", tag, std::process::id()));
    p
}

/// Point `httpsrc`'s trust roots at identity A via `SSL_CERT_FILE` (the loader's
/// first stop). Every test sets the same value, so parallel tests cannot race to
/// different bundles.
fn trust_cert_a() {
    std::env::set_var("SSL_CERT_FILE", CERT_A_PEM);
}

/// A rustls server config for the given DER certificate + PKCS#8 key, on the same
/// graviola provider the client uses.
fn server_config(cert_der: &[u8], key_der: &[u8]) -> Arc<rustls::ServerConfig> {
    let certs = vec![CertificateDer::from(cert_der.to_vec())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_der.to_vec()));
    Arc::new(
        rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls_graviola::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .expect("test cert/key"),
    )
}

/// Read a full HTTP request head (up to and including the blank line) off a stream.
/// The first read also drives the server side of the TLS handshake.
fn read_request(stream: &mut impl Read) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => break, // e.g. a failed handshake — the client test asserts
        }
    }
}

/// Spawn a one-shot HTTPS server: accept one connection, handshake, read the
/// request, write `head` + `body`, then close. `send_close_notify` selects a proper
/// TLS shutdown (RFC 8446 §6.1) vs. a bare TCP close — the latter is how the
/// truncation test forges an unauthenticated EOF. Returns the bound port.
fn serve_tls_once(
    cfg: Arc<rustls::ServerConfig>,
    head: &'static str,
    body: Vec<u8>,
    send_close_notify: bool,
) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let conn = ServerConnection::new(cfg).expect("server session");
        let mut tls = StreamOwned::new(conn, stream);
        read_request(&mut tls);
        let _ = tls.write_all(head.as_bytes());
        let _ = tls.write_all(&body);
        if send_close_notify {
            tls.conn.send_close_notify();
        }
        let _ = tls.flush();
        // Dropping `tls` (and the listener) closes the TCP connection.
    });
    port
}

/// Like [`serve_tls_once`] but the server holds the connection open (no close of
/// any kind) until the returned guard is dropped — proving a length- or
/// chunk-framed download terminates on its own framing over TLS.
fn serve_tls_once_no_close(
    cfg: Arc<rustls::ServerConfig>,
    head: &'static str,
    body: Vec<u8>,
) -> (u16, std::sync::mpsc::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let conn = ServerConnection::new(cfg).expect("server session");
        let mut tls = StreamOwned::new(conn, stream);
        read_request(&mut tls);
        let _ = tls.write_all(head.as_bytes());
        let _ = tls.write_all(&body);
        let _ = tls.flush();
        let _ = rx.recv(); // hold the connection until the test finishes
    });
    (port, tx)
}

/// Serve `head` then dribble `body` in `piece`-sized writes with a `gap` between
/// them — each write is its own TLS record, and the client's reactor reads land at
/// arbitrary ciphertext boundaries, so records must reassemble across completions.
fn serve_tls_slow(
    cfg: Arc<rustls::ServerConfig>,
    head: &'static str,
    body: Vec<u8>,
    piece: usize,
    gap: Duration,
) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let conn = ServerConnection::new(cfg).expect("server session");
        let mut tls = StreamOwned::new(conn, stream);
        read_request(&mut tls);
        let _ = tls.write_all(head.as_bytes());
        for chunk in body.chunks(piece.max(1)) {
            if tls.write_all(chunk).is_err() {
                return;
            }
            std::thread::sleep(gap);
        }
        tls.conn.send_close_notify();
        let _ = tls.flush();
    });
    port
}

/// Encode `body` as an HTTP/1.1 `chunked` message body (RFC 9112 §7.1) — same
/// helper as the plain-HTTP tests.
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
    if off < body.len() {
        let piece = &body[off..];
        out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
        out.extend_from_slice(piece);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"0\r\n\r\n");
    out
}

/// Run `httpsrc ! filesink` against `https://127.0.0.1:<port>/file`, returning the
/// downloaded bytes (or the run error).
fn download(port: u16, tag: &str) -> Result<Vec<u8>, streamcraft_core::error::Error> {
    let outp = temp_path(tag);
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("https://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    let run = p.run();
    let got = std::fs::read(&outp).unwrap_or_default();
    let _ = std::fs::remove_file(&outp);
    run.map(|()| got)
}

#[test]
fn https_download_is_byte_identical() {
    trust_cert_a();
    // Close-delimited body, several pool slots' worth, ended by a proper TLS
    // shutdown (close_notify, then TCP close).
    let body: Vec<u8> = (0..300 * 1024u32).map(|i| (i.wrapping_mul(37) % 251) as u8).collect();
    let port = serve_tls_once(
        server_config(CERT_A_DER, KEY_A_DER),
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
        body.clone(),
        true,
    );

    let got = download(port, "eof_ok").expect("run");
    assert_eq!(got.len(), body.len(), "https download length matches");
    assert_eq!(got, body, "https download is byte-identical");
}

#[test]
fn https_body_decrypted_with_headers_is_preserved() {
    trust_cert_a();
    // Head + body in ONE write → one TLS record carries the header terminator and
    // several KiB of body. The header read consumes part of that record's
    // plaintext; the rest is still buffered in the session when `start()` finishes
    // and must be moved into the decoder's input, not lost.
    let body: Vec<u8> = (0..8000u32).map(|i| (i % 249) as u8).collect();
    let mut wire = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
    wire.extend_from_slice(&body);

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let cfg = server_config(CERT_A_DER, KEY_A_DER);
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let conn = ServerConnection::new(cfg).expect("server session");
        let mut tls = StreamOwned::new(conn, stream);
        read_request(&mut tls);
        let _ = tls.write_all(&wire); // a single write: head and body share records
        tls.conn.send_close_notify();
        let _ = tls.flush();
    });

    let got = download(port, "spill").expect("run");
    assert_eq!(got, body, "body bytes decrypted alongside headers are preserved");
}

#[test]
fn https_content_length_terminates_without_close() {
    trust_cert_a();
    let body: Vec<u8> = (0..200 * 1024u32).map(|i| (i.wrapping_mul(53) % 253) as u8).collect();
    let head: &'static str = Box::leak(
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_boxed_str(),
    );
    let (port, done) = serve_tls_once_no_close(server_config(CERT_A_DER, KEY_A_DER), head, body.clone());

    let got = download(port, "clen").expect("run");
    drop(done);
    assert_eq!(got, body, "Content-Length body over TLS is byte-identical");
}

#[test]
fn https_chunked_is_decoded() {
    trust_cert_a();
    let body: Vec<u8> = (0..150 * 1024u32).map(|i| (i % 251) as u8).collect();
    let sizes = [1usize, 5, 4096, 130 * 1024, 7, 0xFED];
    let encoded = chunked_encode(&body, &sizes);
    let (port, done) = serve_tls_once_no_close(
        server_config(CERT_A_DER, KEY_A_DER),
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        encoded,
    );

    let got = download(port, "chunked").expect("run");
    drop(done);
    assert_eq!(got, body, "chunked body over TLS decodes byte-identically");
}

#[test]
fn https_slow_delivery_reassembles_records() {
    trust_cert_a();
    // 137-byte writes → hundreds of small TLS records; reactor completions carry
    // arbitrary slices of ciphertext, so records span completions and partial
    // records sit in the deframer between passes.
    let body: Vec<u8> = (0..48 * 1024u32).map(|i| (i.wrapping_mul(97) % 251) as u8).collect();
    let head: &'static str = Box::leak(
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_boxed_str(),
    );
    let port = serve_tls_slow(
        server_config(CERT_A_DER, KEY_A_DER),
        head,
        body.clone(),
        137,
        Duration::from_micros(200),
    );

    let got = download(port, "slow").expect("run");
    assert_eq!(got, body, "trickled TLS records reassemble byte-identically");
}

#[test]
fn https_truncation_without_close_notify_errors() {
    trust_cert_a();
    // Close-delimited body ended by a bare TCP FIN — no close_notify. That EOF is
    // unauthenticated (an on-path attacker can forge it to truncate the download),
    // so the run must error (RFC 8446 §6.1), not accept a silently short body.
    let body: Vec<u8> = (0..64 * 1024u32).map(|i| (i % 251) as u8).collect();
    let port = serve_tls_once(
        server_config(CERT_A_DER, KEY_A_DER),
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
        body,
        false, // no close_notify: just drop the socket
    );

    assert!(
        download(port, "trunc").is_err(),
        "close-delimited body without close_notify must error, not truncate"
    );
}

#[test]
fn https_untrusted_certificate_errors() {
    trust_cert_a();
    // Identity B is valid TLS material but not in the trusted bundle — the
    // handshake must fail verification and the run must error. This is the test
    // that proves verification actually happens.
    let body = b"should never arrive".to_vec();
    let port = serve_tls_once(
        server_config(CERT_B_DER, KEY_B_DER),
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
        body,
        true,
    );

    assert!(
        download(port, "untrusted").is_err(),
        "a certificate outside the trust roots must fail the run"
    );
}
