//! Hostile inputs to `httpsrc`: response heads that are malformed or adversarial
//! (RFC 9112 §5, §6.3; RFC 9110 §8.4), and URLs carrying injected control
//! characters (RFC 9112 §11.1/§11.2).
//!
//! Every case here is one the element used to get *quietly* wrong — a body framed
//! at the wrong length, a still-coded payload handed downstream as if it were the
//! resource, or an attacker-chosen field line put on the wire. A loud error is a
//! fine outcome; a short file and `Ok(())` is not.
//!
//! Hermetic, same shape as `download.rs`: a `std::net::TcpListener` on
//! `127.0.0.1:0` serving one canned response from a background thread.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

use pf_http::HttpSrc;
use profluens_core::error::Error;
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::FileSink;

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_hostile_{}_{}.bin", tag, std::process::id()));
    p
}

/// Read a full HTTP request head (up to and including the blank line) off a stream
/// and return it, so a test can assert on what actually went out on the wire.
fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => break,
        }
    }
    buf
}

/// Spawn a one-shot server that writes each of `head_pieces` as its own `write`
/// with a gap between them (so the client sees the header block split across
/// separate reads), then the body, then closes. The request head it read is sent
/// back on the returned channel *before* the response, so a test that completes a
/// download can `try_recv()` it without racing.
fn serve(head_pieces: &'static [&'static str], body: Vec<u8>) -> (u16, Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let _ = tx.send(read_request(&mut stream));
        for (i, piece) in head_pieces.iter().enumerate() {
            if stream.write_all(piece.as_bytes()).is_err() {
                return;
            }
            // Separate the pieces in time as well as in `write` calls: the point of
            // a multi-piece head is that the client must reassemble it across reads
            // (`serve_slow` in `download.rs` uses the same trick for bodies).
            if i + 1 < head_pieces.len() {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let _ = stream.write_all(&body);
        // Dropping `stream` closes the connection → the client sees EOF.
    });
    (port, rx)
}

/// Run `HttpSrc ! FileSink` against `url` and return the downloaded bytes, or the
/// error the run reported.
fn download(url: &str, tag: &str) -> Result<Vec<u8>, Error> {
    let outp = temp_path(tag);
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(url));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    let result = p.run();
    let got = std::fs::read(&outp).unwrap_or_default();
    let _ = std::fs::remove_file(&outp);
    result.map(|()| got)
}

/// `chunked` framing (RFC 9112 §7.1) of `body` as a single chunk.
fn one_chunk(body: &[u8]) -> Vec<u8> {
    let mut out = format!("{:x}\r\n", body.len()).into_bytes();
    out.extend_from_slice(body);
    out.extend_from_slice(b"\r\n0\r\n\r\n");
    out
}

/// A minimal gzip member (RFC 1952) holding `hello`: the 10-byte header (magic 1f 8b,
/// CM=deflate, no flags, zero MTIME, XFL=0, OS=unknown), one stored (BTYPE=00) DEFLATE
/// block, then the CRC32 + ISIZE trailer. Nothing here depends on the exact bytes —
/// only that they are visibly *not* the payload, so emitting them unchanged shows up.
fn gzip_hello() -> Vec<u8> {
    let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
    gz.extend_from_slice(&[0x01, 0x05, 0x00, 0xfa, 0xff]); // final stored block, len 5
    gz.extend_from_slice(b"hello");
    gz.extend_from_slice(&0x3610_a686u32.to_le_bytes()); // CRC32("hello")
    gz.extend_from_slice(&5u32.to_le_bytes()); // ISIZE
    gz
}

// --- bug 1: a `Content-Length` split over several field lines ----------------

#[test]
fn split_content_length_field_must_not_truncate() {
    // RFC 9110 §5.3: repeated field lines of the same name are equivalent to one
    // field whose value is the comma-joined list. So this head declares the list
    // "100, 3" — and RFC 9112 §6.3 rule 5 says a Content-Length list whose values
    // are not all identical makes the framing invalid, which "a user agent MUST
    // treat as an unrecoverable error". Reading only one of the lines instead
    // frames a 100-byte body at 3 bytes and reports success: a silent short file,
    // which is exactly the failure mode a caller cannot detect.
    let body: Vec<u8> = (0..100u32).map(|i| (i % 251) as u8).collect();
    let (port, _rx) = serve(
        &["HTTP/1.1 200 OK\r\nContent-Length: 100\r\nContent-Length: 3\r\n\r\n"],
        body.clone(),
    );

    match download(&format!("http://127.0.0.1:{port}/file"), "split_clen") {
        Err(_) => {}
        Ok(got) => panic!(
            "conflicting Content-Length (100 vs 3) accepted: downloaded {} of {} \
             bytes and reported success (RFC 9112 §6.3 rule 5)",
            got.len(),
            body.len()
        ),
    }
}

#[test]
fn folded_field_value_must_not_inject_content_length() {
    // RFC 9112 §5.2 obs-fold: a field value may be continued on the next line by
    // starting it with SP/HTAB, and "a user agent that receives an obs-fold in a
    // response message ... MUST replace each received obs-fold with one or more SP
    // octets prior to interpreting the field value". Here the continuation belongs
    // to `X-Note`; treating it as a field line of its own conjures a second
    // `Content-Length: 3` that shadows the real one and truncates the body.
    let body: Vec<u8> = (0..100u32).map(|i| (i % 251) as u8).collect();
    let (port, _rx) = serve(
        &["HTTP/1.1 200 OK\r\nContent-Length: 100\r\nX-Note: see\r\n Content-Length: 3\r\n\r\n"],
        body.clone(),
    );

    let got = download(&format!("http://127.0.0.1:{port}/file"), "fold_clen")
        .expect("an obs-fold in a value must be unfolded, leaving Content-Length: 100");
    assert_eq!(
        got.len(),
        body.len(),
        "obs-fold continuation was read as a field line and truncated the body"
    );
    assert_eq!(got, body, "unfolded response body is byte-identical");
}

#[test]
fn signed_content_length_must_not_truncate() {
    // RFC 9112 §6.2 spells the field `1*DIGIT`, but `str::parse::<u64>` also accepts a
    // leading `+`. That one octet is enough: the value reads as 5, the 100-byte body is
    // framed at 5 bytes, and the run reports success. §6.3 rule 5 again — an invalid
    // Content-Length is unrecoverable, not something to interpret generously.
    let body: Vec<u8> = (0..100u32).map(|i| (i % 251) as u8).collect();
    let (port, _rx) = serve(&["HTTP/1.1 200 OK\r\nContent-Length: +5\r\n\r\n"], body.clone());

    match download(&format!("http://127.0.0.1:{port}/file"), "signed_clen") {
        Err(_) => {}
        Ok(got) => panic!(
            "`Content-Length: +5` accepted: downloaded {} of {} bytes and reported \
             success (RFC 9112 §6.2 is 1*DIGIT)",
            got.len(),
            body.len()
        ),
    }
}

#[test]
fn header_block_split_across_reads_is_reassembled() {
    // The other way a `Content-Length` field arrives "split": across TCP reads rather
    // than across field lines, with the boundaries landing *inside* the field name, the
    // value, and the CRLF-CRLF terminator itself. `read_headers` accumulates until it
    // sees the terminator before interpreting anything, so this already held — it is
    // here to keep it that way.
    let body: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(31) % 253) as u8).collect();
    let (port, _rx) = serve(
        &[
            "HTTP/1.1 200 OK\r\nCon",
            "tent-Len",
            "gth: 4096\r",
            "\n\r",
            "\n",
        ],
        body.clone(),
    );

    let got = download(&format!("http://127.0.0.1:{port}/file"), "split_head")
        .expect("a header block split across reads must be reassembled");
    assert_eq!(got.len(), body.len(), "split header block framed the body correctly");
    assert_eq!(got, body, "body after a split header block is byte-identical");
}

// --- bug 2: a content coding we cannot decode --------------------------------

#[test]
fn content_encoding_must_not_emit_undecoded_bytes() {
    // RFC 9110 §8.4: `Content-Encoding` names codings applied to the
    // *representation* — "what decoding mechanisms have to be applied in order to
    // obtain data in the media type". `Transfer-Encoding: chunked` is stripped by
    // this element's chunk decoder, but that only removes the framing: what comes
    // out is still the gzip member, not the resource. Pushing it downstream as the
    // payload hands a demuxer DEFLATE noise where a container should be.
    let gz = gzip_hello();
    let (port, _rx) = serve(
        &["HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Encoding: gzip\r\n\r\n"],
        one_chunk(&gz),
    );

    match download(&format!("http://127.0.0.1:{port}/file"), "ce_gzip") {
        Err(_) => {}
        Ok(got) => panic!(
            "Content-Encoding: gzip passed through undecoded: got {} bytes starting \
             {:02x?} (the gzip magic) instead of the {} payload bytes, and reported \
             success",
            got.len(),
            &got[..got.len().min(4)],
            b"hello".len()
        ),
    }
}

#[test]
fn identity_content_encoding_is_accepted() {
    // RFC 9110 §8.4.1: `identity` is "no transformation" — the body is the
    // resource, so it must still download (the coding check must not be a blanket
    // "any Content-Encoding is fatal").
    let body = b"plain bytes, no coding".to_vec();
    let (port, _rx) = serve(
        &["HTTP/1.1 200 OK\r\nContent-Encoding: identity\r\nConnection: close\r\n\r\n"],
        body.clone(),
    );

    let got = download(&format!("http://127.0.0.1:{port}/file"), "ce_identity")
        .expect("Content-Encoding: identity is not a coding we need to decode");
    assert_eq!(got, body, "identity-coded body is passed through unchanged");
}

#[test]
fn transfer_coding_stack_must_not_emit_undecoded_bytes() {
    // The same silent corruption reached through the *transfer* coding instead. RFC
    // 9112 §6.1: the field "lists the transfer coding names corresponding to the
    // sequence of transfer codings that have been ... applied", and its own example is
    // this one — `gzip, chunked` means gzip first, then chunked on top.
    //
    // Which coding is final decides the framing (§6.3 rule 4), and chunked is, so
    // chunked framing is the right answer and `parse_framing` keeps giving it. But
    // undoing that outer layer leaves the gzip member: a recipient has to undo *every*
    // coding in the list, and this element implements only `chunked`. Emitting what
    // falls out is indistinguishable from the `Content-Encoding` case above.
    let gz = gzip_hello();
    let (port, _rx) = serve(
        &["HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked\r\n\r\n"],
        one_chunk(&gz),
    );

    match download(&format!("http://127.0.0.1:{port}/file"), "te_stack") {
        Err(_) => {}
        Ok(got) => panic!(
            "Transfer-Encoding: gzip, chunked de-chunked but not un-gzipped: got {} \
             bytes starting {:02x?} (the gzip magic) instead of the {} payload bytes, \
             and reported success",
            got.len(),
            &got[..got.len().min(4)],
            b"hello".len()
        ),
    }
}

#[test]
fn plain_chunked_transfer_encoding_still_downloads() {
    // The happy path, pinned next to the check that could over-reject it: `chunked`
    // alone is the one transfer coding this element undoes (RFC 9112 §7.1), so a
    // response using it must still decode to the exact body.
    let body = b"chunked but not otherwise coded".to_vec();
    let (port, _rx) = serve(
        &["HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"],
        one_chunk(&body),
    );

    let got = download(&format!("http://127.0.0.1:{port}/file"), "te_chunked")
        .expect("`chunked` alone is a coding we decode — it must not be rejected");
    assert_eq!(got, body, "plain chunked body is decoded byte-identically");
}

// --- bug 3: CRLF injection through the URL -----------------------------------

#[test]
fn url_with_crlf_must_not_inject_request_fields() {
    // RFC 9112 §11.1 (response splitting) / §11.2 (request smuggling): the request
    // target and `Host` value are formatted straight into the request head, so a CR
    // or LF reaching them ends the field line early and everything after it becomes
    // an attacker-chosen field — or a second request. URLs are data in a media
    // framework (playlists, manifests, config), so this must be rejected before a
    // socket is opened.
    let (port, rx) = serve(&["HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"], Vec::new());
    let url = format!("http://127.0.0.1:{port}/a\r\nX-Injected: evil");

    let result = download(&url, "crlf_path");

    // Whatever the run reported, the decisive evidence is the wire: if we connected
    // at all, the injected field must not be in the request we sent.
    if let Ok(req) = rx.try_recv() {
        let text = String::from_utf8_lossy(&req).into_owned();
        panic!(
            "CRLF in the URL path put an attacker-chosen field line on the wire; \
             request sent was:\n{text}"
        );
    }
    assert!(
        result.is_err(),
        "a URL containing CRLF must be rejected, not silently repaired"
    );
}

#[test]
fn url_with_crlf_in_authority_is_rejected() {
    // The same injection through the authority, which lands in the `Host` field.
    assert!(
        download("http://127.0.0.1\r\nX-Injected: evil/a", "crlf_host").is_err(),
        "a URL whose authority contains CRLF must be rejected"
    );
}
