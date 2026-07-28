//! TLS transport for [`HttpSrc`](crate::HttpSrc)'s `https://` support: **rustls**
//! riding the **graviola** crypto provider (pure Rust, from the rustls project — no
//! OpenSSL, no aws-lc).
//!
//! The split mirrors the element's IO design (spec: IO):
//! - [`connect`] runs the TLS handshake **synchronously in `start()`** — one-time
//!   setup on the freshly connected socket, exactly like the plain-HTTP connect +
//!   header read that already happen there.
//! - After `start()`, the returned [`rustls::ClientConnection`] is used **sans-IO**:
//!   the socket fd rides the reactor, and the element feeds completed `Recv`
//!   ciphertext into the state machine (`read_tls` → `process_new_packets` →
//!   `reader()`), never touching the socket itself.
//!
//! ## Trust roots
//! No root-bundle crate (`webpki-roots`/`rustls-native-certs`): the verifier is fed
//! the platform's own CA bundle, located via `SSL_CERT_FILE` (the OpenSSL
//! convention, also our test override) or the first well-known bundle path that
//! exists. The bundle's PEM (RFC 7468 textual encoding) is parsed by a hand-written
//! reader on a hand-written base64 decoder (RFC 4648 §4) — in keeping with this
//! crate's hand-rolled HTTP.

use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConnection, RootCertStore};

use streamcraft_core::error::Error;

/// Well-known system CA bundle locations, tried in order (after `SSL_CERT_FILE`).
/// The usual suspects across distro families; the first path that exists wins.
const CA_BUNDLE_PATHS: &[&str] = &[
    "/etc/ssl/certs/ca-certificates.crt", // Debian/Ubuntu/Arch/NixOS
    "/etc/pki/tls/certs/ca-bundle.crt",   // Fedora/RHEL
    "/etc/ssl/ca-bundle.pem",             // openSUSE
    "/etc/ssl/cert.pem",                  // Alpine/FreeBSD
];

/// Handshake a TLS client session over `stream` (blocking — `start()`-time setup),
/// verifying `host` against the system trust roots. Returns the established
/// [`ClientConnection`] for the element to keep as its sans-IO decrypt state.
// COLD: the whole handshake runs once per connection in `start()`; the ALPN Vec and SNI
// name string are one-time session config, not per-chunk body-path allocations.
#[allow(clippy::disallowed_methods)]
pub(crate) fn connect(
    host: &str,
    stream: &mut TcpStream,
    url: &str,
) -> Result<ClientConnection, Error> {
    let roots = load_roots(url)?;
    let mut config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls_graviola::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| Error::Resource(format!("httpsrc: TLS config for {url}: {e}")))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    // We speak exactly HTTP/1.1 on top; say so (RFC 7301) rather than negotiating
    // nothing — some deployments route on ALPN.
    config.alpn_protocols = vec![b"http/1.1".to_vec()];

    // `ServerName` (SNI + the name verified against the certificate) accepts DNS
    // names and IP addresses — `https://127.0.0.1:8443/` verifies the cert's IP SAN.
    let name = ServerName::try_from(host.to_string()).map_err(|_| {
        Error::Resource(format!(
            "httpsrc: {url}: {host:?} is not a valid TLS server name"
        ))
    })?;

    let mut conn = ClientConnection::new(Arc::new(config), name)
        .map_err(|e| Error::Resource(format!("httpsrc: TLS session for {url}: {e}")))?;
    while conn.is_handshaking() {
        conn.complete_io(stream)
            .map_err(|e| Error::Resource(format!("httpsrc: TLS handshake with {url}: {e}")))?;
    }
    Ok(conn)
}

/// Load the trust roots from the system CA bundle: `SSL_CERT_FILE` if set, else the
/// first of [`CA_BUNDLE_PATHS`] that exists. Unparseable entries in the bundle are
/// skipped (bundles routinely carry a stray oddity; losing one root must not take
/// down every download), but an empty result is an error.
// Reading the bundle is `start()`-time setup, not element streaming IO — the
// sanctioned exception to the workspace blocking-IO lint.
#[allow(clippy::disallowed_methods)]
fn load_roots(url: &str) -> Result<RootCertStore, Error> {
    let path = std::env::var_os("SSL_CERT_FILE")
        .map(PathBuf::from)
        .or_else(|| {
            CA_BUNDLE_PATHS
                .iter()
                .map(PathBuf::from)
                .find(|p| p.exists())
        })
        .ok_or_else(|| {
            Error::Resource(format!(
                "httpsrc: no CA bundle found for {url} (set SSL_CERT_FILE, or provide \
                 one of {CA_BUNDLE_PATHS:?})"
            ))
        })?;

    let pem = std::fs::read(&path).map_err(|e| {
        Error::Resource(format!(
            "httpsrc: read CA bundle {}: {e}",
            path.display()
        ))
    })?;

    let mut roots = RootCertStore::empty();
    roots.add_parsable_certificates(pem_certificates(&pem));
    if roots.is_empty() {
        return Err(Error::Resource(format!(
            "httpsrc: CA bundle {} contains no usable certificates",
            path.display()
        )));
    }
    Ok(roots)
}

/// Extract every `CERTIFICATE` block from a PEM bundle (RFC 7468 §5.1: base64 DER
/// between `-----BEGIN CERTIFICATE-----`/`-----END CERTIFICATE-----` lines).
/// Comment lines and other block types (bundles sometimes interleave plain-text
/// descriptions) are skipped; a block whose base64 fails to decode is dropped —
/// [`load_roots`] treats per-entry damage as skippable.
// COLD: parses the CA bundle once per connection at `start()` (via `load_roots`); the
// cert list and per-block accumulator are setup allocations, not the body path.
#[allow(clippy::disallowed_methods)]
fn pem_certificates(pem: &[u8]) -> Vec<CertificateDer<'static>> {
    let mut out = Vec::new();
    let mut body: Option<Vec<u8>> = None;
    for line in pem.split(|&b| b == b'\n') {
        let line = trim_ascii_line(line);
        match (line, &mut body) {
            (b"-----BEGIN CERTIFICATE-----", _) => body = Some(Vec::new()),
            (b"-----END CERTIFICATE-----", b @ Some(_)) => {
                if let Some(der) = b.take().and_then(|b64| decode_base64(&b64)) {
                    out.push(CertificateDer::from(der));
                }
            }
            (l, Some(b)) => b.extend_from_slice(l),
            _ => {} // outside any CERTIFICATE block
        }
    }
    out
}

/// Trim ASCII whitespace (notably the `\r` of CRLF line endings) from a PEM line.
fn trim_ascii_line(mut l: &[u8]) -> &[u8] {
    while let [rest @ .., last] = l {
        if last.is_ascii_whitespace() {
            l = rest;
        } else {
            break;
        }
    }
    while let [first, rest @ ..] = l {
        if first.is_ascii_whitespace() {
            l = rest;
        } else {
            break;
        }
    }
    l
}

/// The base64 alphabet (RFC 4648 §4, table 1) as a value lookup.
fn base64_value(b: u8) -> Option<u32> {
    match b {
        b'A'..=b'Z' => Some(u32::from(b - b'A')),
        b'a'..=b'z' => Some(u32::from(b - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(b - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Decode base64 (RFC 4648 §4: the standard alphabet, `=` padding) from an input
/// with no interior whitespace (PEM body lines are concatenated before the call).
/// Returns `None` on any symbol outside the alphabet or an impossible length
/// (a final group of one symbol encodes fewer than 8 bits — §4: invalid).
// COLD: decodes each CA-bundle cert once per connection at `start()`; the output DER
// buffer is a setup allocation, not the per-chunk body path.
#[allow(clippy::disallowed_methods)]
fn decode_base64(mut s: &[u8]) -> Option<Vec<u8>> {
    // Strip trailing `=` padding; the leftover symbol count encodes the tail length.
    while let [rest @ .., b'='] = s {
        s = rest;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 2);
    let mut acc = 0u32;
    let mut n = 0u32;
    for &b in s {
        acc = (acc << 6) | base64_value(b)?;
        n += 1;
        if n == 4 {
            out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]);
            acc = 0;
            n = 0;
        }
    }
    match n {
        0 => {}
        2 => out.push((acc >> 4) as u8), // 12 bits carry 1 octet
        3 => out.extend_from_slice(&[(acc >> 10) as u8, (acc >> 2) as u8]), // 18 bits, 2 octets
        _ => return None, // 1 leftover symbol (6 bits) encodes no whole octet
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{decode_base64, pem_certificates};

    #[test]
    fn base64_round_trips_known_vectors() {
        // RFC 4648 §10 test vectors.
        assert_eq!(decode_base64(b"").unwrap(), b"");
        assert_eq!(decode_base64(b"Zg==").unwrap(), b"f");
        assert_eq!(decode_base64(b"Zm8=").unwrap(), b"fo");
        assert_eq!(decode_base64(b"Zm9v").unwrap(), b"foo");
        assert_eq!(decode_base64(b"Zm9vYg==").unwrap(), b"foob");
        assert_eq!(decode_base64(b"Zm9vYmE=").unwrap(), b"fooba");
        assert_eq!(decode_base64(b"Zm9vYmFy").unwrap(), b"foobar");
    }

    #[test]
    fn base64_rejects_garbage() {
        assert!(decode_base64(b"Zm9%").is_none()); // symbol outside the alphabet
        assert!(decode_base64(b"Z").is_none()); // 1 leftover symbol: impossible length
    }

    #[test]
    fn pem_extracts_certificate_blocks() {
        // Two blocks with noise between them; the DER here is just recognizable
        // bytes, not a real certificate (parsing DER is the verifier's job).
        let pem = b"# comment\n\
            -----BEGIN CERTIFICATE-----\n\
            AAEC\n\
            -----END CERTIFICATE-----\n\
            interleaved text\n\
            -----BEGIN TRUST ANCHOR-----\n\
            ////\n\
            -----END TRUST ANCHOR-----\n\
            -----BEGIN CERTIFICATE-----\r\n\
            /v8=\r\n\
            -----END CERTIFICATE-----\r\n";
        let certs = pem_certificates(pem);
        assert_eq!(certs.len(), 2, "two CERTIFICATE blocks, other block skipped");
        assert_eq!(certs[0].as_ref(), &[0x00, 0x01, 0x02]);
        assert_eq!(certs[1].as_ref(), &[0xfe, 0xff]);
    }

    #[test]
    fn pem_drops_undecodable_block_keeps_rest() {
        let pem = b"-----BEGIN CERTIFICATE-----\n\
            not!base64\n\
            -----END CERTIFICATE-----\n\
            -----BEGIN CERTIFICATE-----\n\
            AAEC\n\
            -----END CERTIFICATE-----\n";
        let certs = pem_certificates(pem);
        assert_eq!(certs.len(), 1, "damaged block dropped, good one kept");
        assert_eq!(certs[0].as_ref(), &[0x00, 0x01, 0x02]);
    }
}
