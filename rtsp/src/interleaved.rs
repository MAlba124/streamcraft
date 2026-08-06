//! Interleaved binary data over the RTSP TCP connection (RFC 2326 §10.12).
//!
//! When SETUP negotiates `RTP/AVP/TCP;interleaved=n-m`, the server mixes
//! stream data into the control connection: each binary frame is
//!
//! ```text
//! '$' (0x24) | channel (u8) | length (u16, network byte order) | payload
//! ```
//!
//! — "an ASCII dollar sign (24 hexadecimal), followed by a one-byte channel
//! identifier, followed by the length of the encapsulated binary data as a
//! binary, two-byte integer in network byte order" (§10.12). "Each $ block
//! contains exactly one upper-layer protocol data unit, e.g., one RTP
//! packet." Between frames, ordinary RTSP messages (responses, or server →
//! client requests like GET_PARAMETER pings) continue to flow as text.
//!
//! [`Demux`] separates the two: feed it raw socket bytes with
//! [`Demux::push`] — partial reads are fine, state is kept — and pull
//! complete [`Item`]s with [`Demux::pop`]. It is pure (no sockets, no
//! framework types); the eventual `rtspsrc` element owns the reads.

/// Cap on an RTSP message head (start-line + headers) inside the interleaved
/// stream, so a desynced or hostile peer can't make us buffer text forever
/// while we hunt for the header terminator.
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// One demultiplexed unit from the connection.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Item {
    /// A `$` frame (§10.12): one upper-layer PDU — one RTP or RTCP packet —
    /// on `channel` (the number bound by the Transport header's
    /// `interleaved=` parameter, §12.39).
    Frame { channel: u8, payload: Vec<u8> },
    /// One complete RTSP message, start-line through body end (the body
    /// length comes from `Content-Length`; absent means no body, §4.4 via
    /// [H4.3]). Handed back raw for the RTSP layer to parse.
    Message(Vec<u8>),
}

/// Why the byte stream stopped making sense.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DemuxError {
    /// At a frame boundary the bytes begin with neither `$` nor an RTSP
    /// start-line (§10.12 admits only those two on the connection) — the
    /// stream is desynchronized and unrecoverable (a `$` inside lost text
    /// framing would misread arbitrary bytes as lengths).
    Desync,
    /// An apparent RTSP message head exceeded the 64 KiB sanity cap without
    /// its `CRLF CRLF` terminator.
    OversizedHead,
    /// An RTSP message head carried an unparsable `Content-Length`.
    BadContentLength,
}

impl std::fmt::Display for DemuxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemuxError::Desync => write!(
                f,
                "interleaved: desynchronized (not a $ frame or RTSP message)"
            ),
            DemuxError::OversizedHead => {
                write!(f, "interleaved: RTSP head exceeded {MAX_HEAD_BYTES} bytes")
            }
            DemuxError::BadContentLength => write!(f, "interleaved: unparsable Content-Length"),
        }
    }
}

impl std::error::Error for DemuxError {}

/// Incremental demuxer for the interleaved connection byte stream.
#[derive(Default)]
pub struct Demux {
    buf: Vec<u8>,
}

impl Demux {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw connection bytes. Any read size is fine — including one byte
    /// at a time; nothing is interpreted until [`pop`](Self::pop).
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pull the next complete item, `Ok(None)` when more bytes are needed.
    /// Errors are sticky in effect: the buffer is left untouched, so a
    /// desync keeps reporting until the caller tears the connection down
    /// (there is no resynchronization point in the interleaved stream).
    // Pure demuxer with no reactor `ctx`/scratch; the returned copy lands in the
    // fixed public `Item::{Frame,Message}(Vec<u8>)` API. The reactor path (arena
    // reuse) lives in the future `rtspsrc` element that owns the reads.
    #[allow(clippy::disallowed_methods)]
    pub fn pop(&mut self) -> Result<Option<Item>, DemuxError> {
        // Robustness: skip stray CRLF bytes between items (some servers pad
        // message ends; harmless, and unambiguous — neither `$` nor a token).
        let skip = self
            .buf
            .iter()
            .take_while(|&&b| b == b'\r' || b == b'\n')
            .count();
        if skip > 0 {
            self.buf.drain(..skip);
        }

        let Some(&first) = self.buf.first() else {
            return Ok(None);
        };

        if first == b'$' {
            // §10.12: '$', channel byte, u16 length (network byte order),
            // then exactly that many payload bytes, no trailing CRLF.
            if self.buf.len() < 4 {
                return Ok(None);
            }
            let channel = self.buf[1];
            let len = usize::from(u16::from_be_bytes([self.buf[2], self.buf[3]]));
            if self.buf.len() < 4 + len {
                return Ok(None);
            }
            let payload = self.buf[4..4 + len].to_vec();
            self.buf.drain(..4 + len);
            return Ok(Some(Item::Frame { channel, payload }));
        }

        // Not a frame — it must be RTSP text. An RTSP start-line always
        // carries "RTSP/" (as the version of a response's status-line §7.1,
        // or of a request-line §6.1); anything else is desync. Check as soon
        // as the first line is complete.
        if let Some(eol) = find(&self.buf, b"\r\n") {
            if !contains(&self.buf[..eol], b"RTSP/") {
                return Err(DemuxError::Desync);
            }
        } else if !self.buf[0].is_ascii_graphic() {
            // Not even a plausible start of a token/version — fail early
            // rather than buffering binary garbage to the head cap.
            return Err(DemuxError::Desync);
        }

        let Some(head_end) = find(&self.buf, b"\r\n\r\n") else {
            if self.buf.len() > MAX_HEAD_BYTES {
                return Err(DemuxError::OversizedHead);
            }
            return Ok(None);
        };

        // Body length: Content-Length if present, else zero (§4.4 leans on
        // [H4.3]; RTSP has no chunked coding).
        let head = &self.buf[..head_end];
        let body_len = match content_length(head) {
            Ok(n) => n,
            Err(()) => return Err(DemuxError::BadContentLength),
        };
        let total = head_end + 4 + body_len;
        if self.buf.len() < total {
            return Ok(None);
        }
        let msg = self.buf[..total].to_vec();
        self.buf.drain(..total);
        Ok(Some(Item::Message(msg)))
    }
}

/// Index of the first occurrence of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Whether `hay` contains `needle`.
fn contains(hay: &[u8], needle: &[u8]) -> bool {
    find(hay, needle).is_some()
}

/// Extract `Content-Length` from a raw head block (field names are
/// case-insensitive, [H4.2]). `Ok(0)` when absent; `Err` when present but
/// not a decimal number.
fn content_length(head: &[u8]) -> Result<usize, ()> {
    for line in head.split(|&b| b == b'\n') {
        let Some(colon) = line.iter().position(|&b| b == b':') else {
            continue;
        };
        if line[..colon]
            .trim_ascii()
            .eq_ignore_ascii_case(b"content-length")
        {
            let v = line[colon + 1..].trim_ascii();
            return std::str::from_utf8(v)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or(());
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(channel: u8, payload: &[u8]) -> Vec<u8> {
        let mut f = vec![b'$', channel];
        f.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        f.extend_from_slice(payload);
        f
    }

    #[test]
    fn mixed_frames_and_messages_demux_in_order() {
        let reply = b"RTSP/1.0 200 OK\r\nCSeq: 5\r\nContent-Length: 4\r\n\r\nBODY";
        let ping = b"GET_PARAMETER rtsp://h/s RTSP/1.0\r\nCSeq: 9\r\n\r\n";
        let mut wire = frame(0, &[1, 2, 3]);
        wire.extend_from_slice(reply);
        wire.extend_from_slice(&frame(1, &[9; 100]));
        wire.extend_from_slice(ping);
        wire.extend_from_slice(&frame(0, &[])); // zero-length payload is legal

        let mut d = Demux::new();
        d.push(&wire);
        assert_eq!(
            d.pop().unwrap(),
            Some(Item::Frame {
                channel: 0,
                payload: vec![1, 2, 3]
            })
        );
        assert_eq!(d.pop().unwrap(), Some(Item::Message(reply.to_vec())));
        assert_eq!(
            d.pop().unwrap(),
            Some(Item::Frame {
                channel: 1,
                payload: vec![9; 100]
            })
        );
        // A server→client request (§10.8 liveness ping) is a message too.
        assert_eq!(d.pop().unwrap(), Some(Item::Message(ping.to_vec())));
        assert_eq!(
            d.pop().unwrap(),
            Some(Item::Frame {
                channel: 0,
                payload: vec![]
            })
        );
        assert_eq!(d.pop().unwrap(), None);
    }

    #[test]
    fn partial_reads_byte_by_byte() {
        let reply = b"RTSP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok";
        let mut wire = frame(2, b"payload bytes");
        wire.extend_from_slice(reply);
        wire.extend_from_slice(&frame(3, b"x"));

        let mut d = Demux::new();
        let mut items = Vec::new();
        for &b in &wire {
            d.push(&[b]); // one byte at a time: every boundary is exercised
            while let Some(item) = d.pop().unwrap() {
                items.push(item);
            }
        }
        assert_eq!(
            items,
            vec![
                Item::Frame {
                    channel: 2,
                    payload: b"payload bytes".to_vec()
                },
                Item::Message(reply.to_vec()),
                Item::Frame {
                    channel: 3,
                    payload: b"x".to_vec()
                },
            ]
        );
    }

    #[test]
    fn interstitial_crlf_is_skipped() {
        let mut d = Demux::new();
        d.push(b"\r\n\r\n");
        d.push(&frame(0, b"a"));
        assert_eq!(
            d.pop().unwrap(),
            Some(Item::Frame {
                channel: 0,
                payload: b"a".to_vec()
            })
        );
        assert_eq!(d.pop().unwrap(), None);
    }

    #[test]
    fn desync_is_an_error() {
        // Binary junk that is not '$'.
        let mut d = Demux::new();
        d.push(&[0x80, 0x60, 0x00]);
        assert_eq!(d.pop().unwrap_err(), DemuxError::Desync);

        // Text whose first line is not an RTSP start-line.
        let mut d = Demux::new();
        d.push(b"HTTP/1.1 200 OK\r\n\r\n");
        assert_eq!(d.pop().unwrap_err(), DemuxError::Desync);
    }

    #[test]
    fn bad_content_length_is_an_error() {
        let mut d = Demux::new();
        d.push(b"RTSP/1.0 200 OK\r\nContent-Length: banana\r\n\r\n");
        assert_eq!(d.pop().unwrap_err(), DemuxError::BadContentLength);
    }

    #[test]
    fn max_length_frame_and_incomplete_frame() {
        // A u16::MAX-length frame demuxes whole...
        let payload = vec![0xABu8; usize::from(u16::MAX)];
        let mut d = Demux::new();
        d.push(&frame(7, &payload));
        assert_eq!(
            d.pop().unwrap(),
            Some(Item::Frame {
                channel: 7,
                payload
            })
        );

        // ...and one byte short of complete yields nothing yet.
        let mut d = Demux::new();
        let f = frame(7, &[1, 2, 3, 4]);
        d.push(&f[..f.len() - 1]);
        assert_eq!(d.pop().unwrap(), None);
        d.push(&f[f.len() - 1..]);
        assert!(matches!(d.pop().unwrap(), Some(Item::Frame { .. })));
    }
}
