//! SDP (RFC 8866) — the subset an RTSP client consumes from a DESCRIBE body.
//!
//! An SDP description is `<type>=<value>` lines (§5: `<type>` is exactly one
//! case-significant character, no whitespace around the `=`): a session-level
//! section starting at `v=`, then zero or more media sections each starting
//! at `m=` and running to the next `m=` or the end. [`parse`] keeps what an
//! RTSP receiver acts on — origin, session name, connection, per-media
//! `rtpmap` (§6.6) / `fmtp` (§6.15) / direction (§6.7) — plus the RTSP
//! `a=control` attribute (RFC 2326 §C.1.1) at both levels, and skips
//! everything else:
//!
//! - unknown/unhandled **type letters** are ignored (§5 allows a parser to
//!   "completely ignore or reject" descriptions with unknown types; a client
//!   consuming server-authored SDP tolerates them),
//! - unknown **attributes** are ignored (§5: "An SDP parser MUST ignore any
//!   attribute it doesn't understand"),
//! - while structurally malformed lines (no `=` as the second character,
//!   an unparsable `m=` line or `a=rtpmap`) are rejected loudly — §5 chose
//!   strict order/format precisely so damaged descriptions fail visibly.
//!
//! Also here: [`decode_base64`]/[`encode_base64`] (RFC 4648 §4) — H.264's
//! `sprop-parameter-sets` fmtp parameter (RFC 6184 §8.1) carries SPS/PPS as
//! base64, and RFC 2617 §2 Basic credentials are base64 too, so the client
//! module borrows the encoder.

// COLD: SDP is parsed from a DESCRIBE body and base64 (SPS/PPS, Basic creds)
// is coded per control message — the control plane, never the per-packet media
// path; every allocation here is one-time per stream setup.
#![allow(clippy::disallowed_methods)]

use std::collections::HashMap;
use std::fmt;

/// Why a description was rejected.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SdpError {
    /// The description does not begin with `v=0` (RFC 8866 §5.1: this
    /// document defines version 0, and `v=` must come first per §5's fixed
    /// line order).
    Version,
    /// A line is not `<type>=<value>` with a one-character type (§5).
    Line(String),
    /// An `m=` line with missing subfields or an unparsable port (§5.14:
    /// `m=<media> <port>[/<n>] <proto> <fmt> ...`, at least one fmt).
    Media(String),
    /// An `a=rtpmap:` value not matching §6.6's
    /// `<pt> <encoding>/<clock>[/<params>]`.
    Rtpmap(String),
    /// An `a=fmtp:` value not matching §6.15's `<fmt> <params>`.
    Fmtp(String),
}

impl fmt::Display for SdpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SdpError::Version => write!(f, "sdp: missing or unsupported v= version line"),
            SdpError::Line(l) => write!(f, "sdp: malformed line {l:?}"),
            SdpError::Media(l) => write!(f, "sdp: malformed m= line {l:?}"),
            SdpError::Rtpmap(l) => write!(f, "sdp: malformed a=rtpmap {l:?}"),
            SdpError::Fmtp(l) => write!(f, "sdp: malformed a=fmtp {l:?}"),
        }
    }
}

impl std::error::Error for SdpError {}

/// A parsed session description (the fields an RTSP client acts on).
#[derive(Clone, Debug)]
pub struct Sdp {
    /// The `o=` value verbatim (§5.2: originator, session id/version,
    /// address) — kept raw; RTSP clients only ever log it.
    pub origin: String,
    /// The `s=` session name (§5.3).
    pub session_name: String,
    /// The session-level `c=` value (§5.7), e.g. `IN IP4 0.0.0.0`. For RTSP
    /// unicast the SETUP Transport response is authoritative instead
    /// (RFC 2326 §12.39), so this stays raw.
    pub connection: Option<String>,
    /// Session-level `a=control` — the aggregate control URL
    /// (RFC 2326 §C.1.1: "If found at the session level, the attribute
    /// indicates the URL for aggregate control").
    pub control: Option<String>,
    /// Session-level direction attribute (§6.7) — the default for media
    /// sections that don't carry their own.
    pub direction: Option<Direction>,
    /// The media sections, in order of appearance.
    pub media: Vec<SdpMedia>,
}

/// The `<media>` subfield of an `m=` line (§5.14 defines "audio", "video",
/// "text", "application", "message"; the list is open-ended).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MediaKind {
    Audio,
    Video,
    /// Any other media type token, kept verbatim.
    Other(String),
}

/// A media direction attribute (§6.7).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    SendRecv,
    SendOnly,
    RecvOnly,
    Inactive,
}

/// One `a=rtpmap` entry (§6.6): dynamic payload type → payload format.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RtpMap {
    /// The encoding name (a media subtype token, e.g. `H264`, `opus`).
    pub encoding: String,
    /// The RTP clock rate in Hz.
    pub clock_rate: u32,
    /// Optional `/<encoding params>` — for audio, the channel count (§6.6).
    pub params: Option<String>,
}

/// One `a=fmtp` entry (§6.15): format-specific parameters, passed by SDP
/// "unchanged to the media tool".
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Fmtp {
    /// The parameter string verbatim (everything after `<fmt> `).
    pub raw: String,
    /// The conventional `key=value;key=value` reading of `raw` (§6.15's own
    /// example uses it; RFC 6184 §8.1 and RFC 7587 §7 both follow it).
    /// Values may contain commas (e.g. `sprop-parameter-sets=<sps>,<pps>`);
    /// parameters without an `=` become a key with an empty value. Purely
    /// advisory — formats with a different syntax still have `raw`.
    pub params: HashMap<String, String>,
}

/// One media section (§5.14 + its attribute lines).
#[derive(Clone, Debug)]
pub struct SdpMedia {
    /// The `<media>` type subfield.
    pub kind: MediaKind,
    /// The transport port. For hierarchical `<port>/<n>` forms the base port
    /// is kept (§5.14). RTSP servers conventionally send 0 here — the real
    /// ports come from SETUP (RFC 2326 §12.39).
    pub port: u16,
    /// The `<proto>` subfield verbatim, e.g. `RTP/AVP`.
    pub proto: String,
    /// The `<fmt>` list parsed as RTP payload types. Non-numeric fmt tokens
    /// (non-RTP protos) are skipped — this crate only consumes RTP media.
    pub payload_types: Vec<u8>,
    /// `a=rtpmap` entries by payload type (§6.6: up to one per format).
    pub rtpmap: HashMap<u8, RtpMap>,
    /// `a=fmtp` entries by payload type (§6.15: at most one per format).
    pub fmtp: HashMap<u8, Fmtp>,
    /// Media-level `a=control` (RFC 2326 §C.1.1: the URL controlling this
    /// stream), absolute or relative to the presentation base.
    pub control: Option<String>,
    /// The effective direction: the media-level attribute, else the
    /// session-level one (§6.7's inheritance), else `None` (§6.7 says
    /// assume `sendrecv` then; the caller decides).
    pub direction: Option<Direction>,
}

/// Parse a session description (RFC 8866 §5). Lines end with CRLF, but LF
/// alone is tolerated (§5: "parsers SHOULD be tolerant and also accept lines
/// terminated with a single newline character") — `str::lines` handles both.
pub fn parse(text: &str) -> Result<Sdp, SdpError> {
    let mut sdp = Sdp {
        origin: String::new(),
        session_name: String::new(),
        connection: None,
        control: None,
        direction: None,
        media: Vec::new(),
    };

    let mut lines = text.lines().filter(|l| !l.is_empty());
    // §5's fixed order: the description starts with `v=`; §5.1: only
    // version 0 exists.
    if lines.next() != Some("v=0") {
        return Err(SdpError::Version);
    }

    // The media section being filled; session-level lines land on `sdp`
    // directly until the first `m=`.
    let mut media: Option<SdpMedia> = None;

    for line in lines {
        // §5: `<type>=<value>`, type exactly one character, no whitespace
        // around the `=`.
        let value = match line.as_bytes() {
            [_, b'=', ..] => &line[2..],
            _ => return Err(SdpError::Line(line.to_string())),
        };
        let ty = line.as_bytes()[0];

        if ty == b'm' {
            // A new media section begins (§5: each runs to the next m= or
            // the end); finish the previous one.
            if let Some(done) = media.replace(parse_media_line(value, sdp.direction)?) {
                sdp.media.push(done);
            }
            continue;
        }

        match &mut media {
            // Session-level section.
            None => match ty {
                b'o' => sdp.origin = value.to_string(),
                b's' => sdp.session_name = value.to_string(),
                b'c' => sdp.connection = Some(value.to_string()),
                b'a' => match parse_attr(value) {
                    ("control", Some(url)) => sdp.control = Some(url.to_string()),
                    (name, None) => {
                        if let Some(d) = parse_direction(name) {
                            sdp.direction = Some(d);
                        }
                    }
                    _ => {} // §5: ignore attributes we don't understand
                },
                // Known-but-unhandled (i/u/e/p/b/t/r/z/k) and unknown type
                // letters alike are skipped — see the module docs.
                _ => {}
            },
            // Media-level section: only `a=` lines carry anything this
            // client acts on (i/c/b/k are tolerated and skipped).
            Some(m) if ty == b'a' => match parse_attr(value) {
                ("control", Some(url)) => m.control = Some(url.to_string()),
                ("rtpmap", Some(v)) => {
                    let (pt, map) = parse_rtpmap(v)?;
                    m.rtpmap.insert(pt, map);
                }
                ("fmtp", Some(v)) => {
                    let (pt, fmtp) = parse_fmtp(v)?;
                    m.fmtp.insert(pt, fmtp);
                }
                (name, None) => {
                    if let Some(d) = parse_direction(name) {
                        m.direction = Some(d); // §6.7: media overrides session
                    }
                }
                _ => {}
            },
            Some(_) => {}
        }
    }

    if let Some(done) = media {
        sdp.media.push(done);
    }
    Ok(sdp)
}

/// Split an attribute value (§5.13): `a=<name>` (property form) or
/// `a=<name>:<value>`.
fn parse_attr(value: &str) -> (&str, Option<&str>) {
    match value.split_once(':') {
        Some((name, v)) => (name, Some(v)),
        None => (value, None),
    }
}

/// Map a property attribute name to a direction (§6.7).
fn parse_direction(name: &str) -> Option<Direction> {
    Some(match name {
        "sendrecv" => Direction::SendRecv,
        "sendonly" => Direction::SendOnly,
        "recvonly" => Direction::RecvOnly,
        "inactive" => Direction::Inactive,
        _ => return None,
    })
}

/// Parse an `m=` line value (§5.14): `<media> <port>[/<n>] <proto> <fmt>...`
/// (the grammar requires at least one fmt). `session_direction` seeds the
/// §6.7 inheritance.
fn parse_media_line(
    value: &str,
    session_direction: Option<Direction>,
) -> Result<SdpMedia, SdpError> {
    let mut fields = value.split_whitespace();
    let (Some(kind), Some(port), Some(proto), first_fmt @ Some(_)) =
        (fields.next(), fields.next(), fields.next(), fields.next())
    else {
        return Err(SdpError::Media(value.to_string()));
    };

    // §5.14: hierarchical encodings use `<port>/<number of ports>`; the base
    // port is what matters to us.
    let port = port
        .split('/')
        .next()
        .unwrap_or("")
        .parse::<u16>()
        .map_err(|_| SdpError::Media(value.to_string()))?;

    // §5.14 <fmt>: for RTP protos these are payload type numbers (0..=127,
    // §6.6); non-numeric tokens (non-RTP protos like `UDP WB`) are skipped.
    let payload_types = first_fmt
        .into_iter()
        .chain(fields)
        .filter_map(|f| f.parse::<u8>().ok())
        .filter(|&pt| pt <= 127)
        .collect();

    Ok(SdpMedia {
        kind: match kind {
            "audio" => MediaKind::Audio,
            "video" => MediaKind::Video,
            other => MediaKind::Other(other.to_string()),
        },
        port,
        proto: proto.to_string(),
        payload_types,
        rtpmap: HashMap::new(),
        fmtp: HashMap::new(),
        control: None,
        direction: session_direction,
    })
}

/// Parse an `a=rtpmap` value (§6.6):
/// `<payload type> <encoding name>/<clock rate>[/<encoding params>]`.
fn parse_rtpmap(v: &str) -> Result<(u8, RtpMap), SdpError> {
    let err = || SdpError::Rtpmap(v.to_string());
    let (pt, rest) = v.split_once(' ').ok_or_else(err)?;
    let pt = pt.parse::<u8>().map_err(|_| err())?;
    let mut parts = rest.splitn(3, '/');
    let encoding = parts.next().filter(|e| !e.is_empty()).ok_or_else(err)?;
    let clock_rate = parts
        .next()
        .and_then(|c| c.parse::<u32>().ok())
        .ok_or_else(err)?;
    Ok((
        pt,
        RtpMap {
            encoding: encoding.to_string(),
            clock_rate,
            params: parts.next().map(str::to_string),
        },
    ))
}

/// Parse an `a=fmtp` value (§6.15): `<fmt> <format specific parameters>`.
/// The parameters are kept raw and additionally read as the conventional
/// `key=value;...` list (see [`Fmtp::params`]).
fn parse_fmtp(v: &str) -> Result<(u8, Fmtp), SdpError> {
    let err = || SdpError::Fmtp(v.to_string());
    let (pt, rest) = v.split_once(' ').ok_or_else(err)?;
    let pt = pt.parse::<u8>().map_err(|_| err())?;
    let mut params = HashMap::new();
    for item in rest.split(';') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        match item.split_once('=') {
            Some((k, val)) => params.insert(k.trim().to_string(), val.to_string()),
            None => params.insert(item.to_string(), String::new()),
        };
    }
    Ok((
        pt,
        Fmtp {
            raw: rest.to_string(),
            params,
        },
    ))
}

// --- base64 (RFC 4648 §4) ---------------------------------------------------

/// The standard alphabet value of one base64 byte (RFC 4648 §4, Table 1:
/// `A`–`Z` = 0–25, `a`–`z` = 26–51, `0`–`9` = 52–61, `+` = 62, `/` = 63).
fn b64_val(b: u8) -> Option<u32> {
    Some(match b {
        b'A'..=b'Z' => u32::from(b - b'A'),
        b'a'..=b'z' => u32::from(b - b'a') + 26,
        b'0'..=b'9' => u32::from(b - b'0') + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return None,
    })
}

/// Decode standard base64 (RFC 4648 §4): 4 characters carry a 24-bit group
/// of 3 octets; the final group may be 2 or 3 characters (8 or 16 data bits)
/// with the remainder as `=` padding. Unpadded input is accepted (§3.2 makes
/// padding optional when the length is otherwise known — H.264
/// `sprop-parameter-sets` values, RFC 6184 §8.1, appear both ways in the
/// wild). Returns `None` for non-alphabet bytes or an impossible length
/// (a lone trailing character can encode no octet).
pub fn decode_base64(s: &str) -> Option<Vec<u8>> {
    let mut s = s.as_bytes();
    // Strip at most two `=` pad characters (§4: the 8/16-bit final quanta).
    let mut pad = 0usize;
    while pad < 2 {
        match s.split_last() {
            Some((b'=', rest)) => {
                s = rest;
                pad += 1;
            }
            _ => break,
        }
    }
    // Padded input must form complete 4-character groups; and 4k+1 data
    // characters (6 leftover bits) can never encode a whole octet (§4).
    if (pad > 0 && !(s.len() + pad).is_multiple_of(4)) || s.len() % 4 == 1 {
        return None;
    }

    let mut out = Vec::with_capacity(s.len() * 3 / 4 + 2);
    for group in s.chunks(4) {
        let mut acc: u32 = 0;
        for &b in group {
            acc = (acc << 6) | b64_val(b)?;
        }
        let bits = group.len() * 6; // 24, 18 or 12 data bits
        acc <<= 24 - bits; // left-align within the 24-bit group (§4)
        out.extend_from_slice(&acc.to_be_bytes()[1..1 + bits / 8]);
    }
    Some(out)
}

/// Encode to standard base64 with padding (RFC 4648 §4) — RFC 2617 §2 Basic
/// credentials are `base64(userid ":" password)`.
pub fn encode_base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for group in data.chunks(3) {
        let mut acc: u32 = 0;
        for (i, &b) in group.iter().enumerate() {
            acc |= u32::from(b) << (16 - 8 * i);
        }
        // §4: emit one character per 6 data bits, then `=` up to the group.
        let chars = group.len() + 1;
        for i in 0..4 {
            if i < chars {
                out.push(ALPHABET[(acc >> (18 - 6 * i)) as usize & 0x3F] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A realistic RTSP DESCRIBE payload: H.264 (RFC 6184 §8.2 media type
    /// mapping to SDP) + Opus (RFC 7587 §7), with control attributes at both
    /// levels — the shape mediamtx and cameras produce.
    const H264_OPUS: &str = "v=0\r\n\
        o=- 1681692777 1681692777 IN IP4 127.0.0.1\r\n\
        s=Big Buck Bunny\r\n\
        i=A test stream\r\n\
        t=0 0\r\n\
        c=IN IP4 0.0.0.0\r\n\
        a=control:*\r\n\
        a=recvonly\r\n\
        m=video 0 RTP/AVP 96\r\n\
        a=rtpmap:96 H264/90000\r\n\
        a=fmtp:96 packetization-mode=1;profile-level-id=42C01E;sprop-parameter-sets=Z0LAHtkA8SJq,aMuDyyA=\r\n\
        a=control:trackID=0\r\n\
        m=audio 0 RTP/AVP 97\r\n\
        a=rtpmap:97 opus/48000/2\r\n\
        a=fmtp:97 sprop-maxcapturerate=48000;stereo=1\r\n\
        a=sendrecv\r\n\
        a=control:trackID=1\r\n";

    #[test]
    fn parses_h264_opus_describe_payload() {
        let sdp = parse(H264_OPUS).unwrap();
        assert_eq!(sdp.origin, "- 1681692777 1681692777 IN IP4 127.0.0.1");
        assert_eq!(sdp.session_name, "Big Buck Bunny");
        assert_eq!(sdp.connection.as_deref(), Some("IN IP4 0.0.0.0"));
        assert_eq!(sdp.control.as_deref(), Some("*"));
        assert_eq!(sdp.direction, Some(Direction::RecvOnly));
        assert_eq!(sdp.media.len(), 2);

        let v = &sdp.media[0];
        assert_eq!(v.kind, MediaKind::Video);
        assert_eq!(v.port, 0);
        assert_eq!(v.proto, "RTP/AVP");
        assert_eq!(v.payload_types, vec![96]);
        assert_eq!(v.control.as_deref(), Some("trackID=0"));
        // §6.7: no media-level direction — the session-level one applies.
        assert_eq!(v.direction, Some(Direction::RecvOnly));
        let map = &v.rtpmap[&96];
        assert_eq!(map.encoding, "H264");
        assert_eq!(map.clock_rate, 90000);
        assert_eq!(map.params, None);
        let fmtp = &v.fmtp[&96];
        assert_eq!(fmtp.params["packetization-mode"], "1");
        assert_eq!(fmtp.params["profile-level-id"], "42C01E");

        let a = &sdp.media[1];
        assert_eq!(a.kind, MediaKind::Audio);
        assert_eq!(a.payload_types, vec![97]);
        // §6.7: the media-level a=sendrecv overrides the session recvonly.
        assert_eq!(a.direction, Some(Direction::SendRecv));
        let map = &a.rtpmap[&97];
        assert_eq!(map.encoding, "opus");
        assert_eq!(map.clock_rate, 48000);
        assert_eq!(map.params.as_deref(), Some("2"));
        assert_eq!(a.fmtp[&97].params["stereo"], "1");
    }

    /// `sprop-parameter-sets` (RFC 6184 §8.1: comma-separated base64 NAL
    /// units) round-trips through our RFC 4648 codec: encode known SPS/PPS
    /// bytes, splice them into the SDP, parse, decode, compare.
    #[test]
    fn sprop_parameter_sets_round_trip() {
        let sps: &[u8] = &[0x67, 0x42, 0xC0, 0x1E, 0xD9, 0x00, 0xF1, 0x22, 0x6A];
        let pps: &[u8] = &[0x68, 0xCB, 0x83, 0xCB, 0x20];
        let sdp_text = format!(
            "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns= \r\nt=0 0\r\n\
             m=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n\
             a=fmtp:96 packetization-mode=1;sprop-parameter-sets={},{}\r\n",
            encode_base64(sps),
            encode_base64(pps),
        );
        let sdp = parse(&sdp_text).unwrap();
        let sprop = &sdp.media[0].fmtp[&96].params["sprop-parameter-sets"];
        let mut nals = sprop.split(',');
        assert_eq!(decode_base64(nals.next().unwrap()).unwrap(), sps);
        assert_eq!(decode_base64(nals.next().unwrap()).unwrap(), pps);
        assert!(nals.next().is_none());
    }

    #[test]
    fn unknown_lines_and_attributes_are_tolerated() {
        // Unknown type letters (x=), unhandled known ones (b=, k=, z=, r=),
        // unknown attributes, and LF-only line endings — all tolerated.
        let text = "v=0\no=- 1 1 IN IP4 10.0.0.1\ns=x\nb=AS:256\nt=0 0\nz=0 0\n\
                    x=experimental\na=tool:pf-test\nm=video 5000 RTP/AVP 96\n\
                    k=clear:obsolete\na=rtpmap:96 VP8/90000\na=weird\n";
        let sdp = parse(text).unwrap();
        assert_eq!(sdp.media.len(), 1);
        assert_eq!(sdp.media[0].rtpmap[&96].encoding, "VP8");
        // Missing optional lines: no c=, no controls anywhere.
        assert_eq!(sdp.connection, None);
        assert_eq!(sdp.control, None);
        assert_eq!(sdp.media[0].control, None);
    }

    #[test]
    fn static_payload_types_need_no_rtpmap() {
        // §6.6: static assignments (PCMU=0, RTP/AVP profile) may omit rtpmap.
        let sdp =
            parse("v=0\no=- 0 0 IN IP4 0.0.0.0\ns= \nt=0 0\nm=audio 49232 RTP/AVP 0 8\n").unwrap();
        assert_eq!(sdp.media[0].payload_types, vec![0, 8]);
        assert!(sdp.media[0].rtpmap.is_empty());
    }

    #[test]
    fn malformed_descriptions_are_rejected() {
        // Not starting with v=0 (§5.1 / §5 line order).
        assert_eq!(
            parse("o=- 0 0 IN IP4 0.0.0.0\ns=x\n").unwrap_err(),
            SdpError::Version
        );
        assert_eq!(parse("v=1\ns=x\n").unwrap_err(), SdpError::Version);
        // A line without `=` as its second character (§5).
        assert!(matches!(
            parse("v=0\nbogus line\n").unwrap_err(),
            SdpError::Line(_)
        ));
        // m= with too few subfields / a bad port (§5.14).
        assert!(matches!(
            parse("v=0\nm=video 0 RTP/AVP\n").unwrap_err(),
            SdpError::Media(_)
        ));
        assert!(matches!(
            parse("v=0\nm=video notaport RTP/AVP 96\n").unwrap_err(),
            SdpError::Media(_)
        ));
        // rtpmap missing its clock rate (§6.6).
        assert!(matches!(
            parse("v=0\nm=video 0 RTP/AVP 96\na=rtpmap:96 H264\n").unwrap_err(),
            SdpError::Rtpmap(_)
        ));
        // fmtp with no parameters after the format (§6.15).
        assert!(matches!(
            parse("v=0\nm=video 0 RTP/AVP 96\na=fmtp:96\n").unwrap_err(),
            SdpError::Fmtp(_)
        ));
    }

    /// RFC 4648 §10's own test vectors, both directions, plus unpadded and
    /// invalid decode inputs.
    #[test]
    fn base64_rfc4648_vectors() {
        for (plain, coded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode_base64(plain.as_bytes()), coded);
            assert_eq!(decode_base64(coded).unwrap(), plain.as_bytes());
        }
        // Unpadded final quanta are accepted (§3.2).
        assert_eq!(decode_base64("Zg").unwrap(), b"f");
        assert_eq!(decode_base64("Zm8").unwrap(), b"fo");
        // Non-alphabet bytes, impossible lengths, and misplaced padding.
        assert_eq!(decode_base64("Z*=="), None);
        assert_eq!(decode_base64("Z"), None);
        assert_eq!(decode_base64("Zm9vZg==="), None);
        assert_eq!(decode_base64("Zg=Z"), None);
    }
}
