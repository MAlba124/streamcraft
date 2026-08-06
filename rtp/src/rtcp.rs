//! RTCP compound packets (RFC 3550 §6): SR/RR/SDES/BYE parse + build, and the
//! sender report's NTP↔RTP timestamp mapping — the anchor inter-stream (A/V)
//! synchronization hangs off.
//!
//! A compound packet is individual packets concatenated "without any
//! intervening separators" (§6.1); each begins with the common 32-bit fixed
//! word — `V=2|P|count`, packet type, and a length "in 32-bit words minus
//! one, including the header and any padding" (§6.4.1). [`parse_compound`]
//! walks that framing with appendix A.2's validity checks (first packet must
//! be a report, lengths must add up to the datagram, unknown types skipped by
//! length per §6.1). [`build_receiver_report`] emits the minimal legal
//! compound (§6.1: a report packet first, then the mandatory SDES CNAME):
//! `RR + SDES`, with the report-block fields computed by appendix A.3's
//! arithmetic over two [`JitterStats`] snapshots.

// COLD: RTCP is a control channel — reports parse/build on the §6.2 timing
// interval (seconds apart per participant), never on the per-media-packet path;
// every allocation here is once per periodic report.
#![allow(clippy::disallowed_methods)]

use crate::jitter::JitterStats;

/// §6.4.1: "PT=SR=200".
const PT_SR: u8 = 200;
/// §6.4.2: "PT=RR=201".
const PT_RR: u8 = 201;
/// §6.5: "PT=SDES=202".
const PT_SDES: u8 = 202;
/// §6.6: "PT=BYE=203".
const PT_BYE: u8 = 203;
/// §6.5.1: "CNAME=1".
const SDES_CNAME: u8 = 1;

/// The sender-report media anchor (RFC 3550 §6.4.1): "the RTP timestamp
/// `rtp_ts` was sampled at NTP wall time `ntp`". Two streams' SRs place both
/// media timelines on one wall clock.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SenderInfo {
    pub ssrc: u32,
    /// 64-bit NTP timestamp (seconds since 1900 in the top 32 bits).
    pub ntp: u64,
    /// The RTP timestamp corresponding to `ntp`, in the stream's clock rate.
    pub rtp_ts: u32,
    pub packet_count: u32,
    pub octet_count: u32,
}

/// One parsed item of a compound RTCP packet (§6.1: compound = concatenated
/// individual packets, SR/RR first). SDES and BYE packets carry one chunk /
/// SSRC each per item, so a mixer's multi-source packet yields several.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RtcpItem {
    SenderReport(SenderInfo),
    /// Receiver report — parsed but unused on the receive path v1.
    ReceiverReport { ssrc: u32 },
    /// Source description; only CNAME is interesting (§6.5.1).
    Sdes { ssrc: u32, cname: Option<String> },
    Bye { ssrc: u32 },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RtcpError {
    /// A length field (packet, chunk item, report block) runs past the data,
    /// or the individual lengths don't add up to the datagram (A.2).
    Truncated,
    /// `V != 2` on some individual packet (A.2).
    Version,
    /// A compound-framing violation (A.2): the first packet is not SR/RR,
    /// carries padding ("padding should only be applied ... to the last
    /// packet"), or a pad count is nonsensical.
    Format,
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Parse a compound RTCP datagram into items (§6.1 framing: each packet's
/// length field is in 32-bit words minus one).
pub fn parse_compound(datagram: &[u8]) -> Result<Vec<RtcpItem>, RtcpError> {
    let mut items = Vec::new();
    let mut at = 0;
    let mut first = true;
    // A.2: "The length fields of the individual RTCP packets must add up to
    // the overall length of the compound RTCP packet as received" — the walk
    // below either lands exactly on the end or errors on the leftover.
    while at < datagram.len() {
        if datagram.len() < at + 4 {
            return Err(RtcpError::Truncated);
        }
        let b0 = datagram[at];
        // A.2: "RTP version field must equal 2" — checked on every packet
        // (it terminates A.2's scan loop; here a violation is an error).
        if b0 >> 6 != 2 {
            return Err(RtcpError::Version);
        }
        let padding = b0 & 0x20 != 0;
        let count = (b0 & 0x1F) as usize;
        let pt = datagram[at + 1];
        let words = u16::from_be_bytes([datagram[at + 2], datagram[at + 3]]) as usize;
        let end = at + 4 * (words + 1);
        if datagram.len() < end {
            return Err(RtcpError::Truncated);
        }
        if first {
            // A.2: "The payload type field of the first RTCP packet in a
            // compound packet must be equal to SR or RR", and its "padding
            // bit (P) should be zero".
            if pt != PT_SR && pt != PT_RR || padding {
                return Err(RtcpError::Format);
            }
            first = false;
        }
        let mut body = &datagram[at + 4..end];
        if padding {
            // §6.4.1: "The last octet of the padding is a count of how many
            // padding octets should be ignored, including itself".
            let pad = *body.last().ok_or(RtcpError::Truncated)? as usize;
            if pad == 0 || pad > body.len() {
                return Err(RtcpError::Format);
            }
            body = &body[..body.len() - pad];
        }
        match pt {
            PT_SR => parse_sr(count, body, &mut items)?,
            PT_RR => parse_rr(count, body, &mut items)?,
            PT_SDES => parse_sdes(count, body, &mut items)?,
            PT_BYE => parse_bye(count, body, &mut items)?,
            // §6.1: "An implementation SHOULD ignore incoming RTCP packets
            // with types unknown to it."
            _ => {}
        }
        at = end;
    }
    if first {
        // Nothing parsed: an empty datagram is no compound at all (§6.1
        // requires "at least two individual packets", the first a report).
        return Err(RtcpError::Truncated);
    }
    Ok(items)
}

/// SR body after the common word (§6.4.1): sender SSRC, 20 octets of sender
/// info, then `count` 24-octet report blocks (skipped — the receive path only
/// wants the NTP↔RTP anchor). Trailing profile-specific extensions (§6.4.3)
/// are ignored.
fn parse_sr(count: usize, body: &[u8], items: &mut Vec<RtcpItem>) -> Result<(), RtcpError> {
    if body.len() < 24 + 24 * count {
        return Err(RtcpError::Truncated);
    }
    items.push(RtcpItem::SenderReport(SenderInfo {
        ssrc: be32(body, 0),
        ntp: (be32(body, 4) as u64) << 32 | be32(body, 8) as u64,
        rtp_ts: be32(body, 12),
        packet_count: be32(body, 16),
        octet_count: be32(body, 20),
    }));
    Ok(())
}

/// RR body (§6.4.2): reporter SSRC then `count` report blocks — "the five
/// words of sender information are omitted" relative to the SR.
fn parse_rr(count: usize, body: &[u8], items: &mut Vec<RtcpItem>) -> Result<(), RtcpError> {
    if body.len() < 4 + 24 * count {
        return Err(RtcpError::Truncated);
    }
    items.push(RtcpItem::ReceiverReport { ssrc: be32(body, 0) });
    Ok(())
}

/// SDES body (§6.5): `count` chunks, each an SSRC plus items of
/// `(type, length, text)`, the list "terminated by one or more null octets"
/// and null-padded "until the next 32-bit boundary".
fn parse_sdes(count: usize, body: &[u8], items: &mut Vec<RtcpItem>) -> Result<(), RtcpError> {
    let mut at = 0;
    for _ in 0..count {
        if body.len() < at + 4 {
            return Err(RtcpError::Truncated);
        }
        let ssrc = be32(body, at);
        at += 4;
        let mut cname = None;
        loop {
            let ty = *body.get(at).ok_or(RtcpError::Truncated)?;
            if ty == 0 {
                // §6.5: "No length octet follows the null item type octet" —
                // consume it and the pad up to the 32-bit boundary. (Chunks
                // start aligned, so alignment within `body` is alignment
                // within the chunk.)
                at = (at + 1 + 3) & !3;
                break;
            }
            let len = *body.get(at + 1).ok_or(RtcpError::Truncated)? as usize;
            if body.len() < at + 2 + len {
                return Err(RtcpError::Truncated);
            }
            if ty == SDES_CNAME && cname.is_none() {
                // §6.5: "encoded according to the UTF-8 encoding" — an item
                // violating that is treated as absent, not fatal.
                cname = String::from_utf8(body[at + 2..at + 2 + len].to_vec()).ok();
            }
            at += 2 + len;
        }
        items.push(RtcpItem::Sdes { ssrc, cname });
    }
    Ok(())
}

/// BYE body (§6.6): `count` SSRC/CSRC identifiers; the optional trailing
/// "reason for leaving" is ignored.
fn parse_bye(count: usize, body: &[u8], items: &mut Vec<RtcpItem>) -> Result<(), RtcpError> {
    if body.len() < 4 * count {
        return Err(RtcpError::Truncated);
    }
    for i in 0..count {
        items.push(RtcpItem::Bye { ssrc: be32(body, 4 * i) });
    }
    Ok(())
}

/// Build a receiver-report compound (RR + SDES CNAME, the minimal §6.1 shape)
/// from jitter-buffer stats.
///
/// A.3's fraction-lost is *per reporting interval* — "calculated from
/// differences in the expected and received packet counts across the
/// interval" — so the caller supplies `prev`, the [`JitterStats`] snapshot
/// taken when the previous report was sent (`Default` before the first).
/// "Expected" is reconstructed as `received + lost`: every packet the buffer
/// accounted for either arrived or was declared lost, and unlike literal A.3
/// (where duplicates make loss negative) the buffer's counters are monotone,
/// so the 24-bit cumulative field only needs the positive 0x7fffff clamp.
///
/// LSR/DLSR are zero — §6.4.1: "If no SR has been received yet, the field is
/// set to zero" (v1 does not track the last-SR arrival time).
pub fn build_receiver_report(
    reporter_ssrc: u32,
    about_ssrc: u32,
    stats: &JitterStats,
    prev: &JitterStats,
    cname: &str,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + 12 + cname.len());

    // ---- RR (§6.4.2): common word, reporter SSRC, one report block. ----
    out.push(0x81); // V=2, P=0, RC=1
    out.push(PT_RR);
    out.extend_from_slice(&7u16.to_be_bytes()); // 32 octets = 8 words - 1
    out.extend_from_slice(&reporter_ssrc.to_be_bytes());
    out.extend_from_slice(&about_ssrc.to_be_bytes());

    // A.3: expected/received deltas across the interval, fraction as an
    // "8-bit fixed point number with the binary point at the left edge".
    let expected_interval = (stats.received + stats.lost) - (prev.received + prev.lost);
    let received_interval = stats.received - prev.received;
    let lost_interval = expected_interval as i64 - received_interval as i64;
    let fraction = if expected_interval == 0 || lost_interval <= 0 {
        // A.3: "if (expected_interval == 0 || lost_interval <= 0) fraction = 0"
        0
    } else {
        // Clamped: a fully lost interval computes 256, one past the field
        // (§6.4.1 notes no report block is issued for an all-lost source,
        // but the arithmetic shouldn't rely on that).
        (((lost_interval as u64) << 8) / expected_interval).min(255) as u8
    };
    out.push(fraction);
    // A.3: cumulative loss is "carried in 24 bits ... clamped at 0x7fffff".
    let cumulative = stats.lost.min(0x7F_FFFF) as u32;
    out.extend_from_slice(&cumulative.to_be_bytes()[1..]);
    // §6.4.1: low 16 bits the highest sequence number, high 16 the cycle
    // count — A.1's extension, truncated to the 32-bit field.
    out.extend_from_slice(&(stats.ext_highest_seq as u32).to_be_bytes());
    // A.8: "rr->jitter = (u_int32) s->jitter" — the estimate truncated.
    out.extend_from_slice(&(stats.jitter as u32).to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // LSR
    out.extend_from_slice(&0u32.to_be_bytes()); // DLSR

    // ---- SDES (§6.5): one chunk, one CNAME item (§6.5.1). ----
    let cname = &cname.as_bytes()[..cname.len().min(255)]; // 8-bit item length
    let item_len = 2 + cname.len();
    // §6.5: terminate with "one or more null octets ... to pad until the
    // next 32-bit boundary" — A.4's `pad = 4 - (len & 0x3)`, always ≥ 1.
    let pad = 4 - (item_len & 0x3);
    out.push(0x81); // V=2, P=0, SC=1
    out.push(PT_SDES);
    out.extend_from_slice(&(((8 + item_len + pad) / 4 - 1) as u16).to_be_bytes());
    out.extend_from_slice(&reporter_ssrc.to_be_bytes());
    out.push(SDES_CNAME);
    out.push(cname.len() as u8);
    out.extend_from_slice(cname);
    out.resize(out.len() + pad, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built from the §6.4.1 wire diagram. Offsets within the packet:
    /// 0 `V|P|RC`, 1 PT, 2..4 length, 4..8 sender SSRC, 8..16 NTP (msw, lsw),
    /// 16..20 RTP ts, 20..24 packet count, 24..28 octet count, then RC
    /// 24-octet report blocks.
    fn sr_bytes() -> Vec<u8> {
        let mut d = vec![0x81, 200, 0x00, 0x0C]; // RC=1 ⇒ 52 octets ⇒ 12 words
        d.extend_from_slice(&0x1111_2222u32.to_be_bytes()); // SSRC
        d.extend_from_slice(&0xDDEE_AADDu32.to_be_bytes()); // NTP msw
        d.extend_from_slice(&0x8000_0000u32.to_be_bytes()); // NTP lsw
        d.extend_from_slice(&0x0001_2345u32.to_be_bytes()); // RTP timestamp
        d.extend_from_slice(&1000u32.to_be_bytes()); // packet count
        d.extend_from_slice(&64000u32.to_be_bytes()); // octet count
        d.extend_from_slice(&[0xAB; 24]); // one report block (skipped)
        d
    }

    /// §6.5 wire: 0 `V|P|SC`, 1 PT=202, 2..4 length, 4..8 SSRC, then items —
    /// here CNAME(1), length 3, "a@b" at 8..13, then 3 nulls (terminator +
    /// pad to the 32-bit boundary). 16 octets ⇒ length field 3.
    fn sdes_bytes() -> Vec<u8> {
        let mut d = vec![0x81, 202, 0x00, 0x03];
        d.extend_from_slice(&0x3333_4444u32.to_be_bytes());
        d.extend_from_slice(&[1, 3]);
        d.extend_from_slice(b"a@b");
        d.extend_from_slice(&[0, 0, 0]);
        d
    }

    /// §6.6 wire: 0 `V|P|SC`, 1 PT=203, 2..4 length, 4..8 SSRC, then the
    /// optional reason: length octet + text, here exactly filling the word.
    fn bye_bytes() -> Vec<u8> {
        let mut d = vec![0x81, 203, 0x00, 0x02];
        d.extend_from_slice(&0x5555_6666u32.to_be_bytes());
        d.extend_from_slice(&[3]);
        d.extend_from_slice(b"cut");
        d
    }

    #[test]
    fn sr_sdes_bye_compound_parses() {
        let mut d = sr_bytes();
        d.extend_from_slice(&sdes_bytes());
        d.extend_from_slice(&bye_bytes());
        let items = parse_compound(&d).unwrap();
        assert_eq!(
            items,
            vec![
                RtcpItem::SenderReport(SenderInfo {
                    ssrc: 0x1111_2222,
                    ntp: 0xDDEE_AADD_8000_0000,
                    rtp_ts: 0x0001_2345,
                    packet_count: 1000,
                    octet_count: 64000,
                }),
                RtcpItem::Sdes { ssrc: 0x3333_4444, cname: Some("a@b".into()) },
                RtcpItem::Bye { ssrc: 0x5555_6666 },
            ]
        );
    }

    #[test]
    fn unknown_types_are_skipped_by_length() {
        let mut d = vec![0x80, 201, 0x00, 0x01]; // empty RR (RC=0, 8 octets)
        d.extend_from_slice(&7u32.to_be_bytes());
        d.extend_from_slice(&[0x80, 204, 0x00, 0x01, 9, 9, 9, 9]); // APP: ignored
        d.extend_from_slice(&bye_bytes());
        let items = parse_compound(&d).unwrap();
        assert_eq!(
            items,
            vec![
                RtcpItem::ReceiverReport { ssrc: 7 },
                RtcpItem::Bye { ssrc: 0x5555_6666 },
            ]
        );
    }

    #[test]
    fn sdes_multi_item_chunk_finds_the_cname() {
        // NAME(2) "xy" then CNAME(1) "u@h": items are contiguous (§6.5),
        // 4 + 4 + (2+2) + (2+3) + 3 nulls = 20 octets ⇒ length field 4.
        let mut d = vec![0x80, 201, 0x00, 0x01, 0, 0, 0, 1]; // leading empty RR
        d.extend_from_slice(&[0x81, 202, 0x00, 0x04]);
        d.extend_from_slice(&2u32.to_be_bytes());
        d.extend_from_slice(&[2, 2]);
        d.extend_from_slice(b"xy");
        d.extend_from_slice(&[1, 3]);
        d.extend_from_slice(b"u@h");
        d.extend_from_slice(&[0, 0, 0]);
        let items = parse_compound(&d).unwrap();
        assert_eq!(items[1], RtcpItem::Sdes { ssrc: 2, cname: Some("u@h".into()) });
    }

    #[test]
    fn receiver_report_round_trips() {
        let prev = JitterStats { received: 50, lost: 4, ..Default::default() };
        let stats = JitterStats {
            received: 90,
            duplicates: 3,
            lost: 10,
            ext_highest_seq: 0x0001_F123,
            jitter: 42.7,
        };
        let d = build_receiver_report(0xAABB_CCDD, 0x1122_3344, &stats, &prev, "rx@sc");
        // Report block offsets (§6.4.2 diagram): the block starts at 8 after
        // the common word + reporter SSRC.
        assert_eq!(be32(&d, 4), 0xAABB_CCDD); // reporter
        assert_eq!(be32(&d, 8), 0x1122_3344); // reportee
        // A.3: expected_interval = (90+10)-(50+4) = 46, received_interval =
        // 40, lost_interval = 6 ⇒ fraction = (6 << 8) / 46 = 33.
        assert_eq!(d[12], 33);
        assert_eq!(be32(&d, 12) & 0x00FF_FFFF, 10); // cumulative lost, 24 bits
        assert_eq!(be32(&d, 16), 0x0001_F123); // extended highest seq
        assert_eq!(be32(&d, 20), 42); // jitter, truncated (A.8)
        assert_eq!(be32(&d, 24), 0); // LSR
        assert_eq!(be32(&d, 28), 0); // DLSR
        // And the whole compound re-parses to RR + SDES CNAME.
        let items = parse_compound(&d).unwrap();
        assert_eq!(
            items,
            vec![
                RtcpItem::ReceiverReport { ssrc: 0xAABB_CCDD },
                RtcpItem::Sdes { ssrc: 0xAABB_CCDD, cname: Some("rx@sc".into()) },
            ]
        );
    }

    #[test]
    fn first_report_has_zero_fraction_and_interval_zero_is_safe() {
        let stats = JitterStats { received: 10, ..Default::default() };
        let d = build_receiver_report(1, 2, &stats, &JitterStats::default(), "c");
        assert_eq!(d[12], 0); // no losses ⇒ lost_interval = 0 ⇒ fraction 0
        // Identical snapshots: expected_interval = 0 must not divide by zero.
        let d = build_receiver_report(1, 2, &stats, &stats, "c");
        assert_eq!(d[12], 0);
        parse_compound(&d).unwrap();
    }

    #[test]
    fn truncated_and_garbage_are_rejected() {
        // Shorter than one common word.
        assert_eq!(parse_compound(&[]), Err(RtcpError::Truncated));
        assert_eq!(parse_compound(&[0x80, 201]), Err(RtcpError::Truncated));
        // A length field running past the datagram.
        let mut short = sr_bytes();
        short.truncate(40);
        assert_eq!(parse_compound(&short), Err(RtcpError::Truncated));
        // V=1 (A.2).
        let mut bad_ver = sr_bytes();
        bad_ver[0] = 0x41;
        assert_eq!(parse_compound(&bad_ver), Err(RtcpError::Version));
        // First packet not a report (A.2).
        assert_eq!(parse_compound(&sdes_bytes()), Err(RtcpError::Format));
        // Padding on the first packet (A.2).
        let mut padded = sr_bytes();
        padded[0] |= 0x20;
        assert_eq!(parse_compound(&padded), Err(RtcpError::Format));
        // Individual lengths must add up to the datagram (A.2): 3 stray
        // bytes after a valid packet can't even hold a common word.
        let mut stray = sr_bytes();
        stray.extend_from_slice(&[0x80, 204, 0x00]);
        assert_eq!(parse_compound(&stray), Err(RtcpError::Truncated));
        // An SR whose body can't hold the claimed report blocks.
        let mut d = sr_bytes();
        d[0] = 0x82; // RC=2, but only one block's worth of octets
        assert_eq!(parse_compound(&d), Err(RtcpError::Truncated));
        // An SDES chunk missing its null terminator: the CNAME item fills
        // the packet exactly, leaving no room for the mandatory null (§6.5).
        let mut d = vec![0x80, 201, 0x00, 0x01, 0, 0, 0, 1];
        d.extend_from_slice(&[0x81, 202, 0x00, 0x02]);
        d.extend_from_slice(&5u32.to_be_bytes());
        d.extend_from_slice(&[1, 2]);
        d.extend_from_slice(b"a@");
        assert_eq!(parse_compound(&d), Err(RtcpError::Truncated));
    }
}
