//! H.264 payloading (RFC 6184): Annex-B access units → RTP payloads.
//!
//! Non-interleaved mode (§6.3, packetization-mode=1): single NAL unit
//! packets (§5.6) for NAL units that fit the MTU, FU-A fragmentation (§5.8)
//! for those that do not, and STAP-A aggregation (§5.7.1) whenever two or
//! more consecutive NAL units fit one payload (typically the SPS+PPS run in
//! front of an IDR). All NAL units of one access unit share one RTP
//! timestamp, so single-time aggregation is always legal within an AU
//! (§5.7.1: STAP "SHOULD be used whenever NAL units are aggregated that all
//! share the same NALU-time").
//!
//! Exists v1 for depay round-trip tests; the send elements come later.

/// STAP-A packet type (§5.2 Table 1).
const STAP_A: u8 = 24;
/// FU-A packet type (§5.2 Table 1).
const FU_A: u8 = 28;

/// Split one Annex-B access unit into RTP payloads of at most `mtu` bytes.
/// Returns `(payload, marker)` pairs in transmission order — all packets of
/// the AU carry the AU's timestamp, and the marker is set on the last packet
/// (§5.1: "Set for the very last packet of the access unit").
///
/// `access_unit` must begin with an Annex-B start code (3- or 4-byte);
/// `mtu` is the maximum RTP *payload* size (header excluded) and must be at
/// least 4 so an FU-A (2 header octets, §5.8) always makes progress.
pub fn pay(access_unit: &[u8], mtu: usize) -> Vec<(Vec<u8>, bool)> {
    assert!(mtu >= 4, "mtu must fit the two FU-A header octets plus data");
    let nals = split_annex_b(access_unit);
    let mut out: Vec<(Vec<u8>, bool)> = Vec::new();

    let mut i = 0;
    while i < nals.len() {
        // Greedy STAP-A (§5.7.1): how many consecutive NAL units fit one
        // payload? Layout: 1 type octet + per unit (2 size octets + NAL).
        // The aggregation-unit size field is 16-bit (§5.7.1), so a unit
        // larger than 65535 can never be aggregated (§5.2 note).
        let mut j = i;
        let mut stap_len = 1usize;
        while j < nals.len()
            && nals[j].len() <= usize::from(u16::MAX)
            && stap_len + 2 + nals[j].len() <= mtu
        {
            stap_len += 2 + nals[j].len();
            j += 1;
        }

        if j - i >= 2 {
            // §5.7: the STAP's F bit is the OR of the aggregated F bits
            // ("MUST be cleared if all ... are zero; otherwise ... set");
            // NRI "MUST be the maximum of all the NAL units carried".
            let mut f = 0u8;
            let mut nri = 0u8;
            for nal in &nals[i..j] {
                f |= nal[0] & 0x80;
                nri = nri.max((nal[0] >> 5) & 0x03);
            }
            let mut p = Vec::with_capacity(stap_len);
            p.push(f | (nri << 5) | STAP_A);
            for nal in &nals[i..j] {
                // §5.7.1: 16-bit big-endian size excluding the two size
                // octets, including the NAL header octet.
                p.extend_from_slice(&(nal.len() as u16).to_be_bytes());
                p.extend_from_slice(nal);
            }
            out.push((p, false));
            i = j;
        } else if nals[i].len() <= mtu {
            // Single NAL unit packet (§5.6): the payload is the NAL unit,
            // verbatim — its header octet co-serves as the payload header.
            out.push((nals[i].to_vec(), false));
            i += 1;
        } else {
            fragment_fu_a(nals[i], mtu, &mut out);
            i += 1;
        }
    }

    // §5.1: marker on the very last packet of the access unit.
    if let Some(last) = out.last_mut() {
        last.1 = true;
    }
    out
}

/// FU-A fragmentation (§5.8) of one NAL unit larger than the MTU.
fn fragment_fu_a(nal: &[u8], mtu: usize, out: &mut Vec<(Vec<u8>, bool)>) {
    // §5.8: the FU indicator copies the fragmented NAL unit's F bit (§5.3)
    // and NRI ("MUST be set according to the value of the NRI field in the
    // fragmented NAL unit"), with the FU-A type.
    let indicator = (nal[0] & 0xE0) | FU_A;
    // §5.8: the NAL unit type octet is not sent as payload; its type field
    // travels in the FU header, so only bytes 2..n are fragmented.
    let body = &nal[1..];
    let chunk = mtu - 2;
    // nal.len() > mtu here, so body.len() >= mtu > chunk: at least two
    // fragments — S and E can never meet in one FU header (§5.8: "MUST NOT
    // both be set to one in the same FU header").
    let last = body.len().div_ceil(chunk) - 1;
    for (k, frag) in body.chunks(chunk).enumerate() {
        let mut header = nal[0] & 0x1F;
        if k == 0 {
            header |= 0x80; // S: start of the fragmented NAL unit (§5.8)
        }
        if k == last {
            header |= 0x40; // E: its last byte ends the NAL unit (§5.8)
        }
        let mut p = Vec::with_capacity(2 + frag.len());
        p.push(indicator);
        p.push(header);
        p.extend_from_slice(frag);
        out.push((p, false));
    }
}

/// Split an Annex-B byte stream into NAL units, start codes removed. Both
/// the 3-byte `00 00 01` prefix and the 4-byte `00 00 00 01` form (leading
/// `zero_byte`, H.264 Annex B) are accepted; empty NAL units are dropped.
/// Panics if a non-empty stream does not begin with a start code — the input
/// is produced by our own encoder/depayloader, so that is a caller bug, and
/// silently skipping leading bytes would hide corruption.
fn split_annex_b(stream: &[u8]) -> Vec<&[u8]> {
    let mut positions = Vec::new(); // index of each 00 00 01
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            positions.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    assert!(
        stream.is_empty()
            || positions
                .first()
                .is_some_and(|&p| p <= 1 && stream[..p].iter().all(|&b| b == 0)),
        "pay(): access unit must begin with an Annex-B start code"
    );

    let mut nals = Vec::new();
    for (k, &p) in positions.iter().enumerate() {
        let start = p + 3;
        let mut end = positions.get(k + 1).copied().unwrap_or(stream.len());
        // One zero before the next 00 00 01 is that start code's zero_byte
        // (4-byte form), not part of this NAL unit. Only one is stripped:
        // further trailing zeros stay with the NAL unit.
        if k + 1 < positions.len() && end > start && stream[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            nals.push(&stream[start..end]);
        }
    }
    nals
}

#[cfg(test)]
mod tests {
    use super::split_annex_b;

    #[test]
    fn splits_three_and_four_byte_start_codes() {
        let s = [
            0, 0, 0, 1, 0x67, 0xAA, // 4-byte code, SPS-ish
            0, 0, 1, 0x68, 0xBB, // 3-byte code
            0, 0, 0, 1, 0x65, 0xCC, 0xDD, // 4-byte code again
        ];
        let nals = split_annex_b(&s);
        assert_eq!(nals, vec![&[0x67, 0xAA][..], &[0x68, 0xBB], &[0x65, 0xCC, 0xDD]]);
    }

    #[test]
    fn empty_nal_between_codes_is_dropped() {
        let s = [0, 0, 1, 0, 0, 1, 0x41, 0x02];
        assert_eq!(split_annex_b(&s), vec![&[0x41, 0x02][..]]);
        assert_eq!(split_annex_b(&[]), Vec::<&[u8]>::new());
    }

    #[test]
    #[should_panic(expected = "start code")]
    fn leading_garbage_panics() {
        split_annex_b(&[0x41, 0, 0, 1, 0x42]);
    }
}
