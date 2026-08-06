//! RFC 6716 conformance gate for the adopted `oxideav-opus` decoder — the "it must actually
//! decode" rubric this workspace holds every codec to. Feeds each canonical test vector's
//! packets to one stateful [`OpusDecoder`] and best-offset-SNRs the output against the
//! reference `.dec` (always stereo, 16-bit LE, 48 kHz; mono decodes are upmixed to match).
//!
//! Vectors: https://opus-codec.org/testvectors/opus_testvectors.tar.gz
//!   cargo run -p pf-opus --example conformance -- /path/to/opus_testvectors

#![allow(clippy::disallowed_methods)] // one-shot validation harness, not a hot path

fn main() {
    let dir = std::env::args().nth(1).expect("usage: conformance <opus_testvectors dir>");
    let mut worst = f64::INFINITY;
    for n in 1..=12 {
        let bit = format!("{dir}/testvector{n:02}.bit");
        let dec = format!("{dir}/testvector{n:02}.dec");
        let Ok(bytes) = std::fs::read(&bit) else { continue };
        let Ok(refb) = std::fs::read(&dec) else { continue };
        let packets = split_bit(&bytes);
        let out = decode_all(&packets);
        let reference: Vec<i16> =
            refb.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let (snr, off) = best_offset_snr(&reference, &out);
        worst = worst.min(snr);
        println!(
            "vec{n:02}  {:>6} packets  SNR {:>7.2} dB  (offset {off} samples)",
            packets.len(),
            snr
        );
    }
    println!("\nworst-vector SNR: {worst:.2} dB");
    // SILK is bit-exact and CELT/hybrid sit at the float-noise floor, so real Opus should clear
    // tens of dB everywhere; treat < 20 dB as a decode failure.
    if worst < 20.0 {
        eprintln!("FAIL: a vector decoded below 20 dB — the decoder is not conformant");
        std::process::exit(1);
    }
    println!("PASS");
}

/// Split a `.bit` vector into packets: each is a 4-byte BE length, a 4-byte BE final-range
/// (ignored on decode), then `length` packet bytes.
fn split_bit(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut i = 0;
    while i + 8 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 8;
        if i + len > data.len() {
            break;
        }
        packets.push(data[i..i + len].to_vec());
        i += len;
    }
    packets
}

fn decode_all(packets: &[Vec<u8>]) -> Vec<i16> {
    let mut dec = oxideav_opus::OpusDecoder::new();
    let mut out = Vec::new();
    let mut pcm = Vec::new(); // reused across packets — validates the no-alloc `decode_packet_into`
    for p in packets {
        if let Ok(channels) = dec.decode_packet_into(p, &mut pcm) {
            if channels == 1 {
                for &s in &pcm {
                    out.push(s);
                    out.push(s); // upmix mono → stereo to match the reference
                }
            } else {
                out.extend_from_slice(&pcm);
            }
        }
    }
    out
}

/// Best SNR (dB) absorbing the decoder's fixed startup delay: find the offset on a 1-second
/// window (cheap), then score the full signal at that offset.
fn best_offset_snr(reference: &[i16], test: &[i16]) -> (f64, usize) {
    const WIN: usize = 96_000; // 1 s of stereo @ 48 kHz
    let mut best_off = 0;
    let mut best_win = f64::NEG_INFINITY;
    for off in (0..9600).step_by(2) {
        if off + WIN >= test.len() || WIN >= reference.len() {
            break;
        }
        let snr = snr_at(&reference[..WIN], &test[off..off + WIN]);
        if snr > best_win {
            best_win = snr;
            best_off = off;
        }
    }
    if best_off >= test.len() {
        return (f64::NEG_INFINITY, 0);
    }
    let t = &test[best_off..];
    let n = reference.len().min(t.len());
    (snr_at(&reference[..n], &t[..n]), best_off)
}

fn snr_at(reference: &[i16], test: &[i16]) -> f64 {
    let n = reference.len().min(test.len());
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for i in 0..n {
        let r = reference[i] as f64;
        let e = r - test[i] as f64;
        sig += r * r;
        err += e * e;
    }
    if err <= 0.0 {
        200.0
    } else {
        10.0 * (sig / err).log10()
    }
}
