//! Deterministic instruction-count harness for the H.264 decode path.
//!
//!   cargo run --release -p pf-h264 --example decode_bench
//!   perf stat -e instructions:u -- <that binary>
//!
//! Wall clock on this class of machine drifts several percent with temperature,
//! and cycles can move opposite to work done; `instructions:u` is reproducible to
//! ~8 significant figures across runs, so an A/B of two builds is meaningful even
//! with other load on the box. The corpus is encoded in-process (no fixtures) and
//! is byte-identical between builds, so the whole delta is attributable to decode.
//!
//! It also prints a checksum of every decoded plane: an optimisation that changes
//! the output is a bug, not a speed-up, and this is what catches it.

#![allow(clippy::disallowed_methods)] // one-shot measurement harness

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: u32 = 128;
const H: u32 = 96;

/// Distinct content per frame so intra and inter both reconstruct something
/// non-trivial (same generator as `h264/tests/h264dec.rs`).
fn make_planes(idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w / 2, h / 2);
    let s = idx as usize;
    let y = (0..w * h)
        .map(|i| {
            let (x, yy) = (i % w, i / w);
            (16 + ((x + yy + s) * (240 - 16)) / (w + h)) as u8
        })
        .collect();
    let u = (0..cw * ch).map(|i| (64 + ((i % cw) * 128) / cw + s) as u8).collect();
    let v = (0..cw * ch).map(|i| (64 + ((i / cw) * 128) / ch + s * 2) as u8).collect();
    (y, u, v)
}

fn encode_i_then_p(p_count: u32, cabac: bool) -> Vec<Vec<u8>> {
    let mut cfg = EncoderConfig::new(W, H);
    // CABAC and CAVLC are completely different entropy decoders; a CAVLC-only corpus
    // never enters `decode_coeff_abs_level_minus1` and so measures none of the CABAC
    // path. Note `EncoderConfig::cabac` alone does *not* switch entropy coder — the
    // CABAC encoder is reached only through the `*_cabac` entry points, and it asserts
    // Main profile (profile_idc >= 77).
    cfg.cabac = cabac;
    if cabac {
        cfg.profile_idc = 77;
    }
    let enc = Encoder::new(cfg);
    let (y0, u0, v0) = make_planes(0);
    let f0 = YuvFrame { width: W, height: H, y: &y0, u: &u0, v: &v0 };
    let idr = if cabac { enc.encode_idr_cabac(&f0) } else { enc.encode_idr(&f0) };
    let mut aus = vec![idr.annex_b.clone()];
    let mut prev = EncodedFrameRef::from(&idr);
    let mut owned = Vec::new();
    for i in 1..=p_count {
        let (y, u, v) = make_planes(i);
        let f = YuvFrame { width: W, height: H, y: &y, u: &u, v: &v };
        let p = if cabac {
            enc.encode_p_cabac(&f, &prev, i, i * 2)
        } else {
            enc.encode_p(&f, &prev, i, i * 2)
        };
        aus.push(p.annex_b.clone());
        owned.push(p);
        prev = EncodedFrameRef::from(owned.last().unwrap());
    }
    aus
}

/// FNV-1a over every emitted plane — cheap, and enough to catch a single
/// changed sample anywhere in the decode.
fn decode_once(aus: &[Vec<u8>], hash: &mut u64) -> usize {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    for au in aus {
        let pkt = Packet::new(0, TimeBase::new(1, 1), au.clone());
        dec.send_packet(&pkt).expect("send_packet");
    }
    dec.flush().expect("flush");
    let mut frames = 0;
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            for plane in vf.image_planes() {
                for &b in plane.data.iter() {
                    *hash ^= b as u64;
                    *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            frames += 1;
        }
    }
    frames
}

fn main() {
    let iters: u32 =
        std::env::var("H264_BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (label, cabac) in [("cavlc", false), ("cabac", true)] {
        let aus = encode_i_then_p(6, cabac);
        let au_bytes: usize = aus.iter().map(Vec::len).sum();
        let mut frames = 0;
        for _ in 0..iters {
            frames += decode_once(&aus, &mut hash);
        }
        println!("{label}: iters={iters} aus={} au_bytes={au_bytes} frames={frames}", aus.len());
    }
    println!("plane_checksum={hash:016x}");
}
