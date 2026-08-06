//! Opus-level VBR — per-frame packet-size election under a
//! target-bitrate drift controller (RFC 6716 §2.1.8 / §3.2.1).
//!
//! RFC 6716 frames VBR as the codec's natural operating mode
//! (§2.1.8): each Opus frame may take any size up to the §3.2.1
//! 1275-byte limit \[R2\], and a transport such as Ogg or RTP carries
//! the per-packet length out of band. This module supplies the
//! *encoder-side election*: given a target bitrate, pick each code-0
//! packet's size so the long-run average lands on target while
//! per-frame sizes track the content — digital-silence frames
//! collapse to the minimum representable packet, transient frames
//! borrow extra bits that later frames repay.
//!
//! Two controller behaviours, both driven by [`VbrRateControl`]:
//!
//! * **Unconstrained VBR** — the election is
//!   `target − drift (+ bias)`, where `drift` accumulates the signed
//!   bit surplus of every emitted packet, clamped to ±one frame's
//!   target so a long silence cannot bank an unbounded spending
//!   spree. The long-run average converges to the target on
//!   non-collapsing content because the CELT finalization fills the
//!   elected size exactly.
//! * **Constrained VBR** — §2.1.8's "simulates a bit reservoir"
//!   discipline: a packet may exceed the per-frame target only by
//!   bits actually banked by previous smaller-than-target packets,
//!   with the bank capped (default: [`DEFAULT_RESERVOIR_MS`] worth of
//!   bits). The RFC defines the reservoir concept but no numeric
//!   bound; the cap here is this crate's documented choice. Under
//!   this discipline the bits of ANY window of `n` packets are
//!   bounded by `n × target + cap`.
//!
//! Mode arms:
//!
//! * **CELT-only** ([`CeltVbrEncoder`]) — full election range
//!   (silence collapse to a 3-byte packet, transient boost).
//! * **Hybrid** ([`HybridVbrEncoderMono`]) — the election is applied
//!   to the CELT tail; the quality-driven §4.2 SILK layer imposes a
//!   per-frame floor
//!   ([`crate::hybrid_packet_encode::HYBRID_MIN_CELT_TAIL_BYTES`]
//!   past the coded SILK bits), and floor raises feed back into the
//!   drift so the average still tracks the target when it is
//!   reachable.
//! * **SILK-only** — the §4.2 layer is inherently VBR (§2.1.8: "the
//!   LP layer is VBR"): [`crate::silk_encoder::SilkEncoderMono`] /
//!   `SilkEncoderStereo` already emit natural quality-driven sizes,
//!   which IS the Opus-level VBR emission for that arm. Electing a
//!   per-frame size around a bitrate target additionally requires
//!   SILK-layer rate control (a quantization-rate knob), which the
//!   crate does not have yet — that frontier is documented, not
//!   faked, so no SILK wrapper appears here.
//!
//! ## Provenance
//!
//! RFC 6716 §2.1.8 / §3.2 / §3.2.1 (staged
//! `docs/audio/opus/rfc6716-opus.txt`). No external library source
//! was consulted.

use crate::celt_frame_encode::CeltFrameEncodeInfo;
use crate::celt_packet_encode::CeltEncoder;
use crate::hybrid_packet_encode::HybridEncoderMono;
use crate::toc::Bandwidth;
use crate::Error;

/// §3.2.1 maximum Opus frame payload (\[R2\]).
const MAX_FRAME_BYTES: usize = 1275;

/// Smallest electable code-0 packet: TOC + the 2-byte CELT minimum.
const MIN_PACKET_BYTES: usize = 3;

/// Largest electable code-0 packet: TOC + the §3.2.1 frame limit.
const MAX_PACKET_BYTES: usize = 1 + MAX_FRAME_BYTES;

/// Default constrained-VBR reservoir depth, in milliseconds of the
/// target bitrate. RFC 6716 §2.1.8 specifies the reservoir *concept*
/// (constrained VBR "simulates a bit reservoir") without a numeric
/// bound; 100 ms is this crate's documented default.
pub const DEFAULT_RESERVOIR_MS: u32 = 100;

/// Per-frame packet-size election under a target-bitrate drift
/// controller (see the module docs for the two behaviours).
///
/// All accounting is at the *packet* level (TOC byte included), so
/// the tracked rate is the transport payload rate.
#[derive(Debug, Clone)]
pub struct VbrRateControl {
    /// Target bits per packet (TOC included).
    target_bits: f64,
    /// Signed accumulated surplus (`produced − target`), clamped to
    /// ±[`Self::drift_window_bits`].
    drift_bits: f64,
    /// Drift clamp: one frame's target.
    drift_window_bits: f64,
    /// Constrained-VBR mode flag.
    constrained: bool,
    /// Banked below-target bits available for above-target spending
    /// (constrained mode only).
    reservoir_bits: f64,
    /// Reservoir cap (constrained mode only).
    reservoir_cap_bits: f64,
}

impl VbrRateControl {
    /// New controller for `target_bitrate_bps` at a frame duration of
    /// `frame_tenths_ms` tenths of a millisecond (25/50/100/200 for
    /// the CELT arm; 100/200 for Hybrid), with the default reservoir.
    ///
    /// Rejects targets no packet ladder can meet: below the 3-byte
    /// minimum packet or above the §3.2.1 maximum per frame.
    pub fn new(
        target_bitrate_bps: u32,
        frame_tenths_ms: u16,
        constrained: bool,
    ) -> Result<Self, Error> {
        Self::with_reservoir_ms(
            target_bitrate_bps,
            frame_tenths_ms,
            constrained,
            DEFAULT_RESERVOIR_MS,
        )
    }

    /// [`Self::new`] with an explicit constrained-VBR reservoir depth
    /// in milliseconds of target bitrate (ignored when
    /// `constrained == false`).
    pub fn with_reservoir_ms(
        target_bitrate_bps: u32,
        frame_tenths_ms: u16,
        constrained: bool,
        reservoir_ms: u32,
    ) -> Result<Self, Error> {
        if target_bitrate_bps == 0 || frame_tenths_ms == 0 {
            return Err(Error::MalformedPacket);
        }
        let target_bits = f64::from(target_bitrate_bps) * f64::from(frame_tenths_ms) / 10_000.0;
        let min_bits = (MIN_PACKET_BYTES * 8) as f64;
        let max_bits = (MAX_PACKET_BYTES * 8) as f64;
        if !(min_bits..=max_bits).contains(&target_bits) {
            return Err(Error::MalformedPacket);
        }
        let reservoir_cap_bits = if constrained {
            f64::from(target_bitrate_bps) * f64::from(reservoir_ms) / 1000.0
        } else {
            0.0
        };
        Ok(Self {
            target_bits,
            drift_bits: 0.0,
            drift_window_bits: target_bits,
            constrained,
            reservoir_bits: 0.0,
            reservoir_cap_bits,
        })
    }

    /// Target bits per packet (TOC included).
    #[must_use]
    pub fn target_bits_per_packet(&self) -> f64 {
        self.target_bits
    }

    /// Whether the controller runs the constrained-VBR discipline.
    #[must_use]
    pub fn is_constrained(&self) -> bool {
        self.constrained
    }

    /// The per-packet hard ceiling the constrained discipline allows
    /// RIGHT NOW: `target + banked reservoir` (unconstrained: the
    /// §3.2.1 packet maximum).
    #[must_use]
    pub fn constrained_ceiling_bits(&self) -> f64 {
        if self.constrained {
            self.target_bits + self.reservoir_bits
        } else {
            (MAX_PACKET_BYTES * 8) as f64
        }
    }

    /// Elect the next packet's total size in bytes (TOC included).
    /// `bias_bits` is a content-driven adjustment (positive on
    /// transients); the constrained ceiling and the §3.2.1 limits are
    /// applied after the drift correction.
    #[must_use]
    pub fn elect_packet_bytes(&self, bias_bits: f64) -> usize {
        let mut want = self.target_bits + bias_bits - self.drift_bits;
        if self.constrained {
            want = want.min(self.constrained_ceiling_bits());
        }
        let bytes = (want / 8.0).round() as i64;
        bytes.clamp(MIN_PACKET_BYTES as i64, MAX_PACKET_BYTES as i64) as usize
    }

    /// Commit an emitted packet of `packet_bytes` total bytes: update
    /// the drift (clamped to ±one frame's target) and, in constrained
    /// mode, the banked reservoir (clamped to `[0, cap]`).
    pub fn commit(&mut self, packet_bytes: usize) {
        let produced = (packet_bytes * 8) as f64;
        self.drift_bits = (self.drift_bits + produced - self.target_bits)
            .clamp(-self.drift_window_bits, self.drift_window_bits);
        if self.constrained {
            self.reservoir_bits = (self.reservoir_bits + self.target_bits - produced)
                .clamp(0.0, self.reservoir_cap_bits);
        }
    }

    /// Forget all carried rate state (stream start / §4.5.2 reset).
    pub fn reset(&mut self) {
        self.drift_bits = 0.0;
        self.reservoir_bits = 0.0;
    }
}

/// Transient pre-detector for the size election: flags a frame whose
/// short-window energy jumps well above the recent past, so the
/// election can grant it extra bits *before* the CELT layer runs.
///
/// This is a VBR-level heuristic on the raw PCM (2.5 ms blocks, an
/// 18 dB max-over-mean energy ratio against the running history,
/// silence-floor gated); the CELT layer's own §5.3 transient analysis
/// still makes the coding decision independently.
#[derive(Debug, Clone)]
struct TransientPredetect {
    /// Running mean block energy carried across frames.
    history_energy: f64,
    channels: usize,
}

/// Pre-detector block length per channel (2.5 ms at 48 kHz).
const PREDETECT_BLOCK: usize = 120;

/// Energy ratio (max block over history mean) that flags a transient:
/// 64× ≈ 18 dB.
const PREDETECT_RATIO: f64 = 64.0;

/// Absolute per-sample mean-square floor below which no transient is
/// flagged (guards silence → noise-floor jumps).
const PREDETECT_FLOOR: f64 = 1000.0;

impl TransientPredetect {
    fn new(channels: usize) -> Self {
        Self {
            history_energy: 0.0,
            channels,
        }
    }

    fn reset(&mut self) {
        self.history_energy = 0.0;
    }

    /// Returns `true` when `pcm` (interleaved) looks transient
    /// against the carried history, and folds this frame's blocks
    /// into the history.
    fn is_transient(&mut self, pcm: &[i16]) -> bool {
        let ch = self.channels;
        let block = PREDETECT_BLOCK * ch;
        let mut transient = false;
        for chunk in pcm.chunks(block) {
            let mut e = 0.0f64;
            for &v in chunk {
                let x = f64::from(v);
                e += x * x;
            }
            e /= chunk.len() as f64; // mean square per sample
            if e > PREDETECT_FLOOR
                && self.history_energy > 0.0
                && e > PREDETECT_RATIO * self.history_energy
            {
                transient = true;
            }
            // Exponential history (quarter-weight per 2.5 ms block).
            self.history_energy = if self.history_energy == 0.0 {
                e
            } else {
                0.75 * self.history_energy + 0.25 * e
            };
        }
        transient
    }
}

/// Fraction of the per-frame target granted as transient bonus.
const TRANSIENT_BOOST_FRACTION: f64 = 0.25;

/// A CELT-only VBR packet encoder: [`CeltEncoder`] driven by a
/// [`VbrRateControl`] election with silence collapse and a transient
/// boost.
#[derive(Debug, Clone)]
pub struct CeltVbrEncoder {
    enc: CeltEncoder,
    rc: VbrRateControl,
    predetect: TransientPredetect,
}

impl CeltVbrEncoder {
    /// New CELT-only VBR encoder (see [`CeltEncoder::new`] for the
    /// configuration matrix) targeting `target_bitrate_bps` on the
    /// wire, constrained per [`VbrRateControl`].
    pub fn new(
        bandwidth: Bandwidth,
        frame_tenths_ms: u16,
        stereo: bool,
        target_bitrate_bps: u32,
        constrained: bool,
    ) -> Result<Self, Error> {
        let enc = CeltEncoder::new(bandwidth, frame_tenths_ms, stereo)?;
        let rc = VbrRateControl::new(target_bitrate_bps, frame_tenths_ms, constrained)?;
        let channels = enc.channels();
        Ok(Self {
            enc,
            rc,
            predetect: TransientPredetect::new(channels),
        })
    }

    /// Samples per channel consumed by one packet (48 kHz).
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        self.enc.frame_samples()
    }

    /// Channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.enc.channels()
    }

    /// The rate controller (drift / reservoir state inspection).
    #[must_use]
    pub fn rate_control(&self) -> &VbrRateControl {
        &self.rc
    }

    /// Reset all carried state (stream start / §4.5.2).
    pub fn reset(&mut self) {
        self.enc.reset();
        self.rc.reset();
        self.predetect.reset();
    }

    /// Encode one frame of interleaved 48 kHz PCM at a VBR-elected
    /// size. Returns the packet (its length is the election) and the
    /// CELT frame info.
    pub fn encode_frame(&mut self, pcm: &[i16]) -> Result<(Vec<u8>, CeltFrameEncodeInfo), Error> {
        if pcm.len() != self.channels() * self.frame_samples() {
            return Err(Error::MalformedPacket);
        }
        let packet_bytes = if pcm.iter().all(|&v| v == 0) {
            // Digital silence: collapse to the minimum packet. The
            // CELT layer codes its §4.3 silence flag inside it.
            MIN_PACKET_BYTES
        } else {
            let bias = if self.predetect.is_transient(pcm) {
                TRANSIENT_BOOST_FRACTION * self.rc.target_bits_per_packet()
            } else {
                0.0
            };
            self.rc.elect_packet_bytes(bias)
        };
        let (packet, info) = self.enc.encode_packet(pcm, packet_bytes - 1)?;
        debug_assert_eq!(packet.len(), packet_bytes);
        self.rc.commit(packet.len());
        Ok((packet, info))
    }
}

/// A mono Hybrid VBR packet encoder: [`HybridEncoderMono`] driven by
/// a [`VbrRateControl`] election, with the SILK-layer floor raise
/// feeding back into the drift (see the module docs).
#[derive(Debug, Clone)]
pub struct HybridVbrEncoderMono {
    enc: HybridEncoderMono,
    rc: VbrRateControl,
}

impl HybridVbrEncoderMono {
    /// New mono Hybrid VBR encoder (SWB/FB × 10/20 ms, per
    /// [`HybridEncoderMono::new`]) targeting `target_bitrate_bps`.
    pub fn new(
        bandwidth: Bandwidth,
        frame_tenths_ms: u16,
        target_bitrate_bps: u32,
        constrained: bool,
    ) -> Result<Self, Error> {
        let enc = HybridEncoderMono::new(bandwidth, frame_tenths_ms)?;
        let rc = VbrRateControl::new(target_bitrate_bps, frame_tenths_ms, constrained)?;
        Ok(Self { enc, rc })
    }

    /// 48 kHz samples per packet.
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        self.enc.frame_samples()
    }

    /// The rate controller (drift / reservoir state inspection).
    #[must_use]
    pub fn rate_control(&self) -> &VbrRateControl {
        &self.rc
    }

    /// Reset all carried state (§4.5.2).
    pub fn reset(&mut self) {
        self.enc.reset();
        self.rc.reset();
    }

    /// Encode one packet of mono 48 kHz PCM at a VBR-elected size.
    /// The SILK layer may raise the election to its floor; the
    /// returned packet's length is the actual size, and the raise is
    /// charged to the drift so later frames repay it.
    pub fn encode_frame(&mut self, pcm: &[i16]) -> Result<Vec<u8>, Error> {
        let elected = self.rc.elect_packet_bytes(0.0);
        let packet = self.enc.encode_packet_elected(pcm, elected - 1)?;
        self.rc.commit(packet.len());
        Ok(packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_rejects_unreachable_targets() {
        // 2.5 ms frames at 6 kb/s → under 2 bytes/packet: unreachable.
        assert!(VbrRateControl::new(6_000, 25, false).is_err());
        // 20 ms at 600 kb/s → 1500 bytes/packet: over the §3.2.1 cap.
        assert!(VbrRateControl::new(600_000, 200, false).is_err());
        assert!(VbrRateControl::new(0, 200, false).is_err());
        // 64 kb/s / 20 ms → 160 bytes: fine.
        let rc = VbrRateControl::new(64_000, 200, false).unwrap();
        assert!((rc.target_bits_per_packet() - 1280.0).abs() < 1e-9);
    }

    #[test]
    fn steady_election_sits_on_target() {
        let mut rc = VbrRateControl::new(64_000, 200, false).unwrap();
        let mut total = 0usize;
        for _ in 0..100 {
            let n = rc.elect_packet_bytes(0.0);
            rc.commit(n);
            total += n;
        }
        // 64 kb/s, 20 ms → exactly 160 bytes/packet.
        assert_eq!(total, 100 * 160);
    }

    #[test]
    fn drift_repays_a_transient_boost() {
        let mut rc = VbrRateControl::new(64_000, 200, false).unwrap();
        // One boosted frame (+25%), then steady: the long-run average
        // must return to target.
        let boosted = rc.elect_packet_bytes(0.25 * rc.target_bits_per_packet());
        assert!(boosted > 160);
        rc.commit(boosted);
        let mut total = boosted;
        for _ in 0..99 {
            let n = rc.elect_packet_bytes(0.0);
            rc.commit(n);
            total += n;
        }
        let avg = total as f64 / 100.0;
        assert!((avg - 160.0).abs() < 1.0, "avg {avg}");
    }

    #[test]
    fn silence_credit_is_clamped() {
        let mut rc = VbrRateControl::new(64_000, 200, false).unwrap();
        // A long run of minimum packets (silence collapse) may bank at
        // most one frame's target of credit: the next election is
        // bounded by 2× target.
        for _ in 0..50 {
            rc.commit(MIN_PACKET_BYTES);
        }
        let n = rc.elect_packet_bytes(0.0);
        assert!(n <= 2 * 160, "post-silence election {n}");
        assert!(n > 160, "post-silence election should spend credit");
    }

    #[test]
    fn constrained_ceiling_needs_banked_bits() {
        let mut rc = VbrRateControl::new(64_000, 200, true).unwrap();
        // Nothing banked yet: even a huge bias cannot pass target.
        let n = rc.elect_packet_bytes(10_000.0);
        assert_eq!(n, 160);
        // Bank some bits with small packets, then the ceiling rises —
        // but never above target + cap (100 ms of 64 kb/s = 800 bytes).
        for _ in 0..100 {
            rc.commit(60);
        }
        let ceil_bits = rc.constrained_ceiling_bits();
        assert!(ceil_bits <= 1280.0 + 6400.0 + 1e-9);
        let n = rc.elect_packet_bytes(1e9);
        assert_eq!(n, ((ceil_bits / 8.0).round()) as usize);
    }

    #[test]
    fn constrained_window_bound_holds_under_adversarial_bias() {
        // Whatever the bias sequence, every WINDOW of packets obeys
        // n*target + cap.
        let mut rc = VbrRateControl::with_reservoir_ms(48_000, 100, true, 50).unwrap();
        let target = rc.target_bits_per_packet(); // 480 bits
        let cap = 48_000.0 * 0.05; // 2400 bits
        let mut produced = Vec::new();
        for i in 0..500 {
            let bias = match i % 7 {
                0 => 1e9,
                1 => -1e9,
                2 => 0.0,
                3 => 5000.0,
                4 => -300.0,
                5 => 900.0,
                _ => 100.0,
            };
            let n = rc.elect_packet_bytes(bias);
            rc.commit(n);
            produced.push((n * 8) as f64);
        }
        for w in [1usize, 5, 20, 100] {
            for win in produced.windows(w) {
                let sum: f64 = win.iter().sum();
                assert!(
                    sum <= w as f64 * target + cap + 1e-6,
                    "window {w} bust: {sum}"
                );
            }
        }
    }

    #[test]
    fn reset_forgets_drift_and_reservoir() {
        let mut rc = VbrRateControl::new(64_000, 200, true).unwrap();
        for _ in 0..10 {
            rc.commit(60);
        }
        rc.reset();
        assert_eq!(rc.elect_packet_bytes(0.0), 160);
        assert!((rc.constrained_ceiling_bits() - 1280.0).abs() < 1e-9);
    }

    #[test]
    fn predetector_flags_a_click_after_quiet_tone() {
        let mut p = TransientPredetect::new(1);
        // 20 ms of quiet tone.
        let mut quiet = vec![0i16; 960];
        for (i, v) in quiet.iter_mut().enumerate() {
            *v = ((i as f64 * 0.05).sin() * 200.0) as i16;
        }
        assert!(!p.is_transient(&quiet));
        // Same tone with a hard click in the middle.
        let mut click = quiet.clone();
        for v in click.iter_mut().skip(480).take(120) {
            *v = 24_000;
        }
        assert!(p.is_transient(&click));
        // Silence never flags.
        let mut p2 = TransientPredetect::new(1);
        assert!(!p2.is_transient(&vec![0i16; 960]));
    }
}
