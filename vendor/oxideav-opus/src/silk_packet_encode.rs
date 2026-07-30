//! SILK-only Opus **packet** encoding — RFC 6716 §3.1 / §4.2.2-§4.2.6.
//!
//! Composes the encode-side mirrors into a complete, decoder-ready
//! Opus packet: the §3.1 TOC byte (code 0, one Opus frame), the
//! §4.2.3 / §4.2.4 header bits, and one regular SILK frame per 20 ms
//! time interval (or a single 10 ms frame) written in Table-5 order by
//! [`crate::silk_decode::encode_silk_frame`], finalized through the
//! §5.1.5 range-coder termination.
//!
//! The produced packets decode end-to-end through
//! [`crate::decoder::OpusDecoder::decode_packet`] on a fresh decoder:
//! the per-frame carried state (previous gain / lag / NLSF) is
//! threaded here exactly the way the packet decoder threads it, so
//! the per-frame `SilkFrameDecoded` predictions returned to the
//! caller equal what the decoder reconstructs.
//!
//! Scope: **mono and stereo**. LBRR (in-band FEC, §4.2.5) emission is
//! supported via [`encode_silk_only_packet_mono_with_lbrr`]: LBRR
//! frames are written ahead of the regular frames with their own
//! independent carried state, exactly mirroring the decode-side LBRR
//! walk, and the §4.2.3 / §4.2.4 LBRR flags are derived from which
//! intervals carry a redundancy script. The **stereo** mid/side
//! interleave (§4.2.2) is provided by
//! [`encode_silk_only_packet_stereo`]: per 20 ms interval the mid
//! SILK frame (carrying the §4.2.7.1 stereo prediction weights and,
//! when the interval's side channel is not active, the §4.2.7.2
//! mid-only flag) is written, then the side SILK frame when coded,
//! with two independent per-channel carried states threaded exactly
//! the way [`crate::decoder::OpusDecoder`]'s stereo walk threads them.

use crate::decoder::ChannelDecodeState;
use crate::range_encoder::RangeEncoder;
use crate::silk_decode::{
    encode_silk_frame, SilkFrameConfig, SilkFrameDecoded, SilkFrameSymbols, StereoHeaderContext,
};
use crate::silk_excitation::SilkFrameSize;
use crate::silk_header::{silk_frame_count, PerFrameLbrr, SilkChannelHeader, SilkHeaderBits};
use crate::toc::{Bandwidth, FrameCountCode, Mode, OpusTocByte};
use crate::Error;

/// Encode one **mono, SILK-only, code-0** Opus packet from per-frame
/// symbol scripts.
///
/// * `bandwidth` — NB / MB / WB (the SILK internal bandwidths).
/// * `frame_size_tenths_ms` — the §3.1 Opus frame duration: 100, 200,
///   400, or 600 (10/20/40/60 ms). A 10 ms packet carries one 10 ms
///   SILK frame; the others carry one 20 ms SILK frame per interval
///   (1, 2, or 3), and `frames.len()` must match.
/// * `frames` — one Table-5 symbol script per SILK frame. Each frame's
///   §4.2.3 VAD bit is derived from its §4.2.7.3 frame type (types
///   `0..=1` are inactive, `2..=5` active). Frames after the first
///   must delta-code their first gain (the carried state makes the
///   first subframe non-independent), exactly as the decoder expects.
///
/// Returns the packet bytes plus the per-frame [`SilkFrameDecoded`]
/// predictions (what a fresh decoder will reconstruct).
pub fn encode_silk_only_packet_mono(
    bandwidth: Bandwidth,
    frame_size_tenths_ms: u16,
    frames: &[SilkFrameSymbols<'_>],
) -> Result<(Vec<u8>, Vec<SilkFrameDecoded>), Error> {
    let n = silk_frame_count(frame_size_tenths_ms).ok_or(Error::MalformedPacket)? as usize;
    let no_lbrr = vec![None; n];
    let (packet, regular, _) =
        encode_silk_only_packet_mono_with_lbrr(bandwidth, frame_size_tenths_ms, frames, &no_lbrr)?;
    Ok((packet, regular))
}

/// [`encode_silk_only_packet_mono`] with §4.2.5 LBRR (in-band FEC)
/// emission: `lbrr[idx]` optionally carries a redundancy script for
/// SILK frame `idx`'s time interval (a re-encode of the interval the
/// *previous* packet covered, per §2.1.7). LBRR frames are written
/// ahead of the regular frames with their own independent carried
/// state, exactly the way the packet decoder walks them; each LBRR
/// script's frame type must be active (`2..=5`, §4.2.7.3), its first
/// gain independent only on the first coded LBRR frame, and its LTP
/// scaling present only on the first coded LBRR frame.
///
/// Returns the packet bytes, the regular per-frame predictions, and
/// the LBRR per-frame predictions.
#[allow(clippy::type_complexity)]
pub fn encode_silk_only_packet_mono_with_lbrr(
    bandwidth: Bandwidth,
    frame_size_tenths_ms: u16,
    frames: &[SilkFrameSymbols<'_>],
    lbrr: &[Option<SilkFrameSymbols<'_>>],
) -> Result<
    (
        Vec<u8>,
        Vec<SilkFrameDecoded>,
        Vec<Option<SilkFrameDecoded>>,
    ),
    Error,
> {
    let num_silk_frames = silk_frame_count(frame_size_tenths_ms).ok_or(Error::MalformedPacket)?;
    if frames.len() != num_silk_frames as usize || lbrr.len() != num_silk_frames as usize {
        return Err(Error::MalformedPacket);
    }
    let frame_size = if frame_size_tenths_ms == 100 {
        SilkFrameSize::TenMs
    } else {
        SilkFrameSize::TwentyMs
    };

    // §3.1 TOC byte: SILK-only, mono, code 0 (one Opus frame).
    let toc = OpusTocByte::compose_byte(
        Mode::SilkOnly,
        bandwidth,
        frame_size_tenths_ms,
        false,
        FrameCountCode::One,
    )?;

    let mut re = RangeEncoder::new();

    // §4.2.3 / §4.2.4 header bits: VAD per frame from the frame type,
    // LBRR flags from which intervals carry a redundancy script.
    let mut vad_flags = 0u8;
    for (idx, f) in frames.iter().enumerate() {
        if f.header.frame_type >= 2 {
            vad_flags |= 1 << idx;
        }
    }
    let mut lbrr_bits = 0u8;
    for (idx, l) in lbrr.iter().enumerate() {
        if l.is_some() {
            lbrr_bits |= 1 << idx;
        }
    }
    let header = SilkHeaderBits {
        num_silk_frames,
        mid: SilkChannelHeader {
            vad_flags,
            lbrr_flag: lbrr_bits != 0,
        },
        side: None,
        per_frame_lbrr: PerFrameLbrr {
            mid: lbrr_bits,
            side: 0,
        },
    };
    header.encode(&mut re)?;

    // §4.2.5 LBRR frames: written ahead of the regular frames, with
    // their own independent carried state (they form their own
    // sequence), mirroring the decoder's LBRR walk. Per §4.2.7.3 every
    // LBRR frame is active-coded.
    let mut lbrr_prev_gain: Option<u8> = None;
    let mut lbrr_prev_lag: Option<i32> = None;
    let mut lbrr_first = true;
    let mut lbrr_predictions: Vec<Option<SilkFrameDecoded>> = Vec::with_capacity(lbrr.len());
    for entry in lbrr.iter() {
        let Some(symbols) = entry else {
            // §4.2.4 LBRR-flag gap: the next coded LBRR frame codes
            // independently (§4.2.7.4), with an absolute lag
            // (§4.2.7.6.1) and the §4.2.7.6.3 scaling field present
            // again — the exact mirror of the decoder's gap handling.
            lbrr_prev_gain = None;
            lbrr_prev_lag = None;
            lbrr_first = true;
            lbrr_predictions.push(None);
            continue;
        };
        if symbols.header.frame_type < 2 {
            // §4.2.7.3: LBRR frames use the active PDF.
            return Err(Error::MalformedPacket);
        }
        let cfg = SilkFrameConfig {
            bandwidth,
            frame_size,
            voice_active: true,
            first_subframe_independent: lbrr_first || lbrr_prev_gain.is_none(),
            previous_log_gain: lbrr_prev_gain,
            previous_primary_lag: lbrr_prev_lag,
            ltp_scaling_present: lbrr_first,
            lsf_interp_after_reset: lbrr_first,
            previous_nlsf_q15: None,
            previous_nlsf_len: 0,
            stereo: None,
        };
        let decoded = encode_silk_frame(&mut re, cfg, symbols)?;
        lbrr_prev_gain = Some(decoded.gains.last_log_gain());
        // §4.2.7.6.1: only a VOICED frame arms relative lag coding.
        lbrr_prev_lag = if decoded.ltp.is_voiced() {
            Some(decoded.ltp.primary_lag())
        } else {
            None
        };
        lbrr_first = false;
        lbrr_predictions.push(Some(decoded));
    }

    // §4.2.6 regular SILK frames with the carried state threaded the
    // same way the packet decoder threads it (fresh-decoder start).
    let mut prev_gain: Option<u8> = None;
    let mut prev_lag: Option<i32> = None;
    let mut prev_nlsf: Option<[i16; crate::silk_lsf_stage2::D_LPC_MAX]> = None;
    let mut prev_nlsf_len = 0usize;
    let mut first = true;
    let mut predictions = Vec::with_capacity(frames.len());
    for (idx, symbols) in frames.iter().enumerate() {
        let cfg = SilkFrameConfig {
            bandwidth,
            frame_size,
            voice_active: (vad_flags >> idx) & 1 == 1,
            first_subframe_independent: first || prev_gain.is_none(),
            previous_log_gain: prev_gain,
            previous_primary_lag: prev_lag,
            ltp_scaling_present: first,
            lsf_interp_after_reset: first || prev_nlsf.is_none(),
            previous_nlsf_q15: prev_nlsf,
            previous_nlsf_len: prev_nlsf_len,
            stereo: None,
        };
        let decoded = encode_silk_frame(&mut re, cfg, symbols)?;
        prev_gain = Some(decoded.gains.last_log_gain());
        // §4.2.7.6.1: only a VOICED frame arms relative lag coding.
        prev_lag = if decoded.ltp.is_voiced() {
            Some(decoded.ltp.primary_lag())
        } else {
            None
        };
        prev_nlsf = Some(decoded.nlsf_q15);
        prev_nlsf_len = decoded.d_lpc;
        first = false;
        predictions.push(decoded);
    }

    // §5.1.5 finalize; §3.2 code-0 framing = TOC byte + the single
    // compressed frame. R2: a frame may not exceed 1275 bytes.
    let body = re.finish();
    if body.len() > 1275 {
        return Err(Error::MalformedPacket);
    }
    let mut packet = Vec::with_capacity(1 + body.len());
    packet.push(toc);
    packet.extend_from_slice(&body);
    Ok((packet, predictions, lbrr_predictions))
}

/// One 20 ms time interval's symbol scripts for a **stereo** SILK-only
/// packet (§4.2.2): the mid-channel SILK frame plus, when the side
/// channel is coded for this interval, the side-channel SILK frame.
///
/// Consistency rules enforced by [`encode_silk_only_packet_stereo`]
/// (mirroring the decode-side §4.2.7.1 / §4.2.7.2 gating):
///
/// * `mid.header.stereo` must be `Some` (the §4.2.7.1 weights ride on
///   every mid frame); `side.header.stereo` must be `None`.
/// * The interval's **side VAD flag** is derived as "side coded with an
///   active frame type (`2..=5`)". When it is set, the §4.2.7.2
///   mid-only flag is *absent* (`mid.header.mid_only_flag == None`).
///   When it is unset, the flag is *present* and must equal
///   `side.is_none()`: `Some(true)` skips the side frame entirely,
///   `Some(false)` codes an inactive side frame (frame type `0..=1`).
#[derive(Debug, Clone, Copy)]
pub struct StereoIntervalScripts<'a> {
    /// The mid-channel frame script (carries the §4.2.7.1 weights and,
    /// when signalled, the §4.2.7.2 mid-only flag).
    pub mid: SilkFrameSymbols<'a>,
    /// The side-channel frame script, or `None` when the side channel
    /// is not coded for this interval (mid-only).
    pub side: Option<SilkFrameSymbols<'a>>,
}

/// The per-frame predictions returned by
/// [`encode_silk_only_packet_stereo`]: what a fresh
/// [`crate::decoder::OpusDecoder`] will reconstruct for each channel.
#[derive(Debug, Clone)]
pub struct StereoPacketPredictions {
    /// One decoded mid frame per 20 ms interval.
    pub mid: Vec<SilkFrameDecoded>,
    /// One decoded side frame per interval; `None` where the side
    /// channel is not coded (§4.2.7.2 mid-only).
    pub side: Vec<Option<SilkFrameDecoded>>,
}

/// Encode one **stereo, SILK-only, code-0** Opus packet from
/// per-interval mid/side symbol scripts (§3.1 / §4.2.2-§4.2.6).
///
/// Per 20 ms interval (or the single 10 ms interval) the mid SILK frame
/// is written, then — unless the interval is mid-only — the side SILK
/// frame, in the §4.2.2 interleaved order, with two independent
/// per-channel carried states (previous gain / lag / NLSF) threaded
/// exactly the way the packet decoder threads them. The §4.2.3 header
/// bits carry both channels' VAD flags: the mid VAD is derived from the
/// mid frame type, the side VAD from "side coded with an active frame
/// type" (see [`StereoIntervalScripts`]).
///
/// Returns the packet bytes plus the per-channel
/// [`StereoPacketPredictions`].
pub fn encode_silk_only_packet_stereo(
    bandwidth: Bandwidth,
    frame_size_tenths_ms: u16,
    intervals: &[StereoIntervalScripts<'_>],
) -> Result<(Vec<u8>, StereoPacketPredictions), Error> {
    let n = silk_frame_count(frame_size_tenths_ms).ok_or(Error::MalformedPacket)? as usize;
    let no_lbrr = vec![StereoIntervalLbrr::default(); n];
    let (packet, regular, _) = encode_silk_only_packet_stereo_with_lbrr(
        bandwidth,
        frame_size_tenths_ms,
        intervals,
        &no_lbrr,
    )?;
    Ok((packet, regular))
}

/// One interval's §4.2.5 LBRR (in-band FEC) scripts for
/// [`encode_silk_only_packet_stereo_with_lbrr`]: an optional mid-channel
/// redundancy frame and an optional side-channel redundancy frame.
///
/// Consistency rules (mirroring the decode-side LBRR walk):
///
/// * Both scripts must be active-coded (frame type `2..=5`, §4.2.7.3).
/// * An LBRR **mid** script carries the §4.2.7.1 weights
///   (`header.stereo` must be `Some`) and, when the interval has no
///   side LBRR frame, the §4.2.7.2 mid-only flag, which must be
///   `Some(true)` (a coded side LBRR frame is forbidden by a set
///   mid-only flag, and the header LBRR bits already say none follows).
///   With a side LBRR present the flag is absent (`None`).
/// * A **side-only** LBRR interval (mid `None`, side `Some`) is legal:
///   no weights ride on a side frame (§4.2.7.1).
#[derive(Debug, Clone, Copy, Default)]
pub struct StereoIntervalLbrr<'a> {
    /// Mid-channel LBRR script for this interval, if any.
    pub mid: Option<SilkFrameSymbols<'a>>,
    /// Side-channel LBRR script for this interval, if any.
    pub side: Option<SilkFrameSymbols<'a>>,
}

/// [`encode_silk_only_packet_stereo`] with §4.2.5 LBRR (in-band FEC)
/// emission: `lbrr[idx]` optionally carries mid / side redundancy
/// scripts for interval `idx` (re-encodes of the interval the
/// *previous* packet covered, per §2.1.7). LBRR frames are written
/// ahead of the regular frames in the same §4.2.2 mid/side interleaved
/// order, with their own independent per-channel carried states,
/// exactly the way the packet decoder walks them.
///
/// Returns the packet bytes, the regular predictions, and the LBRR
/// predictions.
pub fn encode_silk_only_packet_stereo_with_lbrr(
    bandwidth: Bandwidth,
    frame_size_tenths_ms: u16,
    intervals: &[StereoIntervalScripts<'_>],
    lbrr: &[StereoIntervalLbrr<'_>],
) -> Result<(Vec<u8>, StereoPacketPredictions, StereoLbrrPredictions), Error> {
    let num_silk_frames = silk_frame_count(frame_size_tenths_ms).ok_or(Error::MalformedPacket)?;
    if intervals.len() != num_silk_frames as usize || lbrr.len() != num_silk_frames as usize {
        return Err(Error::MalformedPacket);
    }
    let frame_size = if frame_size_tenths_ms == 100 {
        SilkFrameSize::TenMs
    } else {
        SilkFrameSize::TwentyMs
    };

    // §3.1 TOC byte: SILK-only, stereo, code 0 (one Opus frame).
    let toc = OpusTocByte::compose_byte(
        Mode::SilkOnly,
        bandwidth,
        frame_size_tenths_ms,
        true,
        FrameCountCode::One,
    )?;

    let mut re = RangeEncoder::new();

    // §4.2.3 / §4.2.4 header bits. VAD per channel per interval:
    // mid from the mid frame type; side from "side coded with an
    // active frame type". LBRR bitmaps from which intervals carry
    // redundancy scripts.
    let mut mid_vad_flags = 0u8;
    let mut side_vad_flags = 0u8;
    for (idx, iv) in intervals.iter().enumerate() {
        if iv.mid.header.frame_type >= 2 {
            mid_vad_flags |= 1 << idx;
        }
        if let Some(side) = &iv.side {
            if side.header.frame_type >= 2 {
                side_vad_flags |= 1 << idx;
            }
        }
    }
    let mut mid_lbrr_bits = 0u8;
    let mut side_lbrr_bits = 0u8;
    for (idx, l) in lbrr.iter().enumerate() {
        if l.mid.is_some() {
            mid_lbrr_bits |= 1 << idx;
        }
        if l.side.is_some() {
            side_lbrr_bits |= 1 << idx;
        }
    }
    let header = SilkHeaderBits {
        num_silk_frames,
        mid: SilkChannelHeader {
            vad_flags: mid_vad_flags,
            lbrr_flag: mid_lbrr_bits != 0,
        },
        side: Some(SilkChannelHeader {
            vad_flags: side_vad_flags,
            lbrr_flag: side_lbrr_bits != 0,
        }),
        per_frame_lbrr: PerFrameLbrr {
            mid: mid_lbrr_bits,
            side: side_lbrr_bits,
        },
    };
    header.encode(&mut re)?;

    // §4.2.5 LBRR frames: per interval, mid (if present) then side (if
    // present), interleaved per §4.2.2, ahead of every regular frame,
    // with independent per-channel carried states — the exact mirror of
    // the decoder's stereo LBRR walk.
    let mut lbrr_mid_state = ChannelDecodeState::new();
    let mut lbrr_side_state = ChannelDecodeState::new();
    let mut lbrr_mid_pred: Vec<Option<SilkFrameDecoded>> = Vec::with_capacity(lbrr.len());
    let mut lbrr_side_pred: Vec<Option<SilkFrameDecoded>> = Vec::with_capacity(lbrr.len());
    for entry in lbrr.iter() {
        let side_lbrr = entry.side.is_some();
        if let Some(mid_sym) = &entry.mid {
            if mid_sym.header.frame_type < 2 {
                // §4.2.7.3: LBRR frames use the active PDF.
                return Err(Error::MalformedPacket);
            }
            // §4.2.7.2: the mid-only flag is present on the mid LBRR
            // frame iff the interval's side LBRR frame is not coded —
            // and then it must be set (the header LBRR bits already
            // promise no side frame follows; a cleared flag would
            // contradict them).
            if !side_lbrr && mid_sym.header.mid_only_flag != Some(true) {
                return Err(Error::MalformedPacket);
            }
            let stereo_ctx = StereoHeaderContext {
                has_mid_only_flag: !side_lbrr,
            };
            let decoded = encode_silk_frame(
                &mut re,
                lbrr_mid_state.config(bandwidth, frame_size, true, Some(stereo_ctx)),
                mid_sym,
            )?;
            lbrr_mid_state.advance(&decoded);
            lbrr_mid_pred.push(Some(decoded));
        } else {
            // §4.2.4 LBRR-flag gap for the mid channel.
            lbrr_mid_state.mark_interval_uncoded(true);
            lbrr_mid_pred.push(None);
        }
        if let Some(side_sym) = &entry.side {
            if side_sym.header.frame_type < 2 {
                return Err(Error::MalformedPacket);
            }
            let decoded = encode_silk_frame(
                &mut re,
                lbrr_side_state.config(bandwidth, frame_size, true, None),
                side_sym,
            )?;
            lbrr_side_state.advance(&decoded);
            lbrr_side_pred.push(Some(decoded));
        } else {
            // §4.2.4 LBRR-flag gap for the side channel.
            lbrr_side_state.mark_interval_uncoded(true);
            lbrr_side_pred.push(None);
        }
    }

    // §4.2.6 regular SILK frames: per interval, the mid frame then
    // (unless mid-only) the side frame, with per-channel carried state.
    let mut mid_state = ChannelDecodeState::new();
    let mut side_state = ChannelDecodeState::new();
    let mut mid_pred: Vec<SilkFrameDecoded> = Vec::with_capacity(intervals.len());
    let mut side_pred: Vec<Option<SilkFrameDecoded>> = Vec::with_capacity(intervals.len());
    for (idx, iv) in intervals.iter().enumerate() {
        let side_active = (side_vad_flags >> idx) & 1 == 1;
        // §4.2.7.2: the mid-only flag is present iff the interval's side
        // channel is not active; the script must match, and its value
        // must agree with whether a side script is present.
        match iv.mid.header.mid_only_flag {
            Some(flag) => {
                if side_active || flag != iv.side.is_none() {
                    return Err(Error::MalformedPacket);
                }
            }
            None => {
                // No flag ⇒ side VAD must be set ⇒ side frame coded.
                if !side_active || iv.side.is_none() {
                    return Err(Error::MalformedPacket);
                }
            }
        }
        let stereo_ctx = StereoHeaderContext {
            has_mid_only_flag: !side_active,
        };
        let mid_decoded = encode_silk_frame(
            &mut re,
            mid_state.config(
                bandwidth,
                frame_size,
                (mid_vad_flags >> idx) & 1 == 1,
                Some(stereo_ctx),
            ),
            &iv.mid,
        )?;
        mid_state.advance(&mid_decoded);
        mid_pred.push(mid_decoded);

        if let Some(side_sym) = &iv.side {
            let side_decoded = encode_silk_frame(
                &mut re,
                side_state.config(bandwidth, frame_size, side_active, None),
                side_sym,
            )?;
            side_state.advance(&side_decoded);
            side_pred.push(Some(side_decoded));
        } else {
            // §4.2.7.2 / §4.5.2: an uncoded side frame resets the side
            // carried state — the next coded side frame codes its gains
            // independently with the clamp skipped (§4.2.7.4), its lag
            // absolutely (§4.2.7.6.1), and its LSF interpolation factor
            // forced to 4 (§4.2.7.5.5) — mirroring the decoder.
            side_state.mark_interval_uncoded(false);
            side_pred.push(None);
        }
    }

    // §5.1.5 finalize; §3.2 code-0 framing. R2: a frame may not exceed
    // 1275 bytes.
    let body = re.finish();
    if body.len() > 1275 {
        return Err(Error::MalformedPacket);
    }
    let mut packet = Vec::with_capacity(1 + body.len());
    packet.push(toc);
    packet.extend_from_slice(&body);
    Ok((
        packet,
        StereoPacketPredictions {
            mid: mid_pred,
            side: side_pred,
        },
        StereoLbrrPredictions {
            mid: lbrr_mid_pred,
            side: lbrr_side_pred,
        },
    ))
}

/// The per-interval §4.2.5 LBRR predictions returned by
/// [`encode_silk_only_packet_stereo_with_lbrr`].
#[derive(Debug, Clone)]
pub struct StereoLbrrPredictions {
    /// One decoded mid LBRR frame per interval that carried one.
    pub mid: Vec<Option<SilkFrameDecoded>>,
    /// One decoded side LBRR frame per interval that carried one.
    pub side: Vec<Option<SilkFrameDecoded>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{FrameDecodeStatus, OpusDecoder};
    use crate::range_decoder::RangeDecoder;
    use crate::silk_decode::decode_silk_frame;
    use crate::silk_excitation::{shell_block_count, ExcitationSymbols, SHELL_BLOCK_SAMPLES};
    use crate::silk_frame::SilkHeaderSymbols;
    use crate::silk_gains::GainSymbol;
    use crate::silk_ltp::{LagSymbols, LtpSymbols, LTP_MAX_SUBFRAMES};

    /// A tiny deterministic LCG for the packet-level sweeps.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
    }

    /// Owns the per-frame script buffers (the `SilkFrameSymbols` borrow
    /// slices).
    struct ScriptBufs {
        gains: Vec<GainSymbol>,
        i2: Vec<i8>,
        lsb_counts: Vec<u8>,
        e_raw: Vec<i32>,
        header: SilkHeaderSymbols,
        lsf_stage1: u8,
        lsf_interp_w_q2: Option<u8>,
        ltp: Option<LtpSymbols>,
        lcg_seed: u8,
        rate_level: u8,
    }

    fn random_frame_script(
        rng: &mut Lcg,
        bandwidth: Bandwidth,
        frame_size: SilkFrameSize,
        gains_independent: bool,
        scaling_present: bool,
        has_prev_lag: bool,
    ) -> ScriptBufs {
        let num_subframes = if frame_size == SilkFrameSize::TenMs {
            2usize
        } else {
            4
        };
        let frame_type = rng.below(6) as u8;
        let voiced = frame_type >= 4;
        let gains: Vec<GainSymbol> = (0..num_subframes)
            .map(|k| {
                if k == 0 && gains_independent {
                    GainSymbol::Independent(rng.below(64) as u8)
                } else {
                    GainSymbol::Delta(rng.below(41) as u8)
                }
            })
            .collect();
        let d_lpc = if bandwidth == Bandwidth::Wb { 16 } else { 10 };
        let i2: Vec<i8> = (0..d_lpc).map(|_| rng.below(21) as i8 - 10).collect();
        let ltp = voiced.then(|| {
            let lag_low_count = match bandwidth {
                Bandwidth::Nb => 4u32,
                Bandwidth::Mb => 6,
                _ => 8,
            };
            let lag = if has_prev_lag {
                if rng.below(2) == 0 {
                    LagSymbols::RelativeDelta {
                        delta_index: 1 + rng.below(20) as u8,
                    }
                } else {
                    LagSymbols::RelativeFallback {
                        lag_high: rng.below(32) as u8,
                        lag_low: rng.below(lag_low_count) as u8,
                    }
                }
            } else {
                LagSymbols::Absolute {
                    lag_high: rng.below(32) as u8,
                    lag_low: rng.below(lag_low_count) as u8,
                }
            };
            let contour_cells = match (bandwidth, num_subframes) {
                (Bandwidth::Nb, 2) => 3u32,
                (Bandwidth::Nb, 4) => 11,
                (_, 2) => 12,
                _ => 34,
            };
            let periodicity_index = rng.below(3) as u8;
            let filter_cells = [8u32, 16, 32][periodicity_index as usize];
            let mut filter_indices = [0u8; LTP_MAX_SUBFRAMES];
            for f in filter_indices.iter_mut().take(num_subframes) {
                *f = rng.below(filter_cells) as u8;
            }
            LtpSymbols {
                lag,
                contour_index: rng.below(contour_cells) as u8,
                periodicity_index,
                filter_indices,
                // Must match the frame's §4.2.7.6.3 presence context.
                ltp_scaling_index: scaling_present.then(|| rng.below(3) as u8),
            }
        });
        let blocks = shell_block_count(bandwidth, frame_size).unwrap();
        let total = blocks * SHELL_BLOCK_SAMPLES;
        let mut lsb_counts = vec![0u8; blocks];
        let mut e_raw = vec![0i32; total];
        for (b, lc) in lsb_counts.iter_mut().enumerate() {
            let lsbs = if rng.below(4) == 0 { 1 } else { 0 };
            *lc = lsbs as u8;
            let budget = rng.below(17);
            let base = b * SHELL_BLOCK_SAMPLES;
            let mut spent = 0u32;
            while spent < budget {
                let i = base + rng.below(16) as usize;
                let add = 1 + rng.below(budget - spent);
                e_raw[i] += (add << lsbs) as i32;
                spent += add;
            }
            for slot in e_raw[base..base + SHELL_BLOCK_SAMPLES].iter_mut() {
                if lsbs > 0 {
                    *slot += (rng.next_u32() & 1) as i32;
                }
                if *slot != 0 && rng.below(2) == 0 {
                    *slot = -*slot;
                }
            }
        }
        ScriptBufs {
            gains,
            i2,
            lsb_counts,
            e_raw,
            header: SilkHeaderSymbols {
                stereo: None,
                mid_only_flag: None,
                frame_type,
            },
            lsf_stage1: rng.below(32) as u8,
            lsf_interp_w_q2: (frame_size == SilkFrameSize::TwentyMs).then(|| rng.below(5) as u8),
            ltp,
            lcg_seed: rng.below(4) as u8,
            rate_level: rng.below(9) as u8,
        }
    }

    fn symbols_of(bufs: &ScriptBufs) -> SilkFrameSymbols<'_> {
        SilkFrameSymbols {
            header: bufs.header,
            gains: &bufs.gains,
            lsf_stage1: bufs.lsf_stage1,
            lsf_stage2_i2: &bufs.i2,
            lsf_interp_w_q2: bufs.lsf_interp_w_q2,
            ltp: bufs.ltp,
            lcg_seed: bufs.lcg_seed,
            excitation: ExcitationSymbols {
                rate_level: bufs.rate_level,
                lsb_counts: &bufs.lsb_counts,
                e_raw: &bufs.e_raw,
            },
        }
    }

    /// End-to-end: random mono SILK-only packets (10/20/40/60 ms, all
    /// bandwidths) produced by the packet encoder decode through a
    /// fresh `OpusDecoder::decode_packet` with a real-SILK-PCM status
    /// and the exact §3 sample count, and the frame-level symbols
    /// decode back (via a parallel `decode_silk_frame` walk) equal to
    /// the encoder's predictions.
    #[test]
    fn packet_encode_decodes_end_to_end() {
        let mut rng = Lcg(0x0AC4_E701);
        for round in 0..120 {
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let fs_tenths: u16 = [100u16, 200, 400, 600][rng.below(4) as usize];
            let frame_size = if fs_tenths == 100 {
                SilkFrameSize::TenMs
            } else {
                SilkFrameSize::TwentyMs
            };
            let n = silk_frame_count(fs_tenths).unwrap() as usize;
            let mut prev_voiced = false;
            let bufs: Vec<ScriptBufs> = (0..n)
                .map(|idx| {
                    let b = random_frame_script(
                        &mut rng,
                        bandwidth,
                        frame_size,
                        idx == 0,
                        idx == 0,
                        // §4.2.7.6.1: relative lag only after a coded
                        // VOICED frame in the same Opus frame.
                        idx > 0 && prev_voiced,
                    );
                    prev_voiced = b.header.frame_type >= 4;
                    b
                })
                .collect();
            let scripts: Vec<SilkFrameSymbols<'_>> = bufs.iter().map(symbols_of).collect();

            let (packet, predictions) =
                encode_silk_only_packet_mono(bandwidth, fs_tenths, &scripts)
                    .expect("packet encode");
            assert_eq!(predictions.len(), n);

            // The packet decodes end-to-end on a fresh decoder.
            let mut dec = OpusDecoder::new();
            let out = dec.decode_packet(&packet).expect("packet decode");
            assert_eq!(out.channels, 1, "round {round}");
            assert_eq!(out.frame_outcomes.len(), 1, "round {round}");
            assert_eq!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::SilkParamsDecoded,
                "round {round} bw={bandwidth:?} fs={fs_tenths}"
            );
            // §3: 48 kHz output samples for the Opus frame duration.
            assert_eq!(
                out.samples_per_channel() as u32,
                48_000 * fs_tenths as u32 / 10_000,
                "round {round}"
            );

            // Frame-level: a parallel Table-5 walk over the packet body
            // reconstructs exactly the encoder's predictions.
            let mut rd = RangeDecoder::new(&packet[1..]);
            let header = SilkHeaderBits::decode(&mut rd, n as u8, false).expect("header bits");
            assert!(!header.mid.lbrr_flag);
            let mut prev_gain: Option<u8> = None;
            let mut prev_lag: Option<i32> = None;
            let mut prev_nlsf: Option<[i16; crate::silk_lsf_stage2::D_LPC_MAX]> = None;
            let mut prev_nlsf_len = 0usize;
            let mut first = true;
            for (idx, expected) in predictions.iter().enumerate() {
                let cfg = SilkFrameConfig {
                    bandwidth,
                    frame_size,
                    voice_active: header.mid_vad(idx as u8),
                    first_subframe_independent: first || prev_gain.is_none(),
                    previous_log_gain: prev_gain,
                    previous_primary_lag: prev_lag,
                    ltp_scaling_present: first,
                    lsf_interp_after_reset: first || prev_nlsf.is_none(),
                    previous_nlsf_q15: prev_nlsf,
                    previous_nlsf_len: prev_nlsf_len,
                    stereo: None,
                };
                let decoded = decode_silk_frame(&mut rd, cfg).expect("frame decode");
                assert_eq!(&decoded, expected, "round {round} frame {idx}");
                prev_gain = Some(decoded.gains.last_log_gain());
                prev_lag = if decoded.ltp.is_voiced() {
                    Some(decoded.ltp.primary_lag())
                } else {
                    None
                };
                prev_nlsf = Some(decoded.nlsf_q15);
                prev_nlsf_len = decoded.d_lpc;
                first = false;
            }
            assert!(!rd.has_error());
        }
    }

    /// LBRR emission closes the FEC loop: a packet carrying §4.2.5
    /// redundancy scripts still decodes its regular frames end-to-end
    /// (the LBRR bits shift every later symbol, so this pins the
    /// range-coder alignment), `decode_packet_fec` recovers real audio
    /// (`Recovered`), and a no-LBRR packet reports `NoLbrr`.
    #[test]
    fn packet_encode_with_lbrr_fec_roundtrip() {
        use crate::decoder::FecDecodeStatus;
        let mut rng = Lcg(0xFEC0_0382);
        for round in 0..60 {
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let fs_tenths: u16 = [100u16, 200, 400, 600][rng.below(4) as usize];
            let frame_size = if fs_tenths == 100 {
                SilkFrameSize::TenMs
            } else {
                SilkFrameSize::TwentyMs
            };
            let n = silk_frame_count(fs_tenths).unwrap() as usize;
            let mut prev_voiced = false;
            let bufs: Vec<ScriptBufs> = (0..n)
                .map(|idx| {
                    let b = random_frame_script(
                        &mut rng,
                        bandwidth,
                        frame_size,
                        idx == 0,
                        idx == 0,
                        // §4.2.7.6.1: relative lag only after a coded
                        // VOICED frame in the same Opus frame.
                        idx > 0 && prev_voiced,
                    );
                    prev_voiced = b.header.frame_type >= 4;
                    b
                })
                .collect();
            let scripts: Vec<SilkFrameSymbols<'_>> = bufs.iter().map(symbols_of).collect();

            // Random non-empty LBRR subset. LBRR scripts must be
            // active-coded and follow the LBRR carried-state rules, so
            // build them with the same generator but force an active
            // frame type and first/subsequent shape per coded order.
            let mut which = vec![false; n];
            which[rng.below(n as u32) as usize] = true;
            for w in which.iter_mut() {
                if rng.below(2) == 0 {
                    *w = true;
                }
            }
            let mut lbrr_bufs: Vec<Option<ScriptBufs>> = Vec::with_capacity(n);
            let mut coded_first = true;
            let mut lbrr_prev_voiced = false;
            for &w in &which {
                if !w {
                    // §4.2.4 LBRR-flag gap: the carried state re-arms
                    // (independent gains, absolute lag, scaling present).
                    lbrr_bufs.push(None);
                    coded_first = true;
                    lbrr_prev_voiced = false;
                    continue;
                }
                let mut b = random_frame_script(
                    &mut rng,
                    bandwidth,
                    frame_size,
                    coded_first,
                    coded_first,
                    !coded_first && lbrr_prev_voiced,
                );
                // Force an active frame type (§4.2.7.3: LBRR is
                // active-coded) while keeping the generator's LTP shape
                // consistent with it.
                if b.header.frame_type < 2 {
                    b.header.frame_type += 2; // 0/1 -> 2/3 (unvoiced, no LTP)
                }
                lbrr_prev_voiced = b.header.frame_type >= 4;
                lbrr_bufs.push(Some(b));
                coded_first = false;
            }
            let lbrr_scripts: Vec<Option<SilkFrameSymbols<'_>>> = lbrr_bufs
                .iter()
                .map(|b| b.as_ref().map(symbols_of))
                .collect();

            let (packet, regular, lbrr_pred) = encode_silk_only_packet_mono_with_lbrr(
                bandwidth,
                fs_tenths,
                &scripts,
                &lbrr_scripts,
            )
            .expect("packet encode with lbrr");
            assert_eq!(regular.len(), n);
            assert_eq!(
                lbrr_pred.iter().filter(|p| p.is_some()).count(),
                which.iter().filter(|&&w| w).count()
            );

            // Regular decode still lands (range-coder alignment past
            // the LBRR frames).
            let mut dec = OpusDecoder::new();
            let out = dec.decode_packet(&packet).expect("packet decode");
            assert_eq!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::SilkParamsDecoded,
                "round {round} bw={bandwidth:?} fs={fs_tenths}"
            );

            // FEC recovery from the LBRR we emitted.
            let mut fec_dec = OpusDecoder::new();
            let rec = fec_dec.decode_packet_fec(&packet).expect("fec decode");
            assert_eq!(
                rec.status,
                FecDecodeStatus::Recovered,
                "round {round} bw={bandwidth:?} fs={fs_tenths}"
            );
            assert_eq!(
                rec.pcm.len() as u32,
                48_000 * fs_tenths as u32 / 10_000,
                "round {round}"
            );
        }

        // A no-LBRR packet reports NoLbrr.
        let bufs = random_frame_script(
            &mut rng,
            Bandwidth::Nb,
            SilkFrameSize::TwentyMs,
            true,
            true,
            false,
        );
        let script = symbols_of(&bufs);
        let (packet, _) =
            encode_silk_only_packet_mono(Bandwidth::Nb, 200, &[script]).expect("encode");
        let mut dec = OpusDecoder::new();
        let rec = dec.decode_packet_fec(&packet).expect("fec decode");
        assert_eq!(rec.status, FecDecodeStatus::NoLbrr);
    }

    use crate::silk_frame::StereoWeightSymbols;

    fn random_weights(rng: &mut Lcg) -> StereoWeightSymbols {
        StereoWeightSymbols {
            n: rng.below(25) as u8,
            i0: rng.below(3) as u8,
            i1: rng.below(5) as u8,
            i2: rng.below(3) as u8,
            i3: rng.below(5) as u8,
        }
    }

    /// Per-interval side-channel coding pattern for the stereo sweeps.
    #[derive(Clone, Copy, PartialEq)]
    enum SidePattern {
        /// Side coded with an active frame type (side VAD set, no
        /// §4.2.7.2 mid-only flag on the mid frame).
        Active,
        /// Side coded but inactive (side VAD clear, mid-only flag
        /// present and cleared).
        InactiveCoded,
        /// Side not coded (mid-only flag present and set).
        MidOnly,
    }

    /// Build one stereo interval's (mid, side) script buffers.
    ///
    /// `mid_first` / `mid_has_prev_lag` describe the mid channel's
    /// carried state. The side channel decouples its gain-independence
    /// from the §4.2.7.6.3 scaling presence: after an uncoded side
    /// interval the next side frame codes gains independently
    /// (§4.2.7.4 "previous ... was not coded") and its lag absolutely,
    /// but the scaling field only ever rides the FIRST time interval
    /// of the Opus frame for regular frames.
    #[allow(clippy::too_many_arguments)]
    fn random_stereo_interval(
        rng: &mut Lcg,
        bandwidth: Bandwidth,
        frame_size: SilkFrameSize,
        pattern: SidePattern,
        mid_first: bool,
        mid_has_prev_lag: bool,
        side_gains_independent: bool,
        side_scaling_present: bool,
        side_has_prev_lag: bool,
    ) -> (ScriptBufs, Option<ScriptBufs>) {
        let mut mid = random_frame_script(
            rng,
            bandwidth,
            frame_size,
            mid_first,
            mid_first,
            mid_has_prev_lag,
        );
        mid.header.stereo = Some(random_weights(rng));
        mid.header.mid_only_flag = match pattern {
            SidePattern::Active => None,
            SidePattern::InactiveCoded => Some(false),
            SidePattern::MidOnly => Some(true),
        };
        let side = match pattern {
            SidePattern::MidOnly => None,
            SidePattern::Active => {
                let mut s = random_frame_script(
                    rng,
                    bandwidth,
                    frame_size,
                    side_gains_independent,
                    side_scaling_present,
                    side_has_prev_lag,
                );
                if s.header.frame_type < 2 {
                    // Force an active type without disturbing the
                    // generator's LTP shape (0/1 → 2/3, still unvoiced).
                    s.header.frame_type += 2;
                }
                Some(s)
            }
            SidePattern::InactiveCoded => {
                let mut s = random_frame_script(
                    rng,
                    bandwidth,
                    frame_size,
                    side_gains_independent,
                    side_scaling_present,
                    side_has_prev_lag,
                );
                if s.header.frame_type >= 2 {
                    // Force an inactive type; inactive frames carry no
                    // §4.2.7.6 LTP.
                    s.header.frame_type %= 2;
                    s.ltp = None;
                }
                Some(s)
            }
        };
        (mid, side)
    }

    /// End-to-end: random stereo SILK-only packets across every
    /// bandwidth × duration × side-coding pattern decode through a fresh
    /// `OpusDecoder::decode_packet` to real stereo SILK PCM with the
    /// exact §3 sample count, and a parallel mid/side Table-5 walk
    /// reconstructs both channels' predictions field-for-field.
    #[test]
    fn stereo_packet_encode_decodes_end_to_end() {
        let mut rng = Lcg(0x57E2_E001);
        for round in 0..120 {
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let fs_tenths: u16 = [100u16, 200, 400, 600][rng.below(4) as usize];
            let frame_size = if fs_tenths == 100 {
                SilkFrameSize::TenMs
            } else {
                SilkFrameSize::TwentyMs
            };
            let n = silk_frame_count(fs_tenths).unwrap() as usize;

            let mut interval_bufs: Vec<(ScriptBufs, Option<ScriptBufs>)> = Vec::with_capacity(n);
            let mut patterns: Vec<SidePattern> = Vec::with_capacity(n);
            let mut mid_prev_voiced = false;
            let mut side_indep = true;
            let mut side_prev_voiced = false;
            for idx in 0..n {
                let pattern = match rng.below(3) {
                    0 => SidePattern::Active,
                    1 => SidePattern::InactiveCoded,
                    _ => SidePattern::MidOnly,
                };
                let iv = random_stereo_interval(
                    &mut rng,
                    bandwidth,
                    frame_size,
                    pattern,
                    idx == 0,
                    idx > 0 && mid_prev_voiced,
                    side_indep,
                    // §4.2.7.6.3: regular frames only carry the scaling
                    // field in the FIRST time interval.
                    idx == 0,
                    side_prev_voiced,
                );
                mid_prev_voiced = iv.0.header.frame_type >= 4;
                match &iv.1 {
                    Some(sf) => {
                        side_indep = false;
                        side_prev_voiced = sf.header.frame_type >= 4;
                    }
                    None => {
                        // Uncoded side interval: fresh gain/lag state.
                        side_indep = true;
                        side_prev_voiced = false;
                    }
                }
                patterns.push(pattern);
                interval_bufs.push(iv);
            }
            let intervals: Vec<StereoIntervalScripts<'_>> = interval_bufs
                .iter()
                .map(|(m, s)| StereoIntervalScripts {
                    mid: symbols_of(m),
                    side: s.as_ref().map(symbols_of),
                })
                .collect();

            let (packet, predictions) =
                encode_silk_only_packet_stereo(bandwidth, fs_tenths, &intervals)
                    .expect("stereo packet encode");
            assert_eq!(predictions.mid.len(), n);
            assert_eq!(predictions.side.len(), n);

            // The packet decodes end-to-end on a fresh decoder.
            let mut dec = OpusDecoder::new();
            let out = dec.decode_packet(&packet).expect("packet decode");
            assert_eq!(out.channels, 2, "round {round}");
            assert_eq!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::SilkStereoDecoded,
                "round {round} bw={bandwidth:?} fs={fs_tenths}"
            );
            assert_eq!(
                out.samples_per_channel() as u32,
                48_000 * fs_tenths as u32 / 10_000,
                "round {round}"
            );
            assert_eq!(
                out.pcm.len() as u32,
                2 * 48_000 * fs_tenths as u32 / 10_000,
                "round {round}"
            );

            // Parallel mid/side Table-5 walk over the packet body
            // reconstructs exactly the encoder's predictions.
            let mut rd = RangeDecoder::new(&packet[1..]);
            let header = SilkHeaderBits::decode(&mut rd, n as u8, true).expect("header bits");
            assert!(!header.mid.lbrr_flag);
            assert!(header.side.is_some_and(|s| !s.lbrr_flag));
            let mut mid_state = crate::decoder::ChannelDecodeState::new();
            let mut side_state = crate::decoder::ChannelDecodeState::new();
            for (idx, pattern) in patterns.iter().enumerate() {
                let side_active = header.side_vad(idx as u8);
                assert_eq!(
                    side_active,
                    *pattern == SidePattern::Active,
                    "round {round} interval {idx}"
                );
                let stereo_ctx = crate::silk_decode::StereoHeaderContext {
                    has_mid_only_flag: !side_active,
                };
                let mid_decoded = decode_silk_frame(
                    &mut rd,
                    mid_state.config(
                        bandwidth,
                        frame_size,
                        header.mid_vad(idx as u8),
                        Some(stereo_ctx),
                    ),
                )
                .expect("mid decode");
                assert_eq!(
                    &mid_decoded, &predictions.mid[idx],
                    "round {round} mid {idx}"
                );
                let side_coded = side_active || mid_decoded.mid_only_flag == Some(false);
                mid_state.advance(&mid_decoded);
                if side_coded {
                    let side_decoded = decode_silk_frame(
                        &mut rd,
                        side_state.config(bandwidth, frame_size, side_active, None),
                    )
                    .expect("side decode");
                    assert_eq!(
                        Some(&side_decoded),
                        predictions.side[idx].as_ref(),
                        "round {round} side {idx}"
                    );
                    side_state.advance(&side_decoded);
                } else {
                    assert!(predictions.side[idx].is_none(), "round {round} side {idx}");
                    // §4.2.7.2: an uncoded side interval resets the side
                    // carried state — mirror the decoder walk.
                    side_state.mark_interval_uncoded(false);
                }
            }
            assert!(!rd.has_error());
        }
    }

    /// Stereo LBRR emission closes the stereo FEC loop: a packet
    /// carrying §4.2.5 mid/side redundancy still decodes its regular
    /// frames end-to-end (pinning the range-coder alignment across the
    /// interleaved LBRR frames), and `decode_packet_fec` recovers real
    /// two-channel audio from the emitted redundancy.
    #[test]
    fn stereo_packet_encode_with_lbrr_fec_roundtrip() {
        use crate::decoder::FecDecodeStatus;
        let mut rng = Lcg(0xFEC0_57E2);
        for round in 0..60 {
            let bandwidth = match rng.below(3) {
                0 => Bandwidth::Nb,
                1 => Bandwidth::Mb,
                _ => Bandwidth::Wb,
            };
            let fs_tenths: u16 = [100u16, 200, 400, 600][rng.below(4) as usize];
            let frame_size = if fs_tenths == 100 {
                SilkFrameSize::TenMs
            } else {
                SilkFrameSize::TwentyMs
            };
            let n = silk_frame_count(fs_tenths).unwrap() as usize;

            // Regular frames: keep the side always actively coded for
            // this sweep (the pattern axis is covered by the other test).
            let mut interval_bufs: Vec<(ScriptBufs, Option<ScriptBufs>)> = Vec::with_capacity(n);
            let mut mid_prev_voiced = false;
            let mut side_prev_voiced = false;
            for idx in 0..n {
                let iv = random_stereo_interval(
                    &mut rng,
                    bandwidth,
                    frame_size,
                    SidePattern::Active,
                    idx == 0,
                    idx > 0 && mid_prev_voiced,
                    idx == 0,
                    idx == 0,
                    idx > 0 && side_prev_voiced,
                );
                mid_prev_voiced = iv.0.header.frame_type >= 4;
                side_prev_voiced = iv.1.as_ref().is_some_and(|sf| sf.header.frame_type >= 4);
                interval_bufs.push(iv);
            }
            let intervals: Vec<StereoIntervalScripts<'_>> = interval_bufs
                .iter()
                .map(|(m, s)| StereoIntervalScripts {
                    mid: symbols_of(m),
                    side: s.as_ref().map(symbols_of),
                })
                .collect();

            // LBRR pattern per interval: none / mid-only / mid+side /
            // side-only, at least one interval carrying something.
            let mut kinds = vec![0u32; n];
            kinds[rng.below(n as u32) as usize] = 1 + rng.below(3);
            for k in kinds.iter_mut() {
                if *k == 0 && rng.below(2) == 0 {
                    *k = 1 + rng.below(3);
                }
            }
            let mut lbrr_bufs: Vec<(Option<ScriptBufs>, Option<ScriptBufs>)> =
                Vec::with_capacity(n);
            let mut mid_first = true;
            let mut mid_prev = false;
            let mut side_first = true;
            let mut side_prev = false;
            for &kind in &kinds {
                let want_mid = kind == 1 || kind == 2;
                let want_side = kind == 2 || kind == 3;
                let mid = if want_mid {
                    let mut b = random_frame_script(
                        &mut rng,
                        bandwidth,
                        frame_size,
                        mid_first,
                        mid_first,
                        !mid_first && mid_prev,
                    );
                    if b.header.frame_type < 2 {
                        b.header.frame_type += 2; // active-coded (§4.2.7.3)
                    }
                    b.header.stereo = Some(random_weights(&mut rng));
                    b.header.mid_only_flag = (!want_side).then_some(true);
                    mid_first = false;
                    mid_prev = b.header.frame_type >= 4;
                    Some(b)
                } else {
                    // §4.2.4 LBRR-flag gap re-arms the mid LBRR state.
                    mid_first = true;
                    mid_prev = false;
                    None
                };
                let side = if want_side {
                    let mut b = random_frame_script(
                        &mut rng,
                        bandwidth,
                        frame_size,
                        side_first,
                        side_first,
                        !side_first && side_prev,
                    );
                    if b.header.frame_type < 2 {
                        b.header.frame_type += 2;
                    }
                    side_first = false;
                    side_prev = b.header.frame_type >= 4;
                    Some(b)
                } else {
                    side_first = true;
                    side_prev = false;
                    None
                };
                lbrr_bufs.push((mid, side));
            }
            let lbrr_scripts: Vec<StereoIntervalLbrr<'_>> = lbrr_bufs
                .iter()
                .map(|(m, s)| StereoIntervalLbrr {
                    mid: m.as_ref().map(symbols_of),
                    side: s.as_ref().map(symbols_of),
                })
                .collect();

            let (packet, regular, lbrr_pred) = encode_silk_only_packet_stereo_with_lbrr(
                bandwidth,
                fs_tenths,
                &intervals,
                &lbrr_scripts,
            )
            .expect("stereo packet encode with lbrr");
            assert_eq!(regular.mid.len(), n);
            assert_eq!(
                lbrr_pred.mid.iter().filter(|p| p.is_some()).count(),
                kinds.iter().filter(|&&k| k == 1 || k == 2).count()
            );
            assert_eq!(
                lbrr_pred.side.iter().filter(|p| p.is_some()).count(),
                kinds.iter().filter(|&&k| k == 2 || k == 3).count()
            );

            // Regular decode still lands (range-coder alignment past the
            // interleaved LBRR frames).
            let mut dec = OpusDecoder::new();
            let out = dec.decode_packet(&packet).expect("packet decode");
            assert_eq!(
                out.frame_outcomes[0].status,
                FrameDecodeStatus::SilkStereoDecoded,
                "round {round} bw={bandwidth:?} fs={fs_tenths} kinds={kinds:?}"
            );

            // FEC recovery from the emitted redundancy: real two-channel
            // audio with the exact §3 sample count. The decoder's FEC
            // policy needs at least one *mid* LBRR frame to reconstruct a
            // stereo signal (a side-only redundancy set has no mid signal
            // to unmix against), so an all-side-only pattern reports
            // `NoLbrr` while every mid-bearing pattern recovers.
            let any_mid_lbrr = kinds.iter().any(|&k| k == 1 || k == 2);
            let mut fec_dec = OpusDecoder::new();
            let rec = fec_dec.decode_packet_fec(&packet).expect("fec decode");
            assert_eq!(
                rec.status,
                if any_mid_lbrr {
                    FecDecodeStatus::Recovered
                } else {
                    FecDecodeStatus::NoLbrr
                },
                "round {round} bw={bandwidth:?} fs={fs_tenths} kinds={kinds:?}"
            );
            assert_eq!(rec.channels, 2, "round {round}");
            assert_eq!(
                rec.pcm.len() as u32,
                2 * 48_000 * fs_tenths as u32 / 10_000,
                "round {round}"
            );
        }
    }

    /// Stereo shape / consistency violations are rejected: wrong
    /// interval count, missing §4.2.7.1 weights, mid-only flag
    /// mismatches, an active side frame under a mid-only interval, and
    /// inactive LBRR scripts.
    #[test]
    fn stereo_packet_encode_rejects_inconsistent_scripts() {
        let mut rng = Lcg(0xBAD5_7E2E);
        let (mid, side) = random_stereo_interval(
            &mut rng,
            Bandwidth::Wb,
            SilkFrameSize::TwentyMs,
            SidePattern::Active,
            true,
            false,
            true,
            true,
            false,
        );
        let side = side.unwrap();

        // Wrong interval count for the duration.
        let iv = StereoIntervalScripts {
            mid: symbols_of(&mid),
            side: Some(symbols_of(&side)),
        };
        assert!(encode_silk_only_packet_stereo(Bandwidth::Wb, 400, &[iv]).is_err());

        // Missing stereo weights on the mid frame.
        let mut mid_no_w = symbols_of(&mid);
        mid_no_w.header.stereo = None;
        let iv = StereoIntervalScripts {
            mid: mid_no_w,
            side: Some(symbols_of(&side)),
        };
        assert!(encode_silk_only_packet_stereo(Bandwidth::Wb, 200, &[iv]).is_err());

        // Active side but a mid-only flag present on the mid frame.
        let mut mid_bad_flag = symbols_of(&mid);
        mid_bad_flag.header.mid_only_flag = Some(false);
        let iv = StereoIntervalScripts {
            mid: mid_bad_flag,
            side: Some(symbols_of(&side)),
        };
        assert!(encode_silk_only_packet_stereo(Bandwidth::Wb, 200, &[iv]).is_err());

        // Mid-only flag set but a side script supplied. The side script
        // must be inactive for the "flag value vs side presence" check
        // to be the failing condition (an active side sets side VAD,
        // which is the previous case).
        let (mid_mo, _) = random_stereo_interval(
            &mut rng,
            Bandwidth::Wb,
            SilkFrameSize::TwentyMs,
            SidePattern::MidOnly,
            true,
            false,
            true,
            true,
            false,
        );
        let mut side_inactive = random_frame_script(
            &mut rng,
            Bandwidth::Wb,
            SilkFrameSize::TwentyMs,
            true,
            true,
            false,
        );
        if side_inactive.header.frame_type >= 2 {
            side_inactive.header.frame_type %= 2;
            side_inactive.ltp = None;
        }
        let iv = StereoIntervalScripts {
            mid: symbols_of(&mid_mo),
            side: Some(symbols_of(&side_inactive)),
        };
        assert!(encode_silk_only_packet_stereo(Bandwidth::Wb, 200, &[iv]).is_err());

        // Stereo weights on a *side* frame are rejected by the
        // per-frame encoder (want_stereo mismatch).
        let mut side_with_w = symbols_of(&side);
        side_with_w.header.stereo = Some(random_weights(&mut rng));
        let iv = StereoIntervalScripts {
            mid: symbols_of(&mid),
            side: Some(side_with_w),
        };
        assert!(encode_silk_only_packet_stereo(Bandwidth::Wb, 200, &[iv]).is_err());

        // Inactive LBRR mid script.
        let (mid_ok, side_ok) = random_stereo_interval(
            &mut rng,
            Bandwidth::Wb,
            SilkFrameSize::TwentyMs,
            SidePattern::Active,
            true,
            false,
            true,
            true,
            false,
        );
        let side_ok = side_ok.unwrap();
        let mut lbrr_mid = random_frame_script(
            &mut rng,
            Bandwidth::Wb,
            SilkFrameSize::TwentyMs,
            true,
            true,
            false,
        );
        if lbrr_mid.header.frame_type >= 2 {
            lbrr_mid.header.frame_type %= 2;
            lbrr_mid.ltp = None;
        }
        lbrr_mid.header.stereo = Some(random_weights(&mut rng));
        lbrr_mid.header.mid_only_flag = Some(true);
        let iv = StereoIntervalScripts {
            mid: symbols_of(&mid_ok),
            side: Some(symbols_of(&side_ok)),
        };
        let lbrr = StereoIntervalLbrr {
            mid: Some(symbols_of(&lbrr_mid)),
            side: None,
        };
        assert!(
            encode_silk_only_packet_stereo_with_lbrr(Bandwidth::Wb, 200, &[iv], &[lbrr]).is_err()
        );

        // LBRR mid with no side LBRR must carry mid_only == Some(true).
        let mut lbrr_mid_bad = random_frame_script(
            &mut rng,
            Bandwidth::Wb,
            SilkFrameSize::TwentyMs,
            true,
            true,
            false,
        );
        if lbrr_mid_bad.header.frame_type < 2 {
            lbrr_mid_bad.header.frame_type += 2;
        }
        lbrr_mid_bad.header.stereo = Some(random_weights(&mut rng));
        lbrr_mid_bad.header.mid_only_flag = Some(false);
        let iv = StereoIntervalScripts {
            mid: symbols_of(&mid_ok),
            side: Some(symbols_of(&side_ok)),
        };
        let lbrr = StereoIntervalLbrr {
            mid: Some(symbols_of(&lbrr_mid_bad)),
            side: None,
        };
        assert!(
            encode_silk_only_packet_stereo_with_lbrr(Bandwidth::Wb, 200, &[iv], &[lbrr]).is_err()
        );
    }

    /// §4.2.7.6.1 regression (round 388): a voiced frame whose
    /// PREVIOUS frame in the same Opus frame was coded but NOT voiced
    /// must code its primary lag absolutely — the walkers used to
    /// carry the unvoiced frame's zero lag and select relative coding
    /// (the wrong PDF, a parse-level divergence on real streams).
    #[test]
    fn voiced_after_unvoiced_uses_absolute_lag() {
        let mut rng = Lcg(0xAB5_01A6);
        let bandwidth = Bandwidth::Wb;
        let frame_size = SilkFrameSize::TwentyMs;

        // Frame 0: unvoiced. Frame 1: voiced with an ABSOLUTE lag.
        let mut f0 = random_frame_script(&mut rng, bandwidth, frame_size, true, true, false);
        if f0.header.frame_type >= 4 {
            f0.header.frame_type -= 2; // force unvoiced (2/3)
            f0.ltp = None;
        }
        let mut f1 = random_frame_script(&mut rng, bandwidth, frame_size, false, false, false);
        if f1.header.frame_type < 4 {
            f1.header.frame_type = 4; // force voiced
        }
        f1.ltp = Some(LtpSymbols {
            lag: LagSymbols::Absolute {
                lag_high: 12,
                lag_low: 3,
            },
            contour_index: 0,
            periodicity_index: 0,
            filter_indices: [0; LTP_MAX_SUBFRAMES],
            ltp_scaling_index: None,
        });
        // Frame 0 gain symbols must start independent; frame 1 delta.
        f0.gains[0] = GainSymbol::Independent(40);
        for g in f1.gains.iter_mut() {
            *g = GainSymbol::Delta(10);
        }

        let scripts = [symbols_of(&f0), symbols_of(&f1)];
        let (packet, predictions) =
            encode_silk_only_packet_mono(bandwidth, 400, &scripts).expect("absolute lag accepted");
        assert_eq!(predictions[1].ltp.primary_lag(), 12 * 8 + 3 + 32);

        let mut dec = OpusDecoder::new();
        let out = dec.decode_packet(&packet).expect("decode");
        assert_eq!(
            out.frame_outcomes[0].status,
            FrameDecodeStatus::SilkParamsDecoded
        );

        // The relative variants must now be REJECTED in this context
        // (the previous coded frame was not voiced ⇒ absolute context).
        let mut f1_rel = f1;
        f1_rel.ltp = Some(LtpSymbols {
            lag: LagSymbols::RelativeDelta { delta_index: 5 },
            contour_index: 0,
            periodicity_index: 0,
            filter_indices: [0; LTP_MAX_SUBFRAMES],
            ltp_scaling_index: None,
        });
        let scripts = [symbols_of(&f0), symbols_of(&f1_rel)];
        assert!(encode_silk_only_packet_mono(bandwidth, 400, &scripts).is_err());
    }

    /// Frame-count / duration mismatches are rejected.
    #[test]
    fn packet_encode_rejects_bad_shape() {
        let mut rng = Lcg(7);
        let bufs = random_frame_script(
            &mut rng,
            Bandwidth::Nb,
            SilkFrameSize::TwentyMs,
            true,
            true,
            false,
        );
        let script = symbols_of(&bufs);
        // 40 ms needs 2 frames.
        assert!(encode_silk_only_packet_mono(Bandwidth::Nb, 400, &[script]).is_err());
        // 2.5 ms is not a SILK duration.
        let script = symbols_of(&bufs);
        assert!(encode_silk_only_packet_mono(Bandwidth::Nb, 25, &[script]).is_err());
        // SWB is not a SILK bandwidth.
        let script = symbols_of(&bufs);
        assert!(encode_silk_only_packet_mono(Bandwidth::Swb, 200, &[script]).is_err());
    }
}
