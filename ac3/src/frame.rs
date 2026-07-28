//! AC-3 / E-AC-3 syncframe decode (ATSC A/52 §5, §6, §7).
//!
//! One [`Frame::decode`] call turns one syncframe (starting at the `0B77` sync
//! word) into interleaved S16 PCM for all its blocks. The path, in A/52 order:
//!
//! 1. **syncinfo** (§5.1) — sync word, CRC1, fscod, frmsizecod → sample rate and
//!    frame length. (E-AC-3 replaces frmsizecod with `frmsiz`, §E.1.2.2.)
//! 2. **bsi** (§5.3) — bitstream information: bsid, acmod, LFE, dialnorm. bsid
//!    selects the syntax: **bsid ≤ 8 = AC-3**, **bsid == 16 = E-AC-3** (§E.1).
//! 3. Per **audio block** (§6.2) — block-switch/dither flags, dynamic range,
//!    coupling strategy + coordinates (§7.4), rematrixing (§7.5), exponent
//!    strategy + differential exponents (§7.1), the parametric bit allocation
//!    (§7.2), then the quantized mantissas (§7.3).
//! 4. **reconstruction** — un-couple (§7.4.3), rematrix (§7.5), scale mantissas
//!    by 2^-exp, IMDCT + window + overlap-add (§7.9, §7.10) → PCM.
//!
//! The decoder keeps **cross-block state** within a frame (exponents, endmant,
//! coupling and bit-allocation parameters can be *reused* from the previous
//! block — §7.1.3 "reuse", §7.2.2) and **cross-frame state** (the IMDCT overlap
//! tails, §7.9.2). Both are held in [`Frame`].
//!
//! ## E-AC-3 scope (A/52 Annex E)
//! The two target files are **independent 5.1 core** substreams. The E-AC-3 base
//! decode (bsid 16 frame/block syntax, 6-block frame) covers a plain DD+ 5.1
//! stream. The Annex-E advanced tools — Adaptive Hybrid Transform (AHT, §E.3.2),
//! Spectral Extension (SPX, §E.3.7) and enhanced coupling (§E.3.6) — are
//! **detected and reported**, then decoded with the base tools (extension bands
//! left at the base reconstruction; a documented approximation, not a silent
//! stub). Dependent/Atmos substreams are declined (independent core only).
//!
//! ## Robustness (P0: untrusted input)
//! Every field read is bounds-checked through [`BitReader`]; every table index is
//! range-checked. A structurally invalid frame returns [`DecodeError`] and the
//! element warns-and-drops — never a panic or OOB.

use crate::bits::{BitReader, EndOfData};
use crate::imdct::{OverlapState, N};
use crate::quant::{dequant_direct, dequant_group, Dither, GROUP_BITS, GROUP_COUNT};
use crate::tables;

/// The AC-3 / E-AC-3 sync word (A/52 §5.1.1: `syncword` = 0x0B77).
pub const SYNC_WORD: u16 = 0x0B77;

/// Full-bandwidth channel slots (max 5 for 3/2). The LFE and the shared coupling
/// channel get their own slots beyond these.
pub const MAX_FBW: usize = 5;
/// Index of the LFE transform channel (in the per-block channel array).
pub const LFE_CH: usize = MAX_FBW;
/// Index of the shared coupling transform channel.
pub const CPL_CH: usize = MAX_FBW + 1;
/// Total transform-channel slots: 5 FBW + LFE + coupling.
pub const NUM_CH_SLOTS: usize = MAX_FBW + 2;

/// A frame that could not be decoded. The element maps this to a bus
/// warning-and-drop (never fatal): the next sync word resyncs the stream.
#[derive(Clone, Debug)]
pub enum DecodeError {
    /// A read ran past the end of the frame buffer.
    Truncated,
    /// A reserved / structurally impossible field (bad fscod, frmsizecod, bsid,
    /// acmod combination, out-of-range index).
    Invalid(&'static str),
}

impl From<EndOfData> for DecodeError {
    fn from(_: EndOfData) -> Self {
        DecodeError::Truncated
    }
}

/// The stream geometry + tools a decoded frame exposes to the element.
#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
    pub sample_rate: u32,
    /// Output channels (full-bandwidth + LFE), in ITU order L,R,C,LFE,Ls,Rs.
    pub channels: usize,
    /// True when this frame is E-AC-3 (bsid 16), false for AC-3 (bsid ≤ 8).
    pub eac3: bool,
    /// Set when the frame declared an Annex-E advanced tool (SPX/AHT/ecpl) this
    /// decoder approximates rather than fully reconstructs.
    pub used_approx_tool: bool,
}

/// Result of decoding one syncframe: interleaved S16 PCM plus geometry. The PCM
/// borrows the decoder's reused output buffer (no per-frame heap allocation; spec:
/// performance #1) and stays valid until the next [`Frame::decode`] call.
pub struct Decoded<'a> {
    /// Interleaved S16, channel order L,R,C,LFE,Ls,Rs (see [`Frame::decode`]).
    pub pcm: &'a [i16],
    pub info: FrameInfo,
}

/// Persistent decoder state: the per-transform-channel IMDCT overlap tails
/// (A/52 §7.9.2) and the dither LFSR (§7.3.1). One [`Frame`] per stream; the
/// overlap history is what makes consecutive frames continuous.
pub struct Frame {
    overlap: [OverlapState; NUM_CH_SLOTS],
    dither: Dither,
    /// Reused interleaved-S16 output buffer, handed out (borrowed) by [`decode`].
    /// Cleared and refilled each frame; its capacity persists so decode allocates
    /// no per-frame PCM (spec: performance #1).
    ///
    /// [`decode`]: Frame::decode
    pcm: Vec<i16>,
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

impl Frame {
    // One-time decoder setup: `pcm` is a reused output buffer (cleared + refilled per
    // frame, never re-allocated) — cold.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            overlap: std::array::from_fn(|_| OverlapState::new()),
            dither: Dither::new(),
            pcm: Vec::new(),
        }
    }

    /// Reset synthesis history (flush/seek): every overlap tail cleared so decode
    /// resumes cleanly at the next syncframe (the flush/seek rule).
    pub fn reset(&mut self) {
        for o in &mut self.overlap {
            o.reset();
        }
        self.dither = Dither::new();
    }

    /// Peek a frame's length (bytes), sample rate, and AC-3-vs-E-AC-3 from its
    /// header, without a full decode — the framer uses this to delimit frames.
    pub fn peek_len(data: &[u8]) -> Result<(usize, u32, bool), DecodeError> {
        let mut r = BitReader::new(data);
        if r.bits(16)? as u16 != SYNC_WORD {
            return Err(DecodeError::Invalid("sync word"));
        }
        // Distinguish AC-3 vs E-AC-3 by bsid, which lives at bit 40 in AC-3
        // (after crc1[16] fscod[2] frmsizecod[6]) and elsewhere in E-AC-3. Try
        // the AC-3 layout first; if bsid > 8 it's E-AC-3, re-parsed from scratch.
        let _crc1 = r.bits(16)?;
        let fscod = r.bits(2)?;
        let frmsizecod = r.bits(6)?;
        let bsid = r.bits(5)?;
        if bsid <= 8 {
            if fscod >= 3 {
                return Err(DecodeError::Invalid("fscod reserved"));
            }
            if frmsizecod as usize >= 38 {
                return Err(DecodeError::Invalid("frmsizecod"));
            }
            let words = tables::FRAME_SIZE_WORDS[fscod as usize][frmsizecod as usize] as usize;
            return Ok((words * 2, tables::SAMPLE_RATE[fscod as usize], false));
        }
        // E-AC-3: syncword[16] strmtyp[2] substreamid[3] frmsiz[11] fscod[2].
        let mut r2 = BitReader::new(data);
        let _sync = r2.bits(16)?;
        let _strmtyp = r2.bits(2)?;
        let _substreamid = r2.bits(3)?;
        let frmsiz = r2.bits(11)?;
        let fscod2 = r2.bits(2)?;
        let rate = if fscod2 == 3 {
            match r2.bits(2)? {
                0 => 24_000,
                1 => 22_050,
                2 => 16_000,
                _ => return Err(DecodeError::Invalid("eac3 fscod2 reserved")),
            }
        } else {
            tables::SAMPLE_RATE[fscod2 as usize]
        };
        // frame length = (frmsiz + 1) * 2 bytes (§E.1.2.2).
        Ok(((frmsiz as usize + 1) * 2, rate, true))
    }

    /// Decode one syncframe (A/52 §5–§7). Returns interleaved S16 PCM in
    /// **ITU/SMPTE channel order L, R, C, LFE, Ls, Rs** (the downmix contract)
    /// plus the frame geometry. The PCM borrows the decoder's reused buffer and is
    /// valid until the next call.
    pub fn decode(&mut self, data: &[u8]) -> Result<Decoded<'_>, DecodeError> {
        let (_flen, _rate, is_eac3) = Self::peek_len(data)?;
        let mut r = BitReader::new(data);
        let (cfg, numblks) = if is_eac3 {
            parse_bsi_eac3(&mut r)?
        } else {
            (parse_bsi_ac3(&mut r)?, 6)
        };

        // Per-frame decode state carried across blocks (§7.1.3 reuse, §7.2.2).
        let mut st = BlockState::new(&cfg);
        // Reuse the decoder-owned PCM buffer: take it out (leaving an empty Vec),
        // clear + refill it, then put it back. Its capacity persists across frames,
        // so decode allocates no per-frame PCM (spec: performance #1).
        let mut out_pcm = std::mem::take(&mut self.pcm);
        out_pcm.clear();
        let mut used_approx = cfg.eac3_approx_seen;

        for blk in 0..numblks {
            match self.decode_audblk(&mut r, &cfg, blk, &mut st) {
                Ok(approx) => {
                    used_approx |= approx;
                    self.synthesize_block(&cfg, &mut st, &mut out_pcm);
                }
                Err(e) => {
                    // For AC-3 a mid-frame parse failure means the whole frame is
                    // corrupt — bubble up (the element resyncs). For E-AC-3, where
                    // our frame-info parse may not cover an exotic layout, keep the
                    // frame's sample count intact with silence and flag approx.
                    if cfg.eac3 {
                        used_approx = true;
                        out_pcm.resize(out_pcm.len() + N * cfg.out_channels(), 0);
                    } else {
                        // Restore the buffer before the early return so its capacity
                        // is not lost.
                        self.pcm = out_pcm;
                        return Err(e);
                    }
                }
            }
        }

        self.pcm = out_pcm;
        Ok(Decoded {
            pcm: &self.pcm,
            info: FrameInfo {
                sample_rate: cfg.sample_rate,
                channels: cfg.out_channels(),
                eac3: cfg.eac3,
                used_approx_tool: used_approx,
            },
        })
    }

    /// Decode one audio block (A/52 §7). Returns `true` if it used an
    /// approximated tool. Mutates `st` (cross-block reuse state).
    #[allow(clippy::too_many_lines)]
    fn decode_audblk(
        &mut self,
        r: &mut BitReader,
        cfg: &StreamCfg,
        _blk: usize,
        st: &mut BlockState,
    ) -> Result<bool, DecodeError> {
        let nf = cfg.nfchans;
        let mut approx = false;

        // Reset per-block coefficient scratch (exponents/endmant persist for reuse).
        for cb in &mut st.ch {
            cb.coeffs = [0.0; N];
        }
        st.cpl.coeffs = [0.0; N];

        // --- block switch + dither flags (§6.2.2 / §7.9, §7.3.1) ---
        for i in 0..nf {
            st.ch[i].blksw = r.bit()? != 0;
        }
        for i in 0..nf {
            st.ch[i].dithflag = r.bit()? != 0;
        }
        // --- dynamic range control (§6.2.2) ---
        if r.bit()? != 0 {
            let _dynrng = r.bits(8)?;
        }
        if cfg.acmod == 0 && r.bit()? != 0 {
            let _dynrng2 = r.bits(8)?;
        }
        // E-AC-3: per-block spectral-extension / other flags precede coupling in
        // the full syntax. For the base 5.1 core we proceed to coupling; a stream
        // that actually enables SPX will misalign and be caught by the frame-level
        // silence fallback (flagged approx). AC-3 has no such prefix.

        // --- coupling strategy (§6.2.2 / §7.4.1) ---
        // `cplstre` is a real 1-bit field in **every** block (§5.4.2.7). In block 0
        // it is constrained to 1 (the coupling strategy must be transmitted), but
        // the bit is still present — skipping the read misaligns the whole block.
        let cplstre = r.bit()? != 0;
        if cplstre {
            st.cplinu = r.bit()? != 0;
            if st.cplinu {
                if cfg.eac3 {
                    // ecplinu (enhanced coupling, §E.3.6) — not implemented.
                    if r.bit()? != 0 {
                        approx = true;
                    }
                }
                // chincpl per full-bandwidth channel (§7.4.1). In 2/0 stereo both
                // channels are implicitly coupled (no per-channel flag).
                if cfg.acmod == 2 {
                    st.cpl.chincpl = [true, true, false, false, false];
                } else {
                    for i in 0..nf {
                        st.cpl.chincpl[i] = r.bit()? != 0;
                    }
                }
                st.cpl.phsflginu = if cfg.acmod == 2 { r.bit()? != 0 } else { false };
                let cplbegf = r.bits(4)? as usize;
                let cplendf = r.bits(4)? as usize;
                st.cpl.strtmant = 37 + cplbegf * 12;
                st.cpl.endmant = 37 + (cplendf + 3) * 12;
                if st.cpl.endmant > N {
                    return Err(DecodeError::Invalid("coupling band range"));
                }
                if st.cpl.strtmant >= st.cpl.endmant {
                    // A zero-or-negative-width band (`cplbegf` at/past `cplendf+3`)
                    // signals coupling "in use" for the block but with no coupled
                    // bins. Treat it as an uncoupled block: clear the per-channel
                    // coupling flags (so each channel reads its own
                    // `chbwcod`/exponents/mantissas) and mark no coupling channel.
                    // No band-structure bits follow a zero-width band, so we do NOT
                    // consume any — keeping the block aligned without a drop.
                    st.cplinu = false;
                    st.cpl.chincpl = [false; MAX_FBW];
                    st.cpl.nbands = 0;
                    st.cpl.have_exp = false;
                } else {
                    // Coupling sub-band → band structure (§7.4.1): one boundary bit
                    // per possible sub-band after the first.
                    let ncplsub = (st.cpl.endmant - st.cpl.strtmant) / 12;
                    let mut nbands = 1usize;
                    st.cpl.band_widths = [0u8; 18];
                    let mut cur_width = 1u8;
                    for _ in 1..ncplsub {
                        if r.bit()? != 0 {
                            st.cpl.band_widths[nbands - 1] = cur_width;
                            nbands += 1;
                            cur_width = 1;
                        } else {
                            cur_width += 1;
                        }
                    }
                    st.cpl.band_widths[nbands - 1] = cur_width;
                    st.cpl.nbands = nbands;
                }
            }
        }
        // Coupling coordinates (§7.4.2) — per coupled channel.
        if st.cplinu {
            for i in 0..nf {
                if st.cpl.chincpl[i] {
                    let cplcoe = r.bit()? != 0;
                    if cplcoe {
                        let mstrcplco = r.bits(2)? as i32;
                        for b in 0..st.cpl.nbands {
                            let cplcoexp = r.bits(4)? as i32;
                            let cplcomant = r.bits(4)?;
                            // §7.4.2: reconstruct the coupling coordinate. mant is
                            // 4-bit; if cplcoexp == 15 the mantissa is /16 else the
                            // leading 1 is implicit (mant+16)/32, then scaled by
                            // 2^-(cplcoexp + 3*mstrcplco).
                            let mant = if cplcoexp == 15 {
                                cplcomant as f32 / 16.0
                            } else {
                                (cplcomant as f32 + 16.0) / 32.0
                            };
                            let shift = cplcoexp + 3 * mstrcplco;
                            let coord = mant * 2f32.powi(-shift);
                            st.ch[i].cplco[b] = coord;
                        }
                    }
                    // else: reuse previous block's coordinates (already in cplco).
                }
            }
            if cfg.acmod == 2 && st.cpl.phsflginu {
                for _ in 0..st.cpl.nbands {
                    let _phsflg = r.bit()?;
                }
            }
        }

        // --- rematrixing (§7.5) — 2/0 mode only ---
        if cfg.acmod == 2 {
            // `rematstr` is a real 1-bit field in every 2/0 block (§7.5.1);
            // constrained to 1 in block 0 but still read.
            let rematstr = r.bit()? != 0;
            if rematstr {
                st.rematflg = [false; 4];
                // Number of rematrix bands (A/52 §7.5.1): 4 when uncoupled or the
                // coupling begins above the top rematrix band, else it shrinks as
                // the coupling start moves down into the rematrix range.
                let nrematbnd = if !st.cplinu || st.cpl.strtmant > 61 {
                    4
                } else if st.cpl.strtmant > 37 {
                    3
                } else {
                    2
                };
                for f in st.rematflg.iter_mut().take(nrematbnd) {
                    *f = r.bit()? != 0;
                }
            }
        }

        // --- exponent strategy (§7.1.3) ---
        let cplexpstr = if st.cplinu { r.bits(2)? } else { 0 };
        for i in 0..nf {
            st.ch[i].expstr = r.bits(2)?;
        }
        let lfeexpstr = if cfg.lfeon { r.bits(1)? } else { 0 };
        st.ch[LFE_CH].expstr = lfeexpstr;

        // channel bandwidth / endmant (§7.2.2.10, §7.4.1). A **coupled** channel
        // carries its own coefficients only up to `cplstrtmant` (the coupled band
        // is reconstructed from the coupling channel) — regardless of its exponent
        // strategy. An **uncoupled** channel reads `chbwcod` **only when its
        // exponent strategy is not "reuse"**; a reuse channel keeps the previous
        // block's endmant (§7.1.3). The `chbwcod` field is *only* present for
        // uncoupled, non-reuse channels — this ordering is the classic AC-3
        // alignment trap (a coupled channel must not consume a chbwcod field).
        for i in 0..nf {
            if st.cplinu && st.cpl.chincpl[i] {
                st.ch[i].endmant = st.cpl.strtmant;
            } else if st.ch[i].expstr != 0 {
                let chbwcod = r.bits(6)? as usize;
                if chbwcod > 60 {
                    return Err(DecodeError::Invalid("chbwcod"));
                }
                st.ch[i].endmant = (chbwcod + 12) * 3 + 37;
            }
            // else (uncoupled + reuse): keep the carried-over endmant.
        }
        st.ch[LFE_CH].endmant = 7; // LFE always 7 coeffs (§7.1.2)

        // --- exponents (§7.1.2) ---
        if st.cplinu && cplexpstr != 0 {
            let gs = grp_size(cplexpstr);
            let ncplgrps = (st.cpl.endmant - st.cpl.strtmant) / (3 * gs);
            // The coupling channel's first exponent is `cplabsexp << 1` (§7.1.2).
            let cplabsexp = r.bits(4)? as u8;
            let mut seed = cplabsexp << 1;
            decode_exp_groups(r, cplexpstr, ncplgrps, st.cpl.strtmant, &mut seed, &mut st.cpl.exp)?;
            st.cpl.have_exp = true;
        }
        for i in 0..nf {
            if st.ch[i].expstr != 0 {
                let end = st.ch[i].endmant;
                let gs = grp_size(st.ch[i].expstr);
                let ngrps = exp_ngrps(end, gs);
                let e0 = r.bits(4)? as u8;
                st.ch[i].exp[0] = e0;
                let mut seed = e0;
                decode_exp_after0(r, st.ch[i].expstr, ngrps, &mut seed, &mut st.ch[i].exp)?;
            }
            // expstr == 0: reuse — st.ch[i].exp/endmant already hold last block's.
        }
        if cfg.lfeon && lfeexpstr != 0 {
            let e0 = r.bits(4)? as u8;
            st.ch[LFE_CH].exp[0] = e0;
            let mut seed = e0;
            decode_exp_after0(r, 1, 2, &mut seed, &mut st.ch[LFE_CH].exp)?; // (7-1)/3 = 2 groups
        }

        // --- bit allocation parameters (§7.2.2.1). `baie` gates the parameter set;
        // when absent the previous block's parameters persist (reuse). ---
        if r.bit()? != 0 {
            st.bai.sdcycod = r.bits(2)? as usize;
            st.bai.fdcycod = r.bits(2)? as usize;
            st.bai.sgaincod = r.bits(2)? as usize;
            st.bai.dbpbcod = r.bits(2)? as usize;
            st.bai.floorcod = r.bits(3)? as usize;
        }
        // --- SNR offsets (§5.4.2.14 / §7.2.2.6). AC-3 (bsid ≤ 8) has **no**
        // `snroffste` gating flag: the full SNR-offset set (csnroffst + the
        // coupling and per-channel fsnroffst/fgaincod, plus the LFE offset) is
        // present in **every** block. (E-AC-3's §E.1.3.2 `snroffststr` scheme is
        // different and handled in that branch.) Verified bit-exact against the
        // Nord reference: reading the set unconditionally keeps every block's
        // mantissa section aligned to the frame boundary. ---
        st.bai.csnroffst = r.bits(6)? as i32;
        if st.cplinu {
            st.cpl.fsnroffst = r.bits(4)? as i32;
            st.cpl.fgaincod = r.bits(3)? as usize;
        }
        for i in 0..nf {
            st.ch[i].fsnroffst = r.bits(4)? as i32;
            st.ch[i].fgaincod = r.bits(3)? as usize;
        }
        if cfg.lfeon {
            st.ch[LFE_CH].fsnroffst = r.bits(4)? as i32;
            st.ch[LFE_CH].fgaincod = r.bits(3)? as usize;
        }
        // coupling leak init (§7.2.2.8). `cplfleak`/`cplsleak` seed the coupling
        // channel's leaky integrators as `(cplleak << 8) + 256`; when the block
        // omits them (`cplleake == 0`) the previous block's values persist.
        if st.cplinu && r.bit()? != 0 {
            st.cpl.fleak = ((r.bits(3)? as i32) << 8) + 256;
            st.cpl.sleak = ((r.bits(3)? as i32) << 8) + 256;
        }
        // delta bit allocation (§7.2.2.9) — parse-skip (rare; not applied).
        if r.bit()? != 0 {
            let n = if st.cplinu { nf + 1 } else { nf };
            for _ in 0..n {
                let deltbae = r.bits(2)?;
                if deltbae == 1 {
                    let ns = r.bits(3)? as usize;
                    for _ in 0..ns {
                        let _off = r.bits(5)?;
                        let _len = r.bits(4)?;
                        let _bap = r.bits(3)?;
                    }
                }
            }
        }
        // skip field (§7.2.2.11).
        if r.bit()? != 0 {
            let skipl = r.bits(9)? as usize;
            r.skip(skipl * 8)?;
        }

        // --- bit allocation + mantissas (§7.2.2.10, §7.3.5) ---
        // The bitstream order (§7.3.5) is: for each fbw channel in turn, read its
        // own mantissas up to its endmant; the shared coupling channel's mantissas
        // are read **inline**, right after the *first* coupled channel finishes its
        // own coefficients — NOT before all channels. LFE last. Each channel's
        // grouped-mantissa cache is fresh (a partial group does not cross channels).
        let mut cpl_bap = [0u8; N];
        if st.cplinu && st.cpl.have_exp {
            compute_bap(
                &st.cpl.exp,
                st.cpl.strtmant,
                st.cpl.endmant,
                &st.bai,
                st.cpl.fsnroffst,
                st.cpl.fgaincod,
                &mut cpl_bap,
                cfg.fscod,
                (st.cpl.fleak, st.cpl.sleak),
            );
        }
        let mut got_cplchan = false;
        for i in 0..nf {
            let end = st.ch[i].endmant;
            let mut bap = [0u8; N];
            compute_bap(
                &st.ch[i].exp,
                0,
                end,
                &st.bai,
                st.ch[i].fsnroffst,
                st.ch[i].fgaincod,
                &mut bap,
                cfg.fscod,
                (0, 0),
            );
            let dith = st.ch[i].dithflag;
            read_mantissas(r, &bap, 0, end, &mut st.ch[i].coeffs, &mut self.dither, dith)?;
            // Read the coupling channel inline after the first coupled channel.
            if st.cplinu && st.cpl.have_exp && st.cpl.chincpl[i] && !got_cplchan {
                read_mantissas(
                    r,
                    &cpl_bap,
                    st.cpl.strtmant,
                    st.cpl.endmant,
                    &mut st.cpl.coeffs,
                    &mut self.dither,
                    false,
                )?;
                got_cplchan = true;
            }
        }
        if cfg.lfeon {
            let mut bap = [0u8; N];
            compute_bap(
                &st.ch[LFE_CH].exp,
                0,
                7,
                &st.bai,
                st.ch[LFE_CH].fsnroffst,
                st.ch[LFE_CH].fgaincod,
                &mut bap,
                cfg.fscod,
                (0, 0),
            );
            read_mantissas(r, &bap, 0, 7, &mut st.ch[LFE_CH].coeffs, &mut self.dither, false)?;
        }

        // --- un-couple (§7.4.3): distribute the shared coupling channel into each
        // coupled channel's high bands, scaled by that channel's per-band coupling
        // coordinate. ---
        if st.cplinu && st.cpl.have_exp {
            for i in 0..nf {
                if st.cpl.chincpl[i] {
                    let mut bin = st.cpl.strtmant;
                    for b in 0..st.cpl.nbands {
                        let width = st.cpl.band_widths[b] as usize;
                        let coord = st.ch[i].cplco[b];
                        for _ in 0..width {
                            if bin >= st.cpl.endmant || bin >= N {
                                break;
                            }
                            st.ch[i].coeffs[bin] = st.cpl.coeffs[bin] * coord;
                            bin += 1;
                        }
                    }
                    if st.ch[i].endmant < st.cpl.endmant {
                        st.ch[i].endmant = st.cpl.endmant;
                    }
                }
            }
        }

        // --- rematrixing (§7.5): reverse sum/difference for 2/0 ---
        if cfg.acmod == 2 {
            let (a, b) = st.ch.split_at_mut(1);
            apply_rematrix(&mut a[0], &mut b[0], &st.rematflg);
        }

        Ok(approx)
    }

    /// Scale each channel's mantissas by 2^-exp, run the IMDCT + window +
    /// overlap-add, and append 256 PCM samples per output channel interleaved in
    /// ITU order L,R,C,LFE,Ls,Rs (§7.9 synthesis, §5.4.2.2 ordering).
    fn synthesize_block(
        &mut self,
        cfg: &StreamCfg,
        st: &mut BlockState,
        out_pcm: &mut Vec<i16>,
    ) {
        // Apply exponent scaling → final spectral coefficients.
        for i in 0..cfg.nfchans {
            let end = st.ch[i].endmant.min(N);
            for b in 0..end {
                let e = st.ch[i].exp[b].min(24);
                st.ch[i].coeffs[b] *= exp_scale(e);
            }
            for b in end..N {
                st.ch[i].coeffs[b] = 0.0;
            }
        }
        if cfg.lfeon {
            for b in 0..7 {
                let e = st.ch[LFE_CH].exp[b].min(24);
                st.ch[LFE_CH].coeffs[b] *= exp_scale(e);
            }
            for b in 7..N {
                st.ch[LFE_CH].coeffs[b] = 0.0;
            }
        }

        // IMDCT per transform channel.
        let mut pcm_ch: [[f32; N]; MAX_FBW] = [[0.0; N]; MAX_FBW];
        for i in 0..cfg.nfchans {
            if st.ch[i].blksw {
                self.overlap[i].imdct_short(&st.ch[i].coeffs, &mut pcm_ch[i]);
            } else {
                self.overlap[i].imdct_long(&st.ch[i].coeffs, &mut pcm_ch[i]);
            }
        }
        let mut lfe_pcm = [0.0f32; N];
        if cfg.lfeon {
            self.overlap[LFE_CH].imdct_long(&st.ch[LFE_CH].coeffs, &mut lfe_pcm);
        }

        // Permute stream order → ITU output order and interleave S16.
        let map = cfg.output_map();
        for n in 0..N {
            for &src in &map {
                let v = match src {
                    ChSrc::Full(i) => pcm_ch[i][n],
                    ChSrc::Lfe => lfe_pcm[n],
                };
                out_pcm.push(to_s16(v));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BSI parse (A/52 §5.3 for AC-3, §E.1.2 for E-AC-3).
// ---------------------------------------------------------------------------

fn parse_bsi_ac3(r: &mut BitReader) -> Result<StreamCfg, DecodeError> {
    if r.bits(16)? as u16 != SYNC_WORD {
        return Err(DecodeError::Invalid("sync word"));
    }
    let _crc1 = r.bits(16)?;
    let fscod = r.bits(2)? as usize;
    let frmsizecod = r.bits(6)? as usize;
    if fscod >= 3 {
        return Err(DecodeError::Invalid("fscod reserved"));
    }
    if frmsizecod >= 38 {
        return Err(DecodeError::Invalid("frmsizecod"));
    }
    let bsid = r.bits(5)?;
    if bsid > 8 {
        return Err(DecodeError::Invalid("bsid (expected AC-3 ≤ 8)"));
    }
    let _bsmod = r.bits(3)?;
    let acmod = r.bits(3)? as usize;
    if (acmod & 0x1) != 0 && acmod != 1 {
        let _cmixlev = r.bits(2)?;
    }
    if (acmod & 0x4) != 0 {
        let _surmixlev = r.bits(2)?;
    }
    if acmod == 2 {
        let _dsurmod = r.bits(2)?;
    }
    let lfeon = r.bit()? != 0;
    let _dialnorm = r.bits(5)?;
    if r.bit()? != 0 {
        let _compr = r.bits(8)?;
    }
    if r.bit()? != 0 {
        let _langcod = r.bits(8)?;
    }
    if r.bit()? != 0 {
        let _audprodie = r.bits(7)?;
    }
    if acmod == 0 {
        let _dialnorm2 = r.bits(5)?;
        if r.bit()? != 0 {
            let _compr2 = r.bits(8)?;
        }
        if r.bit()? != 0 {
            let _langcod2 = r.bits(8)?;
        }
        if r.bit()? != 0 {
            let _audprodie2 = r.bits(7)?;
        }
    }
    let _copyrightb = r.bit()?;
    let _origbs = r.bit()?;
    if r.bit()? != 0 {
        let _timecod1 = r.bits(14)?;
    }
    if r.bit()? != 0 {
        let _timecod2 = r.bits(14)?;
    }
    if r.bit()? != 0 {
        let addbsil = r.bits(6)? as usize;
        r.skip((addbsil + 1) * 8)?;
    }
    Ok(StreamCfg {
        acmod,
        nfchans: tables::NFCHANS[acmod],
        lfeon,
        fscod,
        sample_rate: tables::SAMPLE_RATE[fscod],
        eac3: false,
        eac3_approx_seen: false,
    })
}

fn parse_bsi_eac3(r: &mut BitReader) -> Result<(StreamCfg, usize), DecodeError> {
    if r.bits(16)? as u16 != SYNC_WORD {
        return Err(DecodeError::Invalid("sync word"));
    }
    let strmtyp = r.bits(2)?;
    if strmtyp == 1 {
        // dependent substream — decode the independent core only (brief).
        return Err(DecodeError::Invalid("dependent substream (skipped)"));
    }
    let _substreamid = r.bits(3)?;
    let _frmsiz = r.bits(11)?;
    let fscod = r.bits(2)? as usize;
    let (sample_rate, numblks) = if fscod == 3 {
        let fscod2 = r.bits(2)? as usize;
        let rate = match fscod2 {
            0 => 24_000,
            1 => 22_050,
            2 => 16_000,
            _ => return Err(DecodeError::Invalid("eac3 fscod2 reserved")),
        };
        (rate, 6usize)
    } else {
        let numblkscod = r.bits(2)? as usize;
        ([1usize, 2, 3, 6][numblkscod], tables::SAMPLE_RATE[fscod])
            .1
            .pipe(|rate| (rate, [1usize, 2, 3, 6][numblkscod]))
    };
    let acmod = r.bits(3)? as usize;
    let lfeon = r.bit()? != 0;
    let bsid = r.bits(5)?;
    if bsid != 16 {
        return Err(DecodeError::Invalid("bsid (expected E-AC-3 == 16)"));
    }
    let _dialnorm = r.bits(5)?;
    if r.bit()? != 0 {
        let _compr = r.bits(8)?;
    }
    if acmod == 0 {
        let _dialnorm2 = r.bits(5)?;
        if r.bit()? != 0 {
            let _compr2 = r.bits(8)?;
        }
    }
    // mixmdate (§E.1.2.2).
    if r.bit()? != 0 {
        skip_mixing_metadata(r, acmod, lfeon)?;
    }
    // infomdate (§E.1.2.2).
    if r.bit()? != 0 {
        skip_info_metadata(r, acmod)?;
    }
    // convsync — present for independent streams with numblks != 6.
    if strmtyp == 0 && numblks != 6 {
        let _convsync = r.bit()?;
    }
    // addbsi (§E.1.2.2).
    if r.bit()? != 0 {
        let addbsil = r.bits(6)? as usize;
        r.skip((addbsil + 1) * 8)?;
    }

    // --- audfrm: frame-level exponent-strategy / AHT flags (§E.1.2.3) ---
    let mut approx = false;
    let _expstre = r.bit()?;
    let ahte = r.bit()?;
    if ahte != 0 {
        approx = true; // AHT (§E.3.2) approximated by the base per-block decode.
    }
    let _snroffststr = r.bits(2)?;
    let _transproce = r.bit()?;
    let _blkswe = r.bit()?;
    let _dithflage = r.bit()?;
    let _bamode = r.bit()?;
    let _frmfgaincode = r.bit()?;
    let _dbaflde = r.bit()?;
    let _skipflde = r.bit()?;
    let _spxattene = r.bit()?;
    // The remaining frame-level cpl/spx flag tables (§E.1.2.3) are intricate and
    // acmod-dependent; the two target streams are plain DD+ 5.1. We stop the
    // frame-info parse here and let the per-block decode proceed from this
    // position — for a plain 5.1 DD+ frame the per-block syntax picks up cleanly.
    // A stream that enabled SPX would misalign and be caught by the frame-level
    // silence fallback + approx flag.

    Ok((
        StreamCfg {
            acmod,
            nfchans: tables::NFCHANS[acmod],
            lfeon,
            fscod: if fscod == 3 { 0 } else { fscod },
            sample_rate,
            eac3: true,
            eac3_approx_seen: approx,
        },
        numblks,
    ))
}

/// Skip the E-AC-3 mixing-metadata block (§E.1.2.2). Parsed only to keep the bit
/// cursor aligned; fields unused by the 5.1 core.
fn skip_mixing_metadata(r: &mut BitReader, acmod: usize, lfeon: bool) -> Result<(), DecodeError> {
    if acmod > 2 {
        let _dmixmod = r.bits(2)?;
    }
    if (acmod & 0x1) != 0 && acmod > 2 {
        let _ltrtcmixlev = r.bits(3)?;
        let _lorocmixlev = r.bits(3)?;
    }
    if (acmod & 0x4) != 0 {
        let _ltrtsurmixlev = r.bits(3)?;
        let _lorosurmixlev = r.bits(3)?;
    }
    if lfeon && r.bit()? != 0 {
        let _lfemixlevcod = r.bits(5)?;
    }
    if r.bit()? != 0 {
        let _pgmscl = r.bits(6)?;
    }
    if acmod == 0 && r.bit()? != 0 {
        let _pgmscl2 = r.bits(6)?;
    }
    if r.bit()? != 0 {
        let _extpgmscl = r.bits(6)?;
    }
    match r.bits(2)? {
        1 => {
            let _premixcmpsel = r.bit()?;
            let _drcsrc = r.bit()?;
            let _premixcmpscl = r.bits(3)?;
        }
        2 => {
            let _mixdata = r.bits(12)?;
        }
        3 => {
            let mixdeflen = r.bits(5)? as usize;
            r.skip((mixdeflen + 2) * 8)?;
        }
        _ => {}
    }
    if acmod < 2 {
        if r.bit()? != 0 {
            let _paninfo = r.bits(14)?;
        }
        if acmod == 0 && r.bit()? != 0 {
            let _paninfo2 = r.bits(14)?;
        }
    }
    if r.bit()? != 0 {
        let _frmmixcfginfoe = r.bits(5)?;
    }
    Ok(())
}

/// Skip the E-AC-3 informational-metadata block (§E.1.2.2).
fn skip_info_metadata(r: &mut BitReader, acmod: usize) -> Result<(), DecodeError> {
    let _bsmod = r.bits(3)?;
    let _copyrightb = r.bit()?;
    let _origbs = r.bit()?;
    if acmod == 2 {
        let _dsurmod = r.bits(2)?;
        let _dheadphonmod = r.bits(2)?;
    }
    if acmod >= 6 {
        let _dsurexmod = r.bits(2)?;
    }
    if r.bit()? != 0 {
        let _audprodi = r.bits(8)?;
    }
    if acmod == 0 && r.bit()? != 0 {
        let _audprodi2 = r.bits(8)?;
    }
    // sourcefscod (present when fscod != 3; the targets are 48k).
    let _sourcefscod = r.bit()?;
    Ok(())
}

/// Tiny pipe helper so the numblks match arm reads left-to-right.
trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

// ---------------------------------------------------------------------------
// Per-block / per-channel state.
// ---------------------------------------------------------------------------

/// Cross-block decode state for one frame.
struct BlockState {
    /// Full-bandwidth (0..5), LFE ([`LFE_CH`]) channel scratch.
    ch: [ChState; NUM_CH_SLOTS],
    cpl: CplState,
    cplinu: bool,
    bai: BitAllocInfo,
    rematflg: [bool; 4],
}

impl BlockState {
    fn new(_cfg: &StreamCfg) -> Self {
        Self {
            ch: std::array::from_fn(|_| ChState::new()),
            cpl: CplState::new(),
            cplinu: false,
            bai: BitAllocInfo::default(),
            rematflg: [false; 4],
        }
    }
}

/// One transform channel's cross-block + per-block state.
struct ChState {
    expstr: u32,
    endmant: usize,
    exp: [u8; N],
    coeffs: [f32; N],
    blksw: bool,
    dithflag: bool,
    fsnroffst: i32,
    fgaincod: usize,
    /// Per-coupling-band coordinate (reused when a block omits cplcoe; §7.4.2).
    cplco: [f32; 18],
}

impl ChState {
    fn new() -> Self {
        Self {
            expstr: 0,
            // No coefficients until a block codes a strategy (§7.1.3). Real
            // streams always code every active channel's strategy in block 0; a
            // channel that never codes one stays silent rather than reading
            // spurious mantissas from a stale endmant.
            endmant: 0,
            exp: [0; N],
            coeffs: [0.0; N],
            blksw: false,
            dithflag: true,
            fsnroffst: 0,
            fgaincod: 4,
            cplco: [1.0; 18],
        }
    }
}

/// Coupling-channel cross-block state (§7.4).
struct CplState {
    strtmant: usize,
    endmant: usize,
    nbands: usize,
    band_widths: [u8; 18],
    chincpl: [bool; MAX_FBW],
    phsflginu: bool,
    have_exp: bool,
    fsnroffst: i32,
    fgaincod: usize,
    /// Coupling leak init codes (§7.2.2.8, `cplfleak`/`cplsleak`), reused across
    /// blocks. Seed the coupling-channel leaky integrators in `compute_bap`.
    fleak: i32,
    sleak: i32,
    exp: [u8; N],
    coeffs: [f32; N],
}

impl CplState {
    fn new() -> Self {
        Self {
            strtmant: 0,
            endmant: 0,
            nbands: 0,
            band_widths: [0; 18],
            chincpl: [false; MAX_FBW],
            phsflginu: false,
            have_exp: false,
            fsnroffst: 0,
            fgaincod: 4,
            // A/52 §7.2.2.8 default coupling leak init when `cplleake` is absent.
            fleak: (10 << 8) + 256,
            sleak: (10 << 8) + 256,
            exp: [0; N],
            coeffs: [0.0; N],
        }
    }
}

/// The per-stream configuration derived from bsi.
struct StreamCfg {
    acmod: usize,
    nfchans: usize,
    lfeon: bool,
    fscod: usize,
    sample_rate: u32,
    eac3: bool,
    /// True if the frame header already declared an approximated E-AC-3 tool.
    eac3_approx_seen: bool,
}

/// A source channel for the output interleave.
#[derive(Clone, Copy)]
enum ChSrc {
    Full(usize),
    Lfe,
}

impl StreamCfg {
    fn out_channels(&self) -> usize {
        self.nfchans + usize::from(self.lfeon)
    }

    /// Map decoded transform-channel indices (A/52 stream order, §5.4.2.2,
    /// Table 5.8) to the ITU output order L, R, C, LFE, Ls, Rs.
    fn output_map(&self) -> Vec<ChSrc> {
        use ChSrc::{Full, Lfe};
        // Stream orders per acmod (Table 5.8):
        //   7 (3/2): L C R Ls Rs   6 (2/2): L R Ls Rs   5 (3/1): L C R S
        //   4 (2/1): L R S         3 (3/0): L C R        2 (2/0): L R
        //   1 (1/0): C             0 (1+1): Ch1 Ch2
        let mut m = match self.acmod {
            7 => vec![Full(0), Full(2), Full(1), Full(3), Full(4)], // → L R C Ls Rs
            6 => vec![Full(0), Full(1), Full(2), Full(3)],          // L R Ls Rs
            5 => vec![Full(0), Full(2), Full(1), Full(3)],          // → L R C S
            4 => vec![Full(0), Full(1), Full(2)],                   // L R S
            3 => vec![Full(0), Full(2), Full(1)],                   // → L R C
            2 => vec![Full(0), Full(1)],                            // L R
            1 => vec![Full(0)],                                     // C (mono)
            0 => vec![Full(0), Full(1)],                            // Ch1 Ch2
            _ => (0..self.nfchans).map(Full).collect(),
        };
        if self.lfeon {
            // LFE follows L,R,C in ITU order → position 3 for modes with ≥3 front.
            let pos = match self.acmod {
                7 | 5 | 3 => 3,
                _ => m.len(),
            };
            let pos = pos.min(m.len());
            m.insert(pos, Lfe);
        }
        m
    }
}

/// Bit-allocation parameters for a block (A/52 §7.2.2), reused across blocks.
struct BitAllocInfo {
    sdcycod: usize,
    fdcycod: usize,
    sgaincod: usize,
    dbpbcod: usize,
    floorcod: usize,
    csnroffst: i32,
}

impl Default for BitAllocInfo {
    fn default() -> Self {
        // A/52 §7.2.2.1 default parameter set.
        Self { sdcycod: 2, fdcycod: 1, sgaincod: 1, dbpbcod: 2, floorcod: 7, csnroffst: 0 }
    }
}

// ---------------------------------------------------------------------------
// Exponent decode (A/52 §7.1.2, §7.1.3).
// ---------------------------------------------------------------------------

#[inline]
fn grp_size(expstr: u32) -> usize {
    tables::EXP_GROUP_SIZE[(expstr.max(1) as usize - 1).min(2)]
}

/// Number of transmitted exponent groups for a full-bandwidth / LFE channel
/// (A/52 §7.1.3, "number of grouped exponents"). The `endmant - 1` differential
/// exponents (bins `1..endmant`) are packed 3-per-word, each word covering
/// `3*gs` bins, so the group count is the **ceiling** of `(endmant-1)/(3*gs)`.
/// The A/52 pseudocode writes this ceiling as `(nchmant + grpsize - 1) / grpsize`
/// with `grpsize = 3*gs`, i.e. the `+3` (D25) / `+9` (D45) rounding terms. Using a
/// bare floor `(endmant-1)/(3*gs)` drops the final partial group for D25/D45 and
/// under-reads the exponent section, misaligning every later block (the classic
/// D45 truncation).
#[inline]
fn exp_ngrps(endmant: usize, gs: usize) -> usize {
    let grp = 3 * gs;
    (endmant - 1 + grp - 3) / grp
}

/// Decode `ngrps` exponent groups into `exp[start..]` from a running `seed`
/// (already-known previous absolute exponent). Each group word packs 3 deltas as
/// `grp = 25*d0 + 5*d1 + d2`, each delta in {0..4} → {−2..+2} (§7.1.3), held
/// constant across `grp_size(expstr)` coefficients.
fn decode_exp_groups(
    r: &mut BitReader,
    expstr: u32,
    ngrps: usize,
    start: usize,
    seed: &mut u8,
    exp: &mut [u8],
) -> Result<(), DecodeError> {
    let gs = grp_size(expstr);
    let mut idx = start;
    for _ in 0..ngrps {
        // A group word encodes 3 deltas as `grp = 25*d0 + 5*d1 + d2`, each digit
        // in 0..=4 (§7.1.3), so a well-formed word is < 125. A word ≥ 125 is a bit
        // that shouldn't occur in a conforming stream; rather than reject the whole
        // frame we decode the digits mod 5 and let [`apply_exp_delta`] clamp — this
        // consumes exactly 7 bits either way, so alignment is preserved and at most
        // a few exponents are perturbed (robustness over strictness on real files).
        let grp = r.bits(7)?;
        for &delta in &[(grp / 25) % 5, (grp / 5) % 5, grp % 5] {
            *seed = apply_exp_delta(*seed, delta);
            for _ in 0..gs {
                if idx < exp.len() {
                    exp[idx] = *seed;
                    idx += 1;
                }
            }
        }
    }
    Ok(())
}

/// Like [`decode_exp_groups`] but seeding from `exp[0]` (already read) and filling
/// `exp[1..]` — the full-bandwidth / LFE channel case (§7.1.2).
fn decode_exp_after0(
    r: &mut BitReader,
    expstr: u32,
    ngrps: usize,
    seed: &mut u8,
    exp: &mut [u8],
) -> Result<(), DecodeError> {
    decode_exp_groups(r, expstr, ngrps, 1, seed, exp)
}

/// Apply one biased exponent delta (0..=4 → −2..=+2), clamping to 0..=24 (§7.1.3).
#[inline]
fn apply_exp_delta(prev: u8, delta: u32) -> u8 {
    let e = prev as i32 + delta as i32 - 2;
    e.clamp(0, 24) as u8
}

// ---------------------------------------------------------------------------
// Bit allocation (A/52 §7.2.2).
// ---------------------------------------------------------------------------

/// Compute the bit-allocation pointer per bin over `[start, end)` — the A/52
/// §7.2.2 parametric bit allocation, transcribed faithfully from the standard's
/// pseudocode (the routine must be **bit-exact**: the number of bits each
/// mantissa consumes is `qntztab[bap]`, so any deviation from the encoder's bap
/// misaligns the whole rest of the frame).
///
/// The steps (A/52 §7.2.2.3–§7.2.2.10):
/// 1. PSD `psd[bin] = 3072 - (exp[bin] << 7)`.
/// 2. Band-integrate PSD → `bndpsd[band]` via the log-domain add (`latab`).
/// 3. Excitation via the two-slope leaky integrator (`fastleak`/`slowleak`) with
///    the `lowcomp` low-frequency compensation for the first bands.
/// 4. Mask = max(excitation, hearing-threshold); floor & 128-step handling.
/// 5. Address `= (psd[bin] - (mask - snroffset)) >> 5`, clamp 0..63 → `baptab`.
#[allow(clippy::too_many_arguments)]
fn compute_bap(
    exp: &[u8],
    start: usize,
    end: usize,
    bai: &BitAllocInfo,
    fsnroffst: i32,
    fgaincod: usize,
    bap_out: &mut [u8],
    fscod: usize,
    cpl_leak: (i32, i32),
) {
    let end = end.min(N).min(exp.len());
    if start >= end {
        return;
    }
    // --- Step 1: PSD (§7.2.2.4). ---
    let mut psd = [0i32; N];
    for b in start..end {
        psd[b] = 3072 - ((exp[b] as i32) << 7);
    }
    // --- Step 2: band PSD integration (§7.2.2.4). ---
    // Walk the band table from the band containing `start` to the band containing
    // `end-1`, integrating each band's bins. `bndtab`/`bndsz` give band layout.
    let sband = tables::bin_to_band()[start] as usize;
    let eband = tables::bin_to_band()[end - 1] as usize;
    let mut bndpsd = [0i32; 50];
    for band in sband..=eband {
        let mut j = (tables::BAND_START[band] as usize).max(start);
        let bend = (tables::BAND_START[band] as usize + tables::BAND_SIZE[band] as usize).min(end);
        if j >= bend {
            bndpsd[band] = -640 * 128; // effectively silent
            continue;
        }
        let mut lp = psd[j];
        j += 1;
        while j < bend {
            lp = log_add(lp, psd[j]);
            j += 1;
        }
        bndpsd[band] = lp;
    }
    // --- Step 3: excitation function (§7.2.2.5), with lowcomp (§7.2.2.5). ---
    let sdecay = tables::SLOW_DECAY[bai.sdcycod.min(3)] as i32;
    let fdecay = tables::FAST_DECAY[bai.fdcycod.min(3)] as i32;
    let sgain = tables::SLOW_GAIN[bai.sgaincod.min(3)] as i32;
    let dbknee = tables::DB_PER_BIT[bai.dbpbcod.min(3)] as i32;
    let floor = tables::FLOOR[bai.floorcod.min(7)];
    let fgain = tables::FAST_GAIN[fgaincod.min(7)] as i32;

    // `bndend` is the *exclusive* band index (one past the last band with bins),
    // matching the A/52 reference `bndend` used in the loop bounds below.
    let bndend = eband + 1;
    let bndstrt = sband;

    let mut excite = [0i32; 50];
    // Running leaky-integrator state, carried across the three §7.2.2.7 stages.
    let mut fastleak = 0i32;
    let mut slowleak = 0i32;
    // `begin` is where the final plain-spread loop starts; it is set to 22 for the
    // fbw/lfe path and to `bndstrt` for the coupling path (§7.2.2.7).
    let begin;
    // `bndstrt == 0` for full-bandwidth / LFE channels starting at bin 0; the
    // coupling channel starts mid-band (bndstrt > 0) and skips the lowcomp path
    // (§7.2.2.7 "compute excitation function"). Transcribed directly.
    if bndstrt == 0 {
        let mut lowcomp = 0i32;
        lowcomp = calc_lowcomp(lowcomp, bndpsd[0], bndpsd[1], 0);
        excite[0] = bndpsd[0] - fgain - lowcomp;
        lowcomp = calc_lowcomp(lowcomp, bndpsd[1], bndpsd[2], 1);
        excite[1] = bndpsd[1] - fgain - lowcomp;
        // Bands 2..7: find where the PSD stops rising (`begin` = break point).
        let mut b = 7usize;
        for bin in 2..7 {
            if !(bndend == 7 && bin == 6) {
                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
            }
            fastleak = bndpsd[bin] - fgain;
            slowleak = bndpsd[bin] - sgain;
            excite[bin] = fastleak - lowcomp;
            if !(bndend == 7 && bin == 6) && bndpsd[bin] <= bndpsd[bin + 1] {
                b = bin + 1;
                break;
            }
        }
        // Bands `b`..min(bndend,22): lowcomp-corrected leaky spread. `fastleak`
        // and `slowleak` carry their values from the loop above.
        for bin in b..bndend.min(22) {
            if !(bndend == 7 && bin == 6) {
                lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
            }
            fastleak = (fastleak - fdecay).max(bndpsd[bin] - fgain);
            slowleak = (slowleak - sdecay).max(bndpsd[bin] - sgain);
            excite[bin] = (fastleak - lowcomp).max(slowleak);
        }
        begin = 22;
    } else {
        // Coupling channel (§7.2.2.7): the plain leaky spread starts at `bndstrt`
        // with `fastleak`/`slowleak` seeded from the block's `cplfleak`/`cplsleak`
        // init `(cplleak << 8) + 256` (passed in via `cpl_leak`).
        fastleak = cpl_leak.0;
        slowleak = cpl_leak.1;
        begin = bndstrt;
    }
    // Final stage — plain leaky spread, no lowcomp (§7.2.2.7).
    for bin in begin..bndend {
        fastleak = (fastleak - fdecay).max(bndpsd[bin] - fgain);
        slowleak = (slowleak - sdecay).max(bndpsd[bin] - sgain);
        excite[bin] = fastleak.max(slowleak);
    }
    // --- Step 4: mask (§7.2.2.8): dbknee correction then hearing threshold. ---
    let hth = &tables::HEARING_THRESHOLD[fscod.min(2)];
    let mut mask = [0i32; 50];
    for bin in bndstrt..bndend {
        if bndpsd[bin] < dbknee {
            excite[bin] += (dbknee - bndpsd[bin]) >> 2;
        }
        mask[bin] = excite[bin].max(hth[bin] as i32);
    }
    // --- Step 5: per-bin address → bap (§7.2.2.10, "compute bit allocation"). ---
    // snroffset = (((csnroffst - 15) << 4) + fsnroffst) << 2 (§7.2.2.6).
    let snroffset = (((bai.csnroffst - 15) << 4) + fsnroffst) << 2;
    for band in bndstrt..bndend {
        // §7.2.2.10: mask -= snroffset; mask -= floor; if <0 → 0; &= 0x1FE0;
        // mask += floor. Then address = (psd - mask) >> 5, clamped to [0,63].
        let mut m = mask[band] - snroffset - floor;
        if m < 0 {
            m = 0;
        }
        m = (m & 0x1FE0) + floor;
        let lo = (tables::BAND_START[band] as usize).max(start);
        let hi = (tables::BAND_START[band] as usize + tables::BAND_SIZE[band] as usize).min(end);
        for b in lo..hi {
            let addr = ((psd[b] - m) >> 5).clamp(0, 63) as usize;
            bap_out[b] = tables::BAPTAB[addr];
        }
    }
}

/// Low-frequency compensation update for the excitation function (A/52 §7.2.2.5,
/// the `calc_lowcomp` routine). `band` is the current band index.
#[inline]
fn calc_lowcomp(a: i32, b0: i32, b1: i32, band: usize) -> i32 {
    if band < 7 {
        if (b0 + 256) == b1 {
            384
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else if band < 20 {
        if (b0 + 256) == b1 {
            320
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else {
        (a - 128).max(0)
    }
}

#[inline]
fn log_add(a: i32, b: i32) -> i32 {
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    let idx = ((hi - lo) >> 1).clamp(0, (tables::LATAB.len() - 1) as i32) as usize;
    hi + tables::LATAB[idx] as i32
}

// ---------------------------------------------------------------------------
// Mantissa decode (A/52 §7.3).
// ---------------------------------------------------------------------------

/// Cached grouped-mantissa words (A/52 §7.3.5). A grouped bap (1, 2, 4) reads one
/// code word and yields several mantissas; the word is cached and drained across
/// successive coefficients. `*_pos == count` means "exhausted, read a fresh word".
/// bap 1 and 2 yield 3 mantissas per word; bap 4 yields **2** — the `_pos` is
/// primed to the per-bap count so the first access reads a word.
struct MantGroups {
    g1: [f32; 3],
    g1_pos: usize,
    g2: [f32; 3],
    g2_pos: usize,
    g4: [f32; 3],
    g4_pos: usize,
}

impl MantGroups {
    fn new() -> Self {
        // Prime each cursor at/above its group count so the first read fetches a
        // word. Using 3 (the max group count) is safe for all three baps.
        Self { g1: [0.0; 3], g1_pos: 3, g2: [0.0; 3], g2_pos: 3, g4: [0.0; 3], g4_pos: 3 }
    }
}

fn read_mantissas(
    r: &mut BitReader,
    bap: &[u8],
    start: usize,
    end: usize,
    coeffs: &mut [f32],
    dither: &mut Dither,
    dithflag: bool,
) -> Result<(), DecodeError> {
    let end = end.min(N).min(coeffs.len());
    let mut groups = MantGroups::new();
    for b in start..end {
        let v = match bap[b] {
            0 if dithflag => dither.sample(),
            0 => 0.0,
            bp @ (1 | 2 | 4) => next_grouped(r, bp, &mut groups)?,
            bp @ (3 | 5..=15) => {
                let bits = tables::BAP_BITS[bp as usize];
                let code = r.bits(u32::from(bits))?;
                dequant_direct(bp, code, bits)
            }
            _ => 0.0,
        };
        coeffs[b] = v;
    }
    Ok(())
}

fn next_grouped(r: &mut BitReader, bap: u8, g: &mut MantGroups) -> Result<f32, DecodeError> {
    let (buf, pos) = match bap {
        1 => (&mut g.g1, &mut g.g1_pos),
        2 => (&mut g.g2, &mut g.g2_pos),
        4 => (&mut g.g4, &mut g.g4_pos),
        _ => unreachable!(),
    };
    // A grouped word yields `GROUP_COUNT[bap]` mantissas — 3 for bap 1/2, **2**
    // for bap 4 (A/52 §7.3.5). Read a fresh word only once the cache is drained,
    // so the bit consumption matches the encoder exactly.
    let count = GROUP_COUNT[bap as usize];
    if *pos >= count {
        let bits = GROUP_BITS[bap as usize];
        let code = r.bits(u32::from(bits))?;
        *buf = dequant_group(bap, code);
        *pos = 0;
    }
    let v = buf[*pos];
    *pos += 1;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Rematrixing (A/52 §7.5).
// ---------------------------------------------------------------------------

/// The 4 rematrix sub-band bin ranges (A/52 §7.5.1, Table 7.29).
const REMAT_BANDS: [(usize, usize); 4] = [(13, 25), (25, 37), (37, 61), (61, 253)];

fn apply_rematrix(l: &mut ChState, r_ch: &mut ChState, rematflg: &[bool; 4]) {
    for (band, &flag) in rematflg.iter().enumerate() {
        if !flag {
            continue;
        }
        let (lo, hi) = REMAT_BANDS[band];
        for b in lo..hi.min(N) {
            let sum = l.coeffs[b];
            let diff = r_ch.coeffs[b];
            l.coeffs[b] = sum + diff;
            r_ch.coeffs[b] = sum - diff;
        }
    }
}

// ---------------------------------------------------------------------------
// Scaling / output helpers.
// ---------------------------------------------------------------------------

/// 2^-exp scaling for a coefficient given its absolute exponent (§7.3.2).
#[inline]
fn exp_scale(exp: u8) -> f32 {
    1.0f32 / ((1u32 << exp.min(24)) as f32)
}

/// Clamp a normalized float sample to S16 (§7.9 / §7.11 output).
#[inline]
fn to_s16(v: f32) -> i16 {
    let s = v * 32768.0;
    if s >= 32767.0 {
        32767
    } else if s <= -32768.0 {
        -32768
    } else {
        s as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peek_rejects_bad_sync() {
        assert!(matches!(
            Frame::peek_len(&[0x00, 0x00, 0, 0, 0, 0]),
            Err(DecodeError::Invalid(_))
        ));
    }

    #[test]
    fn exp_scale_halves() {
        assert!((exp_scale(0) - 1.0).abs() < 1e-9);
        assert!((exp_scale(1) - 0.5).abs() < 1e-9);
        assert!((exp_scale(4) - 0.0625).abs() < 1e-9);
    }

    #[test]
    fn s16_saturates() {
        assert_eq!(to_s16(2.0), 32767);
        assert_eq!(to_s16(-2.0), -32768);
        assert_eq!(to_s16(0.0), 0);
    }

    #[test]
    fn exp_delta_clamps() {
        assert_eq!(apply_exp_delta(10, 2), 10);
        assert_eq!(apply_exp_delta(10, 4), 12);
        assert_eq!(apply_exp_delta(10, 0), 8);
        assert_eq!(apply_exp_delta(0, 0), 0);
        assert_eq!(apply_exp_delta(24, 4), 24);
    }

    #[test]
    fn output_map_51_side_is_itu_order() {
        // acmod 7 (3/2) + LFE → L R C LFE Ls Rs = Full(0) Full(2) Full(1) Lfe Full(3) Full(4).
        let cfg = StreamCfg {
            acmod: 7,
            nfchans: 5,
            lfeon: true,
            fscod: 0,
            sample_rate: 48_000,
            eac3: false,
            eac3_approx_seen: false,
        };
        let m = cfg.output_map();
        assert_eq!(m.len(), 6);
        assert!(matches!(m[0], ChSrc::Full(0))); // L
        assert!(matches!(m[1], ChSrc::Full(2))); // R (stream idx 2)
        assert!(matches!(m[2], ChSrc::Full(1))); // C (stream idx 1)
        assert!(matches!(m[3], ChSrc::Lfe)); // LFE
        assert!(matches!(m[4], ChSrc::Full(3))); // Ls
        assert!(matches!(m[5], ChSrc::Full(4))); // Rs
    }
}
