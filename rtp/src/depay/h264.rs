//! H.264 depayloading (RFC 6184): RTP payloads → Annex-B access units.
//!
//! Scope: packetization modes 0 and 1 — single NAL unit mode (§6.2) and
//! non-interleaved mode (§6.3) — i.e. single NAL unit packets (§5.6), STAP-A
//! (§5.7.1), and FU-A (§5.8). The interleaved mode's packet types (STAP-B 25,
//! MTAP16 26, MTAP24 27, FU-B 29) transmit NAL units out of decoding order,
//! keyed by decoding order numbers (§5.5); reassembling them without the DON
//! machinery would emit misordered NAL units, so they fail loudly as
//! [`H264DepayError::Unsupported`] instead (never silent corruption).
//!
//! Access-unit boundary: one access unit = one RTP timestamp (§5.1 — the
//! timestamp is "the sampling timestamp of the content", 90 kHz). The marker
//! bit is set on the AU's last packet, but §5.1 says receivers "MAY use this
//! bit as an early indication ... but MUST NOT rely on this property", so a
//! marker completes the AU immediately and a timestamp change is the fallback
//! that flushes the previous AU when the marker was absent (or its packet
//! lost).
//!
//! Input contract: payloads arrive in RTP sequence-number order (§7.1 — the
//! jitter buffer upstream reorders); a detected gap is announced via
//! [`H264Depay::discontinuity`], never fed through silently.

/// Why a payload was rejected. **Any error drops the partial access unit and
/// leaves the depacketizer resynchronizing, exactly as after
/// [`H264Depay::discontinuity`]** — the erroring packet contributed no data,
/// and output resumes at the next provable access-unit boundary. The caller
/// may keep pushing (streams recover at the next AU) or tear down on
/// [`H264DepayError::Unsupported`], which indicates an interleaved-mode
/// sender the whole session cannot handle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum H264DepayError {
    /// An interleaved-mode packet type: STAP-B (25), MTAP16 (26), MTAP24 (27)
    /// — §5.7 — or FU-B (29) — §5.8. Allowed only in packetization mode 2
    /// (§5.4 Table 3), which is out of scope here.
    Unsupported {
        /// The type field of the payload's first octet (§5.2).
        packet_type: u8,
    },
    /// The payload is shorter than its declared structure: empty, an FU-A
    /// without its two header octets (§5.8), or a STAP-A aggregation-unit
    /// size running past the payload end (§5.7.1).
    Truncated,
    /// A STAP-A aggregation unit is invalid: zero-length NAL unit, or an
    /// aggregated packet type 24–29 (§5.7: "Aggregation packets MUST NOT be
    /// nested" and "MUST NOT contain fragmentation units").
    BadAggregation,
    /// The FU-A state machine was violated: S and E both set (§5.8: "MUST
    /// NOT"); a fragment type that is itself an aggregate/fragment (§5.8:
    /// STAPs/MTAPs MUST NOT be fragmented, FUs MUST NOT be nested); a start
    /// while another NAL unit's fragments were open, a continuation with none
    /// open, a timestamp change mid-NAL, or a non-FU packet between fragments
    /// (§5.8: fragments are consecutive and share the NALU-time); or a marker
    /// bit before the final fragment (§5.8: the last fragment carries E).
    BadFragmentation,
}

/// The Annex-B start code prepended to every output NAL unit.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The stateful depacketizer: payloads in (in sequence order — the jitter
/// buffer upstream guarantees it, gaps declared via
/// [`H264Depay::discontinuity`]), Annex-B access units out.
#[derive(Debug, Default)]
pub struct H264Depay {
    /// Out-of-band parameter sets (§8.1 `sprop-parameter-sets`, already
    /// base64-decoded by the SDP layer), Annex-B framed; §8.1 defines them as
    /// NAL units "that can be placed in the NAL unit stream to precede any
    /// other NAL units in decoding order" — prepended to the first output.
    sprop: Vec<u8>,
    sprop_pending: bool,
    /// The access unit under assembly, Annex-B framed.
    au: Vec<u8>,
    /// RTP timestamp of the AU under assembly (meaningful while `au` is
    /// non-empty or `fu` is open).
    ts: u32,
    /// The fragmented NAL unit under FU-A reassembly (§5.8): the
    /// reconstructed NAL unit type octet followed by the fragments so far.
    fu: Option<Vec<u8>>,
    /// Resynchronizing after loss/error: skip payloads until a fresh access
    /// unit provably starts.
    resync: bool,
    /// Timestamp of the partial access unit being skipped during resync.
    resync_ts: Option<u32>,
}

impl H264Depay {
    /// `sprop`: the decoded `sprop-parameter-sets` NAL units (§8.1; already
    /// base64-decoded by the SDP layer), prepended before the first output.
    ///
    /// The stream is assumed to begin at an access-unit boundary (a fresh
    /// sender starts clean); a caller joining an in-progress stream should
    /// call [`H264Depay::discontinuity`] first to resynchronize.
    // COLD: builds the fixed sprop parameter-set state once per stream at setup.
    #[allow(clippy::disallowed_methods)]
    pub fn new(sprop: Vec<Vec<u8>>) -> H264Depay {
        let mut annexb = Vec::new();
        for nal in &sprop {
            if nal.is_empty() {
                continue;
            }
            annexb.extend_from_slice(&START_CODE);
            annexb.extend_from_slice(nal);
        }
        H264Depay {
            sprop_pending: !annexb.is_empty(),
            sprop: annexb,
            ..H264Depay::default()
        }
    }

    /// Feed one payload (marker + timestamp from the RTP header). Returns the
    /// access units completed by this packet — usually zero or one, but two
    /// when a timestamp change flushes the previous AU *and* this packet's
    /// marker completes the next in the same call.
    ///
    /// API deviation from the original stub (`-> Option<Vec<u8>>`): the
    /// error enum is required for the loud-failure contract, and the §5.1
    /// timestamp fallback makes the two-AU case real.
    pub fn push(
        &mut self,
        payload: &[u8],
        marker: bool,
        timestamp: u32,
    ) -> Result<Vec<Vec<u8>>, H264DepayError> {
        if payload.is_empty() {
            return Err(self.fail(timestamp, H264DepayError::Truncated));
        }
        let packet_type = payload[0] & 0x1F;

        // §5.4 Table 3: types 0 and 30-31 are reserved and "MUST be ignored
        // by a receiver" — treated as if never received (no NAL units, no
        // boundary evidence, marker not trusted).
        if matches!(packet_type, 0 | 30 | 31) {
            return Ok(Vec::new());
        }

        // Interleaved-mode-only types (§5.4 Table 3) — out of scope, loud.
        if matches!(packet_type, 25..=27 | 29) {
            return Err(self.fail(timestamp, H264DepayError::Unsupported { packet_type }));
        }

        // Resynchronization (after discontinuity()/error): the first packets
        // may continue an access unit whose head was lost, and RFC 6184 gives
        // no in-payload AU-start flag. Skip until a boundary proves a fresh
        // AU: the packet after a marker, or a timestamp change — packets of
        // one AU share the timestamp (§5.1) and, in sequence order with no
        // undeclared gap, a new timestamp is a new AU's first packet.
        if self.resync {
            let t0 = *self.resync_ts.get_or_insert(timestamp);
            if timestamp == t0 {
                if marker {
                    // The skipped AU ends here; the next packet starts fresh.
                    self.resync = false;
                    self.resync_ts = None;
                }
                return Ok(Vec::new());
            }
            // Timestamp changed: this packet starts a new access unit.
            self.resync = false;
            self.resync_ts = None;
        }

        let mut done = Vec::new();

        // §5.1 fallback boundary: a timestamp change with an AU pending means
        // the previous AU is complete (its marker packet was absent or lost).
        if (!self.au.is_empty() || self.fu.is_some()) && timestamp != self.ts {
            if self.fu.is_some() {
                // §5.8: fragments of one NAL unit are consecutive and all
                // carry its NALU-time; a mid-NAL timestamp change means the
                // fragment tail was lost without a declared discontinuity.
                return Err(self.fail(timestamp, H264DepayError::BadFragmentation));
            }
            done.push(self.take_au());
        }
        self.ts = timestamp;

        match packet_type {
            // Single NAL unit packet (§5.6): the payload is the NAL unit,
            // its first octet co-serving as the payload header (§5.2).
            1..=23 => {
                if self.fu.is_some() {
                    // §5.8: no other packets of the stream may be sent
                    // between the fragments of a NAL unit.
                    return Err(self.fail(timestamp, H264DepayError::BadFragmentation));
                }
                self.au.extend_from_slice(&START_CODE);
                self.au.extend_from_slice(payload);
            }
            // STAP-A (§5.7.1): type octet, then aggregation units.
            24 => {
                if self.fu.is_some() {
                    return Err(self.fail(timestamp, H264DepayError::BadFragmentation));
                }
                self.push_stap_a(&payload[1..])?;
            }
            // FU-A (§5.8): indicator octet, header octet, fragment.
            28 => self.push_fu_a(payload)?,
            _ => unreachable!("types 0, 24..=31 handled above"),
        }

        // §5.1: the marker is set on "the very last packet of the access
        // unit" — trust it when present for low-latency completion.
        if marker {
            if self.fu.is_some() {
                // §5.8: the last fragment of the AU's last NAL unit carries
                // E; a marker with the fragment still open means loss.
                return Err(self.fail(timestamp, H264DepayError::BadFragmentation));
            }
            if !self.au.is_empty() {
                done.push(self.take_au());
            }
        }
        Ok(done)
    }

    /// A sequence discontinuity was declared upstream (loss): drop the
    /// partial access unit and any open fragment, then resynchronize — skip
    /// input until a packet provably starts a new access unit (the packet
    /// after a marker, or the first packet of a new timestamp).
    pub fn discontinuity(&mut self) {
        self.au.clear();
        self.fu = None;
        self.resync = true;
        self.resync_ts = None;
    }

    /// End of stream: return the access unit still under assembly, if any
    /// (its marker/next-timestamp packet will never arrive). An open
    /// fragmented NAL unit is incomplete and is discarded (§5.8: receivers
    /// discard the partial fragment run). API addition over the stub.
    pub fn flush(&mut self) -> Option<Vec<u8>> {
        self.fu = None;
        if self.au.is_empty() {
            None
        } else {
            Some(self.take_au())
        }
    }

    /// STAP-A payload after the type octet (§5.7.1): a sequence of
    /// aggregation units — 16-bit big-endian NAL unit size (excluding the two
    /// size octets, including the NAL header octet), then the NAL unit.
    fn push_stap_a(&mut self, mut units: &[u8]) -> Result<(), H264DepayError> {
        if units.is_empty() {
            // §5.7.1: "at least one single-time aggregation unit".
            return Err(self.fail(self.ts, H264DepayError::BadAggregation));
        }
        while !units.is_empty() {
            if units.len() < 2 {
                return Err(self.fail(self.ts, H264DepayError::Truncated));
            }
            let size = u16::from_be_bytes([units[0], units[1]]) as usize;
            if size == 0 {
                // A NAL unit is at least its type octet.
                return Err(self.fail(self.ts, H264DepayError::BadAggregation));
            }
            if units.len() < 2 + size {
                return Err(self.fail(self.ts, H264DepayError::Truncated));
            }
            let nal = &units[2..2 + size];
            match nal[0] & 0x1F {
                // §5.7: aggregation packets MUST NOT be nested and MUST NOT
                // contain fragmentation units.
                24..=29 => return Err(self.fail(self.ts, H264DepayError::BadAggregation)),
                // §5.4: reserved types "MUST be ignored by a receiver".
                0 | 30 | 31 => {}
                _ => {
                    self.au.extend_from_slice(&START_CODE);
                    self.au.extend_from_slice(nal);
                }
            }
            units = &units[2 + size..];
        }
        Ok(())
    }

    /// One FU-A payload (§5.8): FU indicator `F|NRI|28`, FU header
    /// `S|E|R|type`, then the fragment bytes.
    fn push_fu_a(&mut self, payload: &[u8]) -> Result<(), H264DepayError> {
        if payload.len() < 2 {
            return Err(self.fail(self.ts, H264DepayError::Truncated));
        }
        let (indicator, header) = (payload[0], payload[1]);
        let start = header & 0x80 != 0;
        let end = header & 0x40 != 0;
        // The R bit (0x20) "MUST be ignored by the receiver" (§5.8).
        let nal_type = header & 0x1F;
        if start && end {
            // §5.8: "the Start bit and End bit MUST NOT both be set to one
            // in the same FU header".
            return Err(self.fail(self.ts, H264DepayError::BadFragmentation));
        }
        if !(1..=23).contains(&nal_type) {
            // §5.8: "STAPs and MTAPs MUST NOT be fragmented. FUs MUST NOT be
            // nested" — the FU header type is a plain NAL unit type (§5.8:
            // "as defined in Table 7-1 of [1]").
            return Err(self.fail(self.ts, H264DepayError::BadFragmentation));
        }
        if start {
            if self.fu.is_some() {
                // A new fragmented NAL unit while the previous one is open:
                // its tail was lost without an upstream discontinuity() call.
                return Err(self.fail(self.ts, H264DepayError::BadFragmentation));
            }
            // §5.8: the fragmented NAL unit's type octet is not carried in
            // the FU payload; it is reconstructed from the indicator's F+NRI
            // bits and the FU header's type field.
            let mut nal = Vec::with_capacity(payload.len() - 1);
            nal.push((indicator & 0xE0) | nal_type);
            nal.extend_from_slice(&payload[2..]);
            self.fu = Some(nal);
        } else {
            match self.fu.as_mut() {
                // §7.1: fragments are concatenated in sending order.
                Some(fu) => fu.extend_from_slice(&payload[2..]),
                // Continuation with nothing open: the start fragment was
                // lost and the loss was not declared upstream.
                None => return Err(self.fail(self.ts, H264DepayError::BadFragmentation)),
            }
        }
        if end {
            // §5.8 E bit: "the last byte of the payload is also the last
            // byte of the fragmented NAL unit" — the NAL unit is complete.
            let mut nal = self.fu.take().expect("fragment fed above");
            self.au.extend_from_slice(&START_CODE);
            self.au.append(&mut nal);
        }
        Ok(())
    }

    /// The completed AU, with the §8.1 parameter sets prepended exactly once
    /// before the first output.
    fn take_au(&mut self) -> Vec<u8> {
        let au = std::mem::take(&mut self.au);
        if self.sprop_pending {
            self.sprop_pending = false;
            let mut out = std::mem::take(&mut self.sprop);
            out.extend_from_slice(&au);
            return out;
        }
        au
    }

    /// Errors are not recoverable mid-AU: drop partial state and enter the
    /// same resynchronizing wait as [`H264Depay::discontinuity`] — but unlike
    /// declared loss, the erroring packet's timestamp is known (`at_ts`), so
    /// the resync can seed on it: input stayed sequence-contiguous, so the
    /// next packet with a *different* timestamp already provably starts a
    /// fresh access unit (§5.1) and no extra AU is sacrificed.
    fn fail(&mut self, at_ts: u32, e: H264DepayError) -> H264DepayError {
        self.discontinuity();
        self.resync_ts = Some(at_ts);
        e
    }
}
