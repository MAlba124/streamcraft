//! POD audio format vocabulary + the `audio/raw` negotiation offer (spec: Formats;
//! Crate layout). Plain structs and free functions — no new buffer types, no traits.
//!
//! The `audio/raw` family names three fields — `rate` (Int Hz), `channels` (Int), and
//! `sample` (a categorical id like `s16`) — matching the string-keyed offer descriptors
//! the pipeline interns at link time. [`SampleFormat::caps_name`] is the single source of
//! truth for those categorical names, so offers, WAV parsing, and views never drift.

use streamcraft_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};

/// The negotiation family for uncompressed interleaved PCM.
pub const FAMILY: &str = "audio/raw";
/// Sample rate in Hz (`Value::Int`).
pub const FIELD_RATE: &str = "rate";
/// Channel count (`Value::Int`).
pub const FIELD_CHANNELS: &str = "channels";
/// Sample format, a categorical id — see [`SampleFormat::caps_name`].
pub const FIELD_SAMPLE: &str = "sample";

/// Interleaved PCM sample format. Integer formats are little-endian on the wire (WAV and
/// FLAC native); `F32` is native-endian IEEE-754.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleFormat {
    /// Unsigned 8-bit (WAV stores 8-bit PCM biased by 128).
    U8,
    S16,
    S24,
    S32,
    F32,
}

impl SampleFormat {
    /// Bytes per single-channel sample.
    pub const fn bytes(self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::S16 => 2,
            SampleFormat::S24 => 3,
            SampleFormat::S32 => 4,
            SampleFormat::F32 => 4,
        }
    }

    /// Bit depth carried in the format (24 for `S24`, even though it occupies 3 bytes).
    pub const fn bits(self) -> u32 {
        match self {
            SampleFormat::U8 => 8,
            SampleFormat::S16 => 16,
            SampleFormat::S24 => 24,
            SampleFormat::S32 => 32,
            SampleFormat::F32 => 32,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, SampleFormat::F32)
    }

    /// The categorical name used in `audio/raw` offers (`ValueDesc::Id(name)`).
    pub const fn caps_name(self) -> &'static str {
        match self {
            SampleFormat::U8 => "u8",
            SampleFormat::S16 => "s16",
            SampleFormat::S24 => "s24",
            SampleFormat::S32 => "s32",
            SampleFormat::F32 => "f32",
        }
    }

    pub fn from_caps_name(s: &str) -> Option<SampleFormat> {
        Some(match s {
            "u8" => SampleFormat::U8,
            "s16" => SampleFormat::S16,
            "s24" => SampleFormat::S24,
            "s32" => SampleFormat::S32,
            "f32" => SampleFormat::F32,
            _ => return None,
        })
    }

    /// Resolve a WAV `(format tag, bits-per-sample)` pair. `tag` is the *resolved* code
    /// (1 = PCM integer, 3 = IEEE float); `WAVE_FORMAT_EXTENSIBLE` is unwrapped by the
    /// parser before calling this.
    pub fn from_wav(tag: u16, bits: u16) -> Option<SampleFormat> {
        Some(match (tag, bits) {
            (1, 8) => SampleFormat::U8,
            (1, 16) => SampleFormat::S16,
            (1, 24) => SampleFormat::S24,
            (1, 32) => SampleFormat::S32,
            (3, 32) => SampleFormat::F32,
            _ => return None,
        })
    }
}

/// A concrete interleaved-PCM format: the runtime counterpart of a fixated `audio/raw`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: SampleFormat,
}

impl AudioFormat {
    pub const fn new(sample_rate: u32, channels: u16, format: SampleFormat) -> Self {
        Self {
            sample_rate,
            channels,
            format,
        }
    }

    /// Bytes in one interchannel frame (every channel, one sample each) — the stride a
    /// parser must never split across.
    pub const fn frame_stride(self) -> usize {
        self.channels as usize * self.format.bytes()
    }

    pub const fn bits_per_sample(self) -> u32 {
        self.format.bits()
    }
}

// --- The `audio/raw` negotiation offer ------------------------------------------------
//
// A runtime `AudioFormat` cannot be a `'static` offer (the descriptor layer needs
// `&'static` names), so a *parser* whose format is data-dependent advertises the broad
// [`RAW_ANY_OFFER`] and dictates the exact values downstream once runtime caps exist. A
// *fixed-format* element declares its own `static` offer using the [`FIELD_*`] constants
// and [`SampleFormat::caps_name`] — see `elements/tests/negotiation.rs` for the shape.

static SAMPLE_VALUES: [ValueDesc; 5] = [
    ValueDesc::Id("u8"),
    ValueDesc::Id("s16"),
    ValueDesc::Id("s24"),
    ValueDesc::Id("s32"),
    ValueDesc::Id("f32"),
];

static RAW_ANY_FIELDS: [FieldDesc; 3] = [
    FieldDesc {
        field: FIELD_RATE,
        allowed: ConstraintDesc::Any,
        preferred: None,
    },
    FieldDesc {
        field: FIELD_CHANNELS,
        allowed: ConstraintDesc::Any,
        preferred: None,
    },
    FieldDesc {
        field: FIELD_SAMPLE,
        allowed: ConstraintDesc::Set(&SAMPLE_VALUES),
        preferred: None,
    },
];

/// "Any interleaved PCM": matches any rate/channels and any known [`SampleFormat`]. The
/// offer list a raw-audio pad advertises when it accepts whatever the graph fixates.
pub static RAW_ANY_OFFER: [OfferDesc; 1] = [OfferDesc {
    family: FAMILY,
    fields: &RAW_ANY_FIELDS,
}];

/// A borrowed, validated view over interleaved PCM bytes (spec: typed format views). It
/// borrows, so misuse is a compile error and it adds nothing over raw offset math.
pub struct AudioFrameRef<'a> {
    bytes: &'a [u8],
    format: AudioFormat,
}

impl<'a> AudioFrameRef<'a> {
    /// Validate once that `bytes` is a whole number of interchannel frames, then view it.
    pub fn new(bytes: &'a [u8], format: AudioFormat) -> Option<Self> {
        let stride = format.frame_stride();
        if stride == 0 || bytes.len() % stride != 0 {
            return None;
        }
        Some(Self { bytes, format })
    }

    pub fn format(&self) -> AudioFormat {
        self.format
    }

    pub fn frames(&self) -> usize {
        self.bytes.len() / self.format.frame_stride()
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Read interchannel frame `frame`, channel `ch` as a sign-normalised `i32` (U8 is
    /// unbiased to signed; S24 is sign-extended). Integer formats only — see
    /// [`AudioFrameRef::sample_f32`] for `F32`.
    pub fn sample_i32(&self, frame: usize, ch: usize) -> i32 {
        debug_assert!(!self.format.format.is_float(), "use sample_f32 for float PCM");
        let bps = self.format.format.bytes();
        let off = frame * self.format.frame_stride() + ch * bps;
        let b = &self.bytes[off..off + bps];
        match self.format.format {
            SampleFormat::U8 => b[0] as i32 - 128,
            SampleFormat::S16 => i16::from_le_bytes([b[0], b[1]]) as i32,
            // Sign-extend the 24-bit value into an i32 by shifting up then arithmetic-down.
            SampleFormat::S24 => {
                let v = b[0] as i32 | (b[1] as i32) << 8 | (b[2] as i32) << 16;
                (v << 8) >> 8
            }
            SampleFormat::S32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            SampleFormat::F32 => 0, // unreachable in practice (debug_assert above)
        }
    }

    /// Read an `F32` sample. Defined only for the `F32` format.
    pub fn sample_f32(&self, frame: usize, ch: usize) -> f32 {
        debug_assert!(self.format.format.is_float(), "sample_f32 is for F32 PCM only");
        let bps = self.format.format.bytes();
        let off = frame * self.format.frame_stride() + ch * bps;
        let b = &self.bytes[off..off + bps];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_format_widths_and_names() {
        assert_eq!(SampleFormat::S16.bytes(), 2);
        assert_eq!(SampleFormat::S24.bytes(), 3);
        assert_eq!(SampleFormat::S24.bits(), 24);
        assert_eq!(SampleFormat::S16.caps_name(), "s16");
        assert_eq!(SampleFormat::from_caps_name("f32"), Some(SampleFormat::F32));
        assert_eq!(SampleFormat::from_caps_name("nope"), None);
        // caps_name and from_caps_name round-trip for every variant.
        for f in [
            SampleFormat::U8,
            SampleFormat::S16,
            SampleFormat::S24,
            SampleFormat::S32,
            SampleFormat::F32,
        ] {
            assert_eq!(SampleFormat::from_caps_name(f.caps_name()), Some(f));
        }
    }

    #[test]
    fn from_wav_maps_common_pcm() {
        assert_eq!(SampleFormat::from_wav(1, 16), Some(SampleFormat::S16));
        assert_eq!(SampleFormat::from_wav(1, 24), Some(SampleFormat::S24));
        assert_eq!(SampleFormat::from_wav(3, 32), Some(SampleFormat::F32));
        assert_eq!(SampleFormat::from_wav(1, 12), None, "odd bit depth unsupported");
        assert_eq!(SampleFormat::from_wav(7, 16), None, "a-law unsupported");
    }

    #[test]
    fn frame_stride() {
        let f = AudioFormat::new(48_000, 2, SampleFormat::S16);
        assert_eq!(f.frame_stride(), 4);
        assert_eq!(AudioFormat::new(44_100, 1, SampleFormat::S24).frame_stride(), 3);
    }

    #[test]
    fn frame_view_reads_interleaved_samples() {
        // Two stereo S16 frames: L,R = (1,-1), (32767,-32768).
        let mut bytes = Vec::new();
        for s in [1i16, -1, 32767, -32768] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let fmt = AudioFormat::new(48_000, 2, SampleFormat::S16);
        let view = AudioFrameRef::new(&bytes, fmt).expect("whole frames");
        assert_eq!(view.frames(), 2);
        assert_eq!(view.sample_i32(0, 0), 1);
        assert_eq!(view.sample_i32(0, 1), -1);
        assert_eq!(view.sample_i32(1, 0), 32767);
        assert_eq!(view.sample_i32(1, 1), -32768);
        // A ragged length (not a frame multiple) is rejected up front.
        assert!(AudioFrameRef::new(&bytes[..3], fmt).is_none());
    }

    #[test]
    fn s24_sign_extends() {
        // -1 in 24-bit two's complement is 0xFFFFFF little-endian.
        let bytes = [0xFF, 0xFF, 0xFF];
        let fmt = AudioFormat::new(48_000, 1, SampleFormat::S24);
        let view = AudioFrameRef::new(&bytes, fmt).unwrap();
        assert_eq!(view.sample_i32(0, 0), -1);
    }
}
