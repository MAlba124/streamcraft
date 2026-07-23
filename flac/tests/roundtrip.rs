//! The correctness gate: encode → decode is **bit-exact lossless** (spec: Milestone
//! applications §3; First-party codecs — conformance is the primary net).
//!
//! Table-driven over generated PCM: sines, white noise, silence, full-scale and
//! min/max amplitudes, DC, ramps, and impulses; mono and stereo; several sample
//! formats; and a spread of block sizes including odd/edge sizes (§9.2.7 forces
//! partition order 0 for odd blocks, which this exercises). Each case is encoded with
//! [`FlacEncoder`], decoded with [`FlacDecoder`], and every sample compared exactly.

use sc_flac::{FlacDecoder, FlacEncoder, SampleFormat};

/// Deterministic PRNG (SplitMix64) so "white noise" cases are reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The kind of signal to synthesise, in units of raw sample values (already ranged to
/// the format's bit depth by the generator).
#[derive(Clone, Copy, Debug)]
enum Signal {
    Silence,
    Dc(i64),
    FullScalePositive,
    FullScaleNegative,
    /// Alternating +max / -max — worst case for a low-order predictor.
    Square,
    Sine { periods: f64 },
    Ramp,
    Noise { seed: u64 },
    /// A single spike in an otherwise-silent block.
    Impulse,
}

fn sample_limits(fmt: SampleFormat) -> (i64, i64) {
    let bits = fmt.bits_per_sample();
    let max = (1i64 << (bits - 1)) - 1;
    let min = -(1i64 << (bits - 1));
    (min, max)
}

/// Build `n` interchannel samples for `channels` as planar i64, per the signal.
fn gen_planar(sig: Signal, fmt: SampleFormat, channels: u32, n: usize) -> Vec<Vec<i64>> {
    let (min, max) = sample_limits(fmt);
    let mut chans = Vec::with_capacity(channels as usize);
    for c in 0..channels {
        let mut v = Vec::with_capacity(n);
        let mut rng = Rng(0xC0FF_EE00 ^ ((c as u64) << 32));
        for i in 0..n {
            let s = match sig {
                Signal::Silence => 0,
                Signal::Dc(d) => d.clamp(min, max),
                Signal::FullScalePositive => max,
                Signal::FullScaleNegative => min,
                Signal::Square => {
                    if (i + c as usize) % 2 == 0 {
                        max
                    } else {
                        min
                    }
                }
                Signal::Sine { periods } => {
                    let phase =
                        2.0 * std::f64::consts::PI * periods * (i as f64) / (n.max(1) as f64);
                    // Different amplitude/offset per channel so stereo isn't trivial.
                    let amp = (max as f64) * (0.9 - 0.1 * c as f64);
                    (amp * (phase + c as f64).sin()).round() as i64
                }
                Signal::Ramp => {
                    let span = (max - min) as i128;
                    (min as i128 + span * (i as i128) / (n.max(1) as i128)) as i64
                }
                Signal::Noise { seed } => {
                    let r = Rng(seed ^ ((c as u64) << 40) ^ rng.next_u64()).next_u64();
                    (r % ((max - min + 1) as u64)) as i64 + min
                }
                Signal::Impulse => {
                    if i == n / 2 {
                        max
                    } else {
                        0
                    }
                }
            };
            v.push(s.clamp(min, max));
        }
        chans.push(v);
    }
    chans
}

/// Interleave planar channels into little-endian bytes of the given format.
fn interleave_le(planar: &[Vec<i64>], fmt: SampleFormat) -> Vec<u8> {
    let n = planar[0].len();
    let bps = fmt.bytes_per_sample();
    let mut out = Vec::with_capacity(n * planar.len() * bps);
    for i in 0..n {
        for ch in planar {
            let s = ch[i];
            match fmt {
                SampleFormat::S8 => out.push(s as i8 as u8),
                SampleFormat::S16 => out.extend_from_slice(&(s as i16).to_le_bytes()),
                SampleFormat::S24 => {
                    let u = (s as i32) as u32;
                    out.push((u & 0xFF) as u8);
                    out.push(((u >> 8) & 0xFF) as u8);
                    out.push(((u >> 16) & 0xFF) as u8);
                }
                SampleFormat::S32 => out.extend_from_slice(&(s as i32).to_le_bytes()),
            }
        }
    }
    out
}

/// Encode planar PCM to a complete FLAC stream (header patched with the finalised
/// STREAMINFO), then decode it back and return the decoded interleaved samples.
fn round_trip(planar: &[Vec<i64>], fmt: SampleFormat, rate: u32) -> (Vec<i64>, usize, usize) {
    let channels = planar.len() as u32;
    let interleaved = interleave_le(planar, fmt);

    let (mut enc, mut header) = FlacEncoder::new(rate, channels, fmt).expect("encoder");
    let mut frames = Vec::new();
    enc.encode_interleaved(&interleaved, &mut frames).expect("encode");
    let body = enc.finish();
    // Patch the finalised STREAMINFO body over the placeholder.
    header[sc_flac::streaminfo_offset()..sc_flac::streaminfo_offset() + body.len()]
        .copy_from_slice(&body);

    let mut stream = header;
    stream.extend_from_slice(&frames);
    let encoded_len = stream.len();

    let dec = FlacDecoder::decode(&stream).expect("decode");
    assert_eq!(dec.info.channels, channels);
    assert_eq!(dec.info.sample_rate, rate);
    assert_eq!(dec.info.bits_per_sample, fmt.bits_per_sample());
    (dec.samples, encoded_len, interleaved.len())
}

/// Flatten planar to interleaved i64 (the decoder's output order) for comparison.
fn expected_interleaved(planar: &[Vec<i64>]) -> Vec<i64> {
    let n = planar[0].len();
    let mut out = Vec::with_capacity(n * planar.len());
    for i in 0..n {
        for ch in planar {
            out.push(ch[i]);
        }
    }
    out
}

#[test]
fn lossless_round_trip_table() {
    let signals = [
        Signal::Silence,
        Signal::Dc(1234),
        Signal::Dc(-5),
        Signal::FullScalePositive,
        Signal::FullScaleNegative,
        Signal::Square,
        Signal::Sine { periods: 1.0 },
        Signal::Sine { periods: 7.5 },
        Signal::Sine { periods: 63.0 },
        Signal::Ramp,
        Signal::Noise { seed: 1 },
        Signal::Noise { seed: 0xABCD },
        Signal::Impulse,
    ];
    // Block sizes: powers of two, odd (force partition order 0), primes, tiny, and a
    // size that splits into multiple frames (> 4608).
    let block_sizes = [1usize, 2, 3, 15, 16, 17, 31, 192, 577, 1024, 4096, 4609, 10000];
    let formats = [
        SampleFormat::S8,
        SampleFormat::S16,
        SampleFormat::S24,
        SampleFormat::S32,
    ];
    let rates = [44100u32, 48000];

    let mut cases = 0usize;
    for &fmt in &formats {
        for &channels in &[1u32, 2] {
            for &n in &block_sizes {
                for &sig in &signals {
                    let rate = rates[n % rates.len()];
                    let planar = gen_planar(sig, fmt, channels, n);
                    let (decoded, _enc_len, _raw_len) = round_trip(&planar, fmt, rate);
                    let expected = expected_interleaved(&planar);
                    assert_eq!(
                        decoded.len(),
                        expected.len(),
                        "sample count mismatch: fmt={fmt:?} ch={channels} n={n} sig={sig:?}"
                    );
                    assert_eq!(
                        decoded, expected,
                        "LOSSY round trip: fmt={fmt:?} ch={channels} n={n} sig={sig:?} rate={rate}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 1000, "expected a broad table, ran {cases} cases");
}

#[test]
fn empty_stream_is_valid() {
    // Zero samples: header only, no frames. Decoding must succeed with no samples.
    let (mut enc, mut header) = FlacEncoder::new(44100, 2, SampleFormat::S16).unwrap();
    let mut frames = Vec::new();
    enc.encode_interleaved(&[], &mut frames).unwrap();
    let body = enc.finish();
    header[sc_flac::streaminfo_offset()..sc_flac::streaminfo_offset() + body.len()]
        .copy_from_slice(&body);
    assert!(frames.is_empty());
    let dec = FlacDecoder::decode(&header).unwrap();
    assert!(dec.samples.is_empty());
    assert_eq!(dec.info.total_samples, 0);
}

#[test]
fn compression_beats_verbatim_on_a_sine() {
    // A pure tone must compress well below the raw PCM size (FIXED predictors + Rice).
    let fmt = SampleFormat::S16;
    let planar = gen_planar(Signal::Sine { periods: 20.0 }, fmt, 2, 8192);
    let (decoded, enc_len, raw_len) = round_trip(&planar, fmt, 44100);
    assert_eq!(decoded, expected_interleaved(&planar));
    assert!(
        (enc_len as f64) < 0.8 * raw_len as f64,
        "expected < 80% of raw ({raw_len}), got {enc_len}"
    );
}
