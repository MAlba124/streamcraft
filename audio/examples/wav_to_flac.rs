//! Milestone 3: encode a WAV file to FLAC — `filesrc ! wavparse ! flacenc ! filesink`.
//!
//! Run: `cargo run --release -p streamcraft-audio --example wav_to_flac -- [in.wav] [out.flac]`
//! With no args it synthesises a 1-second stereo sine WAV in the temp dir.
//!
//! Because format negotiation is still link-time only, the encoder's parameters
//! (rate/channels/format) are read from the WAV header up front and handed to
//! `FlacEnc::new` — the runtime-caps path that would let `wavparse` dictate them
//! downstream is a follow-up.

use sc_flac::{FlacDecoder, FlacEnc, SampleFormat as FlacFmt};
use streamcraft_audio::{parse_wav_header, write_pcm_wav, AudioFormat, SampleFormat, WavParse};
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::{FileSink, FileSrc};

fn map_format(f: SampleFormat) -> Option<FlacFmt> {
    Some(match f {
        SampleFormat::S16 => FlacFmt::S16,
        SampleFormat::S24 => FlacFmt::S24,
        SampleFormat::S32 => FlacFmt::S32,
        _ => return None, // U8/F32 are not on the integer-PCM FLAC path
    })
}

fn synth_wav(path: &std::path::Path) {
    let fmt = AudioFormat::new(44_100, 2, SampleFormat::S16);
    let n = fmt.sample_rate as usize; // one second
    let mut pcm = Vec::with_capacity(n * 4);
    for i in 0..n {
        let t = i as f64 / fmt.sample_rate as f64;
        let l = (10_000.0 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as i16;
        let r = (10_000.0 * (2.0 * std::f64::consts::PI * 554.37 * t).sin()) as i16;
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&r.to_le_bytes());
    }
    std::fs::write(path, write_pcm_wav(&fmt, &pcm)).unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let inp = args.next().map(std::path::PathBuf::from).unwrap_or_else(|| {
        let p = std::env::temp_dir().join("sc_wav_to_flac_in.wav");
        synth_wav(&p);
        p
    });
    let outp = args
        .next()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sc_wav_to_flac_out.flac"));

    // Parse the header (the first few KiB is plenty) to configure the encoder.
    let head = {
        use std::io::Read;
        let mut f = std::fs::File::open(&inp).expect("open input");
        let mut buf = vec![0u8; 8192];
        let n = f.read(&mut buf).expect("read head");
        buf.truncate(n);
        buf
    };
    let fmt = parse_wav_header(&head).expect("parse WAV header").format;
    let flac_fmt = map_format(fmt.format).unwrap_or_else(|| {
        eprintln!("unsupported sample format {:?}", fmt.format);
        std::process::exit(1);
    });

    println!(
        "in : {} ({} Hz, {} ch, {:?})",
        inp.display(),
        fmt.sample_rate,
        fmt.channels,
        fmt.format
    );

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&inp));
    let wav = p.add(WavParse::new());
    let enc = p.add(FlacEnc::new(fmt.sample_rate, fmt.channels as u32, flac_fmt));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (wav, "sink")).unwrap();
    p.link((wav, "src"), (enc, "sink")).unwrap();
    p.link((enc, "src"), (sink, "sink")).unwrap();
    p.run().expect("pipeline run");

    let in_sz = std::fs::metadata(&inp).map(|m| m.len()).unwrap_or(0);
    let out_sz = std::fs::metadata(&outp).map(|m| m.len()).unwrap_or(0);
    // Decode the output back to confirm it is a well-formed FLAC stream.
    let flac = std::fs::read(&outp).unwrap();
    let dec = FlacDecoder::decode(&flac).expect("output is decodable FLAC");
    println!(
        "out: {} ({} bytes, {:.1}% of WAV) — decoded {} samples @ {} Hz, {}-bit",
        outp.display(),
        out_sz,
        100.0 * out_sz as f64 / in_sz.max(1) as f64,
        dec.samples.len(),
        dec.info.sample_rate,
        dec.info.bits_per_sample
    );
}
