//! `WavParse` in a pipeline: `filesrc ! wavparse ! filesink` strips the RIFF header and
//! streams the PCM payload byte-identically (spec: Milestone applications §3).

use profluens_audio::{parse_wav_header, write_pcm_wav, AudioFormat, SampleFormat, WavParse};
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::{FileSink, FileSrc};

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_wav_{}_{}.bin", tag, std::process::id()));
    p
}

#[test]
fn wavparse_strips_header_and_streams_pcm() {
    // > 128 KiB of PCM so several pooled buffers flow and `emit_data` runs across buffers.
    let fmt = AudioFormat::new(44_100, 2, SampleFormat::S16);
    let frames = 90_000usize; // stereo s16 → 360000 payload bytes
    let mut pcm = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let l = (i as i16).wrapping_mul(31);
        let r = (i as i16).wrapping_mul(-17).wrapping_add(5);
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&r.to_le_bytes());
    }
    let wav = write_pcm_wav(&fmt, &pcm);
    // Sanity: our own writer parses back to the same format.
    assert_eq!(parse_wav_header(&wav).unwrap().format, fmt);

    let inp = temp_path("in");
    let outp = temp_path("out");
    std::fs::write(&inp, &wav).unwrap();

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&inp));
    let wavp = p.add(WavParse::new());
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (wavp, "sink")).expect("link src->wavparse");
    p.link((wavp, "src"), (sink, "sink")).expect("link wavparse->sink");
    p.run().expect("run");

    let got = std::fs::read(&outp).unwrap();
    assert_eq!(got.len(), pcm.len(), "payload length preserved (header stripped)");
    assert_eq!(got, pcm, "PCM streamed byte-identically");

    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
}
