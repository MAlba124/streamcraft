//! `WavParse` in a pipeline: `filesrc ! wavparse ! filesink` strips the RIFF header and
//! streams the PCM payload byte-identically (spec: Milestone applications §3).

use profluens_audio::{parse_wav_header, write_pcm_wav, AudioFormat, SampleFormat, WavParse};
use profluens_core::event::Event;
use profluens_core::harness::Harness;
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

#[test]
fn flush_rewinds_the_payload_cursor_so_repeated_seeks_do_not_truncate() {
    // `data_emitted` clamps output to the declared `data_len`. Before the flush handler it was
    // a running *total*, so every seek added another pass to it; once the sum passed `data_len`
    // the clamp swallowed everything after it and playback silently stopped. Ten seeks back to
    // the top of the payload must each yield the whole payload.
    //
    // The harness installs no seek state, so `ctx.seek_target()` is `None` here — the
    // "flush that is not a seek" path, which rewinds to the start of the data chunk. The
    // `to_byte`-derived arithmetic is unit-tested in `wav.rs` (a `SeekTarget` cannot be
    // installed on a `Ctx` from outside `profluens-core`).
    let fmt = AudioFormat::new(44_100, 2, SampleFormat::S16);
    let pcm: Vec<u8> = (0..4_000u16).flat_map(|i| i.to_le_bytes()).collect();
    let wav = write_pcm_wav(&fmt, &pcm);

    let mut h = Harness::with_slot_size(WavParse::new(), 1 << 16);
    h.start().expect("start");

    let drain = |h: &mut Harness| {
        let mut got = Vec::new();
        while let Some(b) = h.pull("src") {
            got.extend_from_slice(b.memory.data());
        }
        got
    };

    // First pass: the whole file, header and all.
    let buf = h.alloc(&wav);
    h.push("sink", buf).expect("push");
    assert_eq!(drain(&mut h), pcm, "first pass: header stripped, payload streamed");

    // Then ten seeks back to the start of the payload. A source resumes reading at the seek
    // target's byte offset, so only PCM arrives; the parsed header stays.
    for pass in 0..10 {
        h.push_event(Event::FlushStart).expect("flush");
        let buf = h.alloc(&pcm);
        h.push("sink", buf).expect("push");
        let got = drain(&mut h);
        assert_eq!(
            got.len(),
            pcm.len(),
            "pass {pass}: the data_len clamp truncated the payload ({} of {} bytes)",
            got.len(),
            pcm.len()
        );
        assert_eq!(got, pcm, "pass {pass}: payload changed");
    }
}
