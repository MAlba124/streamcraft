//! `play_mkv` — **milestone 5**: play a video file (spec: Milestone applications §5),
//! `filesrc ! mkvdemux ! vp8dec ! sdl3videosink`, clock-paced to a real window.
//!
//! ```text
//! cargo run --release -p pf-mkv --example make_vp8_sample -- /tmp/sample.mkv 5
//! cargo run --release -p pf-sdl3 --example play_mkv -- /tmp/sample.mkv
//! ```
//!
//! `MkvDemux` discovers its tracks during preroll from constructor-supplied header
//! bytes (mid-pipeline elements get no preroll input — see mkv's crate docs), so the
//! example reads the stream head up to the first Cluster and hands it over; the whole
//! file then streams through `FileSrc` as usual.

use pf_mkv::ebml::id;
use pf_mkv::MkvDemux;
use pf_vp8::Vp8Dec;
use pf_sdl3::Sdl3VideoSink;
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::FileSrc;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: play_mkv FILE.mkv");
        std::process::exit(2);
    };

    // Header prefix (EBML Header + Segment Info + Tracks): everything before the
    // first Cluster. Reading the head of the file suffices for a well-formed
    // non-faststart MKV; 1 MiB is generous.
    let head = std::fs::read(&path).expect("read file");
    let cluster = head
        .windows(4)
        .position(|w| w == id::CLUSTER)
        .expect("no Cluster element — not a (supported) MKV?");
    let header = head[..cluster].to_vec();
    drop(head);

    let mut p = Pipeline::new();
    // Fit decoded frames (w*h*3/2) comfortably; 2 MiB covers up to ~1152x864 I420.
    p.set_pool(2 * 1024 * 1024, 32);

    let src = p.add(FileSrc::new(&path));
    let demux = p.add(MkvDemux::new(header));
    p.link((src, "src"), (demux, "sink")).expect("src ! demux");

    let added = p.preroll().expect("preroll");
    let video = added.first().expect("no tracks discovered");
    println!("track pad: {}", video.name);

    let dec = p.add(Vp8Dec::new());
    let sink = p.add(Sdl3VideoSink::new());
    p.link((video.element, &video.name), (dec, "sink")).expect("demux ! dec");
    p.link((dec, "src"), (sink, "sink")).expect("dec ! sink");

    println!("playing {path} — close the window or wait for EOS…");
    match p.run() {
        Ok(()) => println!("done."),
        Err(e) => {
            eprintln!("run error: {e:?}");
            std::process::exit(1);
        }
    }
}
