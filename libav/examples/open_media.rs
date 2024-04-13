// Copyright (C) 2024  MAlba124 <marlhan@proton.me>
use libav::decoding::Decoder;
use libav::demuxing::{Demuxer, ResourceLocation};

fn main() {
    let demuxer = Demuxer::new(ResourceLocation::new_file(std::path::PathBuf::from(
        "/home/merb/Videos/tran.mp4",
    )))
    .unwrap();

    let video_decoder = Decoder::new(demuxer.get_video_stream().unwrap());
    let audio_decoder = Decoder::new(demuxer.get_audio_stream().unwrap());

    // demuxer.read_frame().unwrap();
}
