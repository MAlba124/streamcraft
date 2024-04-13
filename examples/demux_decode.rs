// Copyright (C) 2024  MAlba124 <marlhan@proton.me>
//
// Use StreamCraft to demux and decode video file

use std::path::PathBuf;

use streamcraft::{
    elements::av::{
        audiodecoder::AudioDecoder, demuxsrc::DemuxSrc, videodecoder::VideoDecoder,
        ResourceLocation,
    },
    pipeline::Pipeline,
};

fn main() {
    let mut demuxer = DemuxSrc::new(ResourceLocation::new_file(PathBuf::from(
        "/home/merb/Videos/tran.mp4",
    )))
    .unwrap();

    let vdecoder = VideoDecoder::new();
    let adecoder = AudioDecoder::new();

    demuxer.link_video_sink_element(0, vdecoder);
    demuxer.link_audio_sink_element(1, adecoder);

    let mut pipeline = Pipeline::new(demuxer);
    pipeline.init().unwrap();

    while pipeline.iter().is_ok() {}
}
