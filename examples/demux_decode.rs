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

    let vdecoder = VideoDecoder::new(demuxer.get_video_stream().unwrap()).unwrap();
    let adecoder = AudioDecoder::new(demuxer.get_audio_stream().unwrap()).unwrap();

    demuxer
        .link_video_sink_element(vdecoder.get_stream_index(), vdecoder)
        .unwrap();
    demuxer
        .link_audio_sink_element(adecoder.get_stream_index(), adecoder)
        .unwrap();

    let mut pipeline = Pipeline::new(demuxer);
    pipeline.init().unwrap();

    for _ in 0..15 {
        pipeline.iter().unwrap();
    }
}
