// Copyright (C) 2024  MAlba124 <marlhan@proton.me>
//
// StreamCraft is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// StreamCraft is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with StreamCraft.  If not, see <https://www.gnu.org/licenses/>.

use std::ffi::CString;

use crate::{bindings, decoding::Frame, demuxing::Packet, error::Error};

// To CString
pub enum Codec {
    H264,
    H265,
    Opus,
}

impl Codec {
    pub fn to_cstring(&self) -> Result<CString, Error> {
        Ok(CString::new(
            match self {
                Codec::H264 => "libx264",
                Codec::H265 => "libx265",
                Codec::Opus => "libopus",
            }
        ).unwrap()) // TODO: Return appropriate error
    }
}

pub struct Encoder {
    ctx: *mut bindings::AVCodecContext,
}

unsafe impl Send for Encoder {}
unsafe impl Sync for Encoder {}

impl Encoder {
    pub fn new(codec: Codec) -> Result<Self, Error> {
        unsafe {
            let codec_name = codec.to_cstring().unwrap();
            let codec = bindings::avcodec_find_encoder_by_name(codec_name.into_raw());
            if codec.is_null() {
                panic!("Could not find encoder `libx264`");
            }

            let ctx = bindings::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                panic!("Could not allocate video codec context");
            }

            let ret = bindings::avcodec_open2(ctx, codec, std::ptr::null_mut());
            if ret < 0 {
                panic!("Could not open codec");
            }

            Ok(Self {
                ctx
            })
        }
    }

    // TODO: Error handle
    pub fn encode_frame(&self, frame: Frame) {
        unsafe {
            let ret = bindings::avcodec_send_frame(self.ctx, frame.inner);
            if ret < 0 {
                panic!("Error sending a frame for encoding");
            }

            let mut packets = Vec::new();
            loop {
                let packet = Packet::new().unwrap();
                let ret = bindings::avcodec_receive_packet(self.ctx, packet.inner);
                if ret == bindings::sc_libav_averror_eagain || ret == bindings::sc_libav_averror_eof {
                    break;
                } else if ret < 0 {
                    panic!("Error during encoding");
                }

                packets.push(packet);
            }
        }
    }
}
