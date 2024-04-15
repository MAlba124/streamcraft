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

use crate::{bindings, demuxing::Packet, error::Error};

// TODO: Move somewhre else
pub struct Frame {
    pub inner: *mut bindings::AVFrame,
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe {
            bindings::av_frame_free(&mut self.inner);
        }
    }
}

impl Frame {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            let inner = bindings::av_frame_alloc();

            if inner.is_null() {
                return Err(Error::FailedToAllocFrame);
            }

            Ok(Self { inner })
        }
    }

    pub fn get_pts(&self) -> i64 {
        unsafe { (*self.inner).pts }
    }
}

pub struct Decoder {
    stream_index: i32,
    ctx: *mut bindings::AVCodecContext,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            bindings::avcodec_free_context(&mut self.ctx);
        }
    }
}

impl Decoder {
    pub fn new(
        (stream_index, codec_id, params): (i32, crate::demuxing::CodecID, crate::demuxing::CodecParams),
    ) -> Result<Self, Error> {
        let decoder = unsafe { bindings::avcodec_find_decoder(codec_id) };

        if decoder.is_null() {
            return Err(Error::FailedToFindDecoder);
        }

        let decoder_ctx = unsafe { bindings::avcodec_alloc_context3(decoder) };

        if decoder_ctx.is_null() {
            return Err(Error::FailedToCreateDecoder);
        }

        if unsafe { bindings::avcodec_parameters_to_context(decoder_ctx, &params.inner) } < 0 {
            return Err(Error::FailedToCopyCodecParamsToDecoder);
        }

        if unsafe { bindings::avcodec_open2(decoder_ctx, decoder, std::ptr::null_mut()) } < 0 {
            return Err(Error::FailedToOpenCodec);
        }

        Ok(Self {
            stream_index,
            ctx: decoder_ctx,
        })
    }

    pub fn decode_packet(&self, packet: Packet) -> Result<Vec<Frame>, Error> {
        assert_ne!(packet.stream_index(), self.stream_index);

        if unsafe { bindings::avcodec_send_packet(self.ctx, packet.inner) } < 0 {
            return Err(Error::FailedToSendPacketToDecoder);
        }

        let mut frames = Vec::new();
        loop {
            let frame = Frame::new()?;

            let ret = unsafe { bindings::avcodec_receive_frame(self.ctx, frame.inner) };
            if ret < 0 {
                if ret == bindings::sc_libav_averror_eof || ret == bindings::sc_libav_averror_eagain {
                    break;
                }

                return Err(Error::FailedToReceiveDecodedFrame);
            }

            frames.push(frame);
        }

        Ok(frames)
    }
}
