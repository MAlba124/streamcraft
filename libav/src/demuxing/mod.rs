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

use crate::{bindings, error::Error};

use std::path::PathBuf;

use std::ffi::CString;

pub enum ResourceLocation {
    File(PathBuf),
}

impl ResourceLocation {
    pub fn new_file(path: PathBuf) -> Self {
        Self::File(path)
    }
}

impl std::string::ToString for ResourceLocation {
    fn to_string(&self) -> String {
        match self {
            Self::File(path) => format!("file:{}", path.to_str().unwrap()), // TODO: No unwrap
        }
    }
}

// HACK: wtf is this
#[derive(PartialEq, Debug, Clone)]
pub struct Packet {
    pub inner: *mut bindings::AVPacket,
}

unsafe impl Send for Packet {}
unsafe impl Sync for Packet {}

impl Packet {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            let inner = bindings::av_packet_alloc();

            if inner.is_null() {
                return Err(Error::FailedToAllocPacket)
            }

            Ok(Self { inner })
        }
    }

    #[inline(always)]
    pub fn stream_index(&self) -> i32 {
        unsafe { (*self.inner).stream_index }
    }
}

impl Drop for Packet {
    fn drop(&mut self) {
        unsafe {
            bindings::av_packet_unref(self.inner);
        }
    }
}

pub type CodecID = bindings::AVCodecID;

pub struct CodecParams {
    pub inner: bindings::AVCodecParameters,
}

pub struct Demuxer {
    inner: *mut bindings::AVFormatContext,
}

unsafe impl Send for Demuxer {}
unsafe impl Sync for Demuxer {}

impl Demuxer {
    pub fn new(rl: ResourceLocation) -> Result<Self, Error> {
        let mut inner = std::ptr::null_mut();

        let uri = CString::new(rl.to_string()).unwrap();
        let ret = unsafe {
            bindings::avformat_open_input(
                &mut inner,
                uri.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };

        if ret < 0 {
            unsafe {
                bindings::avformat_close_input(&mut inner);
            }
            return Err(Error::FailedToOpenInput);
        }

        Ok(Self { inner })
    }

    // TODO: Return struct
    fn find_stream(
        &self,
        type_: bindings::AVMediaType,
    ) -> Result<(i32, CodecID, CodecParams), Error> {
        unsafe {
            let ret =
                bindings::av_find_best_stream(self.inner, type_, -1, -1, std::ptr::null_mut(), 0);

            if ret < 0 {
                return Err(Error::FailedToFindBestStream);
            }

            let stream_index = ret;
            let params = (*(*(*self.inner).streams.wrapping_add(stream_index as usize))).codecpar;
            let codec_id = (*params).codec_id;

            Ok((stream_index, codec_id, CodecParams {inner: *params}))
        }
    }

    pub fn get_video_stream(
        &self,
    ) -> Result<(i32, CodecID, CodecParams), Error> {
        self.find_stream(bindings::AVMediaType_AVMEDIA_TYPE_VIDEO)
    }

    pub fn get_audio_stream(
        &self,
    ) -> Result<(i32, CodecID, CodecParams), Error> {
        self.find_stream(bindings::AVMediaType_AVMEDIA_TYPE_AUDIO)
    }

    pub fn read_frame(&self) -> Result<Packet, Error> {
        unsafe {
            let packet = Packet::new()?;

            if bindings::av_read_frame(self.inner, packet.inner) < 0 {
                return Err(Error::FailedToReadFrame);
            }

            Ok(packet)
        }
    }
}

impl Drop for Demuxer {
    fn drop(&mut self) {
        unsafe {
            bindings::avformat_close_input(&mut self.inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_packet() {
        let packet = Packet::new();
    }
}
