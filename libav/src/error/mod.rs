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

use std::fmt::{self, Display, Formatter};

#[derive(Debug)]
pub enum Error {
    FailedToOpenInput,
    FailedToFindBestStream,
    FailedToReadFrame,
    FailedToFindDecoder,
    FailedToCreateDecoder,
    FailedToCopyCodecParamsToDecoder,
    FailedToOpenCodec,
    FailedToAllocFrame,
    FailedToAllocPacket,
    FailedToSendPacketToDecoder,
    FailedToReceiveDecodedFrame,
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::FailedToOpenInput => "Failed to open input",
                Self::FailedToFindBestStream => "Failed to find best stream",
                Self::FailedToReadFrame => "Failed to read frame",
                Self::FailedToFindDecoder => "Failed to find decoder",
                Self::FailedToCreateDecoder => "Failed to create decoder",
                Self::FailedToCopyCodecParamsToDecoder => "Failed to copy codec params to decoder",
                Self::FailedToOpenCodec => "Failed to open codec",
                Self::FailedToAllocFrame => "Failed to alloc frame",
                Self::FailedToAllocPacket => "Failed to alloc packet",
                Self::FailedToSendPacketToDecoder => "Failed to send packet to decoder",
                Self::FailedToReceiveDecodedFrame => "Failed to receive decoded frame",
            }
        )
    }
}
