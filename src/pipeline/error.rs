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

#[derive(Debug)]
pub enum Error {
    NoThreadHandle,
    FailedToJoinThread,
    NoSinkMessageSender,
    NoSinkMessageReceiver,
    MessageSinkFailed,
    NoSinkElement,
    PipelineNotReady,
    NoSinkDatagramSender,
    FailedToRecvFromParent,
    ReceivedInvalidDatagramFromParent,
    ReceivedInvalidDatagramFromSink,
    ReceiveFromSinkFailed,
    MessageParentFailed,
    NoParentMessageSender,
    InvalidSinkType,
    FailedToSendDatagramToSink,
    AVError(libav::error::Error),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::NoThreadHandle => "No thread handle".to_string(),
                Self::FailedToJoinThread => "Failed to join thread".to_string(),
                Self::NoSinkMessageSender => "No sink message sender".to_string(),
                Self::NoSinkMessageReceiver => "No sink message receiver".to_string(),
                Self::MessageSinkFailed => "Message sink failed".to_string(),
                Self::NoSinkElement => "No sink element".to_string(),
                Self::PipelineNotReady => "Pipeline is not ready".to_string(),
                Self::NoSinkDatagramSender => "No sink datagram sender".to_string(),
                Self::FailedToRecvFromParent => "Failed to recv from parent".to_string(),
                Self::ReceivedInvalidDatagramFromParent =>
                    "Received invalid datagram from parent".to_string(),
                Self::ReceivedInvalidDatagramFromSink =>
                    "Received invalid datagram from sink".to_string(),
                Self::ReceiveFromSinkFailed => "Receive from sink failed".to_string(),
                Self::MessageParentFailed => "Message parent failed".to_string(),
                Self::NoParentMessageSender => "No parent message sender".to_string(),
                Self::InvalidSinkType => "Invalid sink type".to_string(),
                Self::FailedToSendDatagramToSink => "Failed to send datagram to sink".to_string(),
                Self::AVError(e) => format!("AVError: {e}"),
            }
        )
    }
}
