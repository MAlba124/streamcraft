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

use crate::{
    element_def,
    element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sink, Srcs},
    error,
    pipeline::{error::Error, Data, Datagram, Message, Parent, SinkPipe},
};

use crossbeam_channel::{bounded, unbounded, Receiver};

///```text
///               +-----------------------------+
///               |______                  _____|
/// AVPacket ---->| sink |  VideoDecoder  | src |----> // TODO
///               |^^^^^^                  ^^^^^|
///               +-----------------------------+
///```
pub struct VideoDecoder {
    sink: SinkPipe,
    parent: Parent,
}

impl Default for VideoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoDecoder {
    pub fn new() -> Self {
        Self {
            sink: SinkPipe::default(),
            parent: Parent::default(),
        }
    }

    /// Link the sink element.
    pub fn link_sink_element(&mut self, sink: impl Element + 'static) -> Result<(), Error> {
        if sink.get_sink_type() != ElementType::TextSink {
            return Err(Error::InvalidSinkType);
        }

        if let Sink::One(format) = sink.get_architecture().sink {
            if format == CommonFormat::Text {
                self.sink.set_element(sink);
                return Ok(());
            }
        }

        Err(Error::InvalidSinkType)
    }

    fn init(&mut self) -> Result<(), Error> {
        // let (datagram_sender, datagram_receiver) = bounded(0);
        // let (msg_sender, my_msg_receiver) = unbounded();
        // let parent = Parent::new(msg_sender);
        // let mut sink_element = self.sink.take_element()?;
        // sink_element.set_parent(parent);
        // let datagram_receiver_clone = datagram_receiver.clone();

        // self.sink.thread_handle = Some(std::thread::spawn(move || {
        //     match sink_element.run(datagram_receiver_clone) {
        //         Ok(_) => {}
        //         Err(e) => error!("{e}"),
        //     }
        // }));
        // self.sink.msg_receiver = Some(my_msg_receiver);
        // self.sink.datagram_sender = Some(datagram_sender);

        Ok(())
    }

    fn run_loop(&mut self, buf: Vec<u8>) -> bool {
        true
    }
}

impl Element for VideoDecoder {
    fn get_sink_type(&self) -> ElementType {
        ElementType::AVPacketVideoSink
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::One(CommonFormat::AVPacket),
            srcs: Srcs::One(CommonFormat::Text),
        }
    }

    fn run(&mut self, parent_datagram_receiver: Receiver<Datagram>) -> Result<(), Error> {
        self.init()?;

        loop {
            match parent_datagram_receiver
                .recv()
                .map_err(|_| Error::FailedToRecvFromParent)?
            {
                Datagram::Message(msg) => match msg {
                    Message::Quit => break,
                    _ => return Err(Error::ReceivedInvalidDatagramFromParent),
                },
                Datagram::Data(data) => match data {
                    Data::Bytes(bytes) => {
                        if !self.run_loop(bytes) {
                            break;
                        }
                    }
                    _ => {
                        error!("Received invalid data type");
                        break;
                    }
                },
            }

            while let Some(_msg) = self.sink.try_recv_msg()? {
                // TODO: Handle messages
            }
        }

        self.parent.send_finished()
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }

    fn cleanup(&mut self) -> Result<(), Error> {
        self.sink.send_quit()?;
        self.sink.drop_data_sender();

        self.sink.join_thread()
    }
}

element_def! {
    VideoDecoder,
    "videodecoder"
}

#[cfg(test)]
mod tests {
    use super::*;
}
