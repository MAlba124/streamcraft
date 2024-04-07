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

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{
    debug, element_def,
    element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sink, Srcs},
    error,
    pipeline::{error::Error, Data, Datagram, Message, Parent, SinkPipe},
};

use crossbeam_channel::{bounded, unbounded, Receiver};

///```text
/// +--------------------+
/// |               _____|
/// |  FileSrc     | src |----> Bytes
/// |               ^^^^^|
/// +--------------------+
///```
pub struct FileSrc {
    sink: SinkPipe,
    parent: Parent,
    reader: BufReader<File>,
}

impl FileSrc {
    pub fn new(file: File) -> Self {
        Self {
            sink: SinkPipe::default(),
            parent: Parent::default(),
            reader: BufReader::new(file),
        }
    }

    pub fn link_sink_element(&mut self, sink: impl Element + 'static) -> Result<(), Error> {
        if sink.get_sink_type() != ElementType::BytesSink {
            return Err(Error::InvalidSinkType);
        }

        if let Sink::One(format) = sink.get_architecture().sink {
            if format == CommonFormat::Bytes {
                self.sink.set_element(sink);
                return Ok(());
            }
        }

        Err(Error::InvalidSinkType)
    }

    fn init(&mut self) -> Result<(), Error> {
        let (datagram_sender, datagram_receiver) = bounded(0);
        let (msg_sender, my_msg_receiver) = unbounded();
        let parent = Parent::new(msg_sender);
        let mut sink_element = self.sink.take_element()?;
        sink_element.set_parent(parent);
        let datagram_receiver_clone = datagram_receiver.clone();

        self.sink.thread_handle = Some(std::thread::spawn(move || {
            match sink_element.run(datagram_receiver_clone) {
                Ok(_) => {}
                Err(e) => error!("Error occurred running sink element: {e}"),
            }
        }));
        self.sink.msg_receiver = Some(my_msg_receiver);
        self.sink.datagram_sender = Some(datagram_sender);

        Ok(())
    }

    fn run_loop(&mut self) -> bool {
        let buf = match self.reader.fill_buf() {
            Ok(buf) => buf.to_vec(),
            Err(_) => return false,
        };

        if buf.is_empty() {
            return false;
        }

        self.reader.consume(buf.len());

        if self
            .sink
            .send_datagram(Datagram::Data(Data::Bytes(buf)))
            .is_err()
        {
            return false;
        }

        true
    }
}

impl Element for FileSrc {
    fn get_sink_type(&self) -> ElementType {
        ElementType::BytesSrc
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::None,
            srcs: Srcs::One(CommonFormat::Bytes),
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
                    Message::Iter => {
                        if !self.run_loop() {
                            debug!("Finished");
                            break;
                        }
                        self.parent.send_iter_fin()?;
                    }
                    Message::Quit => break,
                    _ => return Err(Error::ReceivedInvalidDatagramFromParent),
                },
                _ => return Err(Error::ReceivedInvalidDatagramFromParent),
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
    FileSrc,
    "filesrc"
}
