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
    pipeline::{error::Error, Data, Datagram, Message, Parent},
};

use crossbeam_channel::Receiver;

pub struct TestSink {
    parent: Parent,
    sink_type: ElementType,
    sink_format: CommonFormat,
    on_message: Box<dyn Fn(usize, Message) -> bool + Send + Sync>,
    on_data: Box<dyn Fn(usize, Data) -> bool + Send + Sync>,
    message_count: usize,
    data_count: usize,
}

///```text
///           +-------------------+
///           |______             |
/// ???? ---->| sink |  TestSink  |
///           |^^^^^^             |
///           +-------------------+
///```
impl TestSink {
    pub fn new(
        sink_type: ElementType,
        sink_format: CommonFormat,
        on_message: impl Fn(usize, Message) -> bool + 'static + Send + Sync,
        on_data: impl Fn(usize, Data) -> bool + 'static + Send + Sync,
    ) -> Self {
        Self {
            parent: Parent::default(),
            sink_type,
            sink_format,
            on_message: Box::new(on_message),
            on_data: Box::new(on_data),
            message_count: 0,
            data_count: 0,
        }
    }
}

impl Element for TestSink {
    fn get_sink_type(&self) -> ElementType {
        self.sink_type.clone()
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::One(self.sink_format.clone()),
            srcs: Srcs::None,
        }
    }

    fn run(&mut self, parent_datagram_receiver: Receiver<Datagram>) -> Result<(), Error> {
        loop {
            match parent_datagram_receiver
                .recv()
                .map_err(|_| Error::FailedToRecvFromParent)?
            {
                Datagram::Message(msg) => {
                    if !(self.on_message)(self.message_count, msg) {
                        break;
                    }
                    self.message_count += 1;
                }
                Datagram::Data(data) => {
                    if !(self.on_data)(self.data_count, data) {
                        break;
                    }
                    self.data_count += 1;
                }
            }
        }

        Ok(())
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }

    fn cleanup(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

element_def! {
    TestSink,
    "testsink"
}
