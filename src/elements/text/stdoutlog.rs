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

pub struct StdoutLog {
    parent: Parent,
}

impl Default for StdoutLog {
    fn default() -> Self {
        Self::new()
    }
}

///```text
///           +--------------------+
///           |______              |
/// Text ---->| sink |  StdoutLog  |
///           |^^^^^^              |
///           +--------------------+
///```
impl StdoutLog {
    pub fn new() -> Self {
        Self {
            parent: Parent::default(),
        }
    }

    fn run_loop(&self, text: String) -> bool {
        print!("{text}");

        true
    }
}

impl Element for StdoutLog {
    fn get_sink_type(&self) -> ElementType {
        ElementType::TextSink
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::One(CommonFormat::Text),
            srcs: Srcs::None,
        }
    }

    fn run(&mut self, parent_datagram_receiver: Receiver<Datagram>) -> Result<(), Error> {
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
                    Data::Text(text) => {
                        if !self.run_loop(text) {
                            break;
                        }
                    }
                    _ => return Err(Error::ReceivedInvalidDatagramFromParent),
                },
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
    StdoutLog,
    "stdoutlog"
}
