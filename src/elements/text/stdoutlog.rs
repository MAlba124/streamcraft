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
    debug_log, element_def, element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sink, Srcs}, pipeline::{Data, Datagram, Parent}
};

use crossbeam_channel::Receiver;

pub struct StdoutLog {
    parent: Parent,
}

impl StdoutLog {
    pub fn new() -> Self {
        Self {
            parent: Parent::default(),
        }
    }

    fn run_loop(&self, data_receiver: &Receiver<Datagram>) -> bool {
        match data_receiver.recv() {
            Ok(datagram) => match datagram {
                Datagram::Message(msg) => match msg {
                    crate::pipeline::Message::Quit => return false,
                    _ => unreachable!(), // TODO: Handle invalid messages better
                },
                Datagram::Data(data) => {
                    if let Data::Text(s) = data {
                        print!("{s}");
                    }
                }
            },
            Err(e) => {
                debug_log!("Failed to receive data from src: {e}");
                return false;
            }
        }

        true
    }
}

impl Element for StdoutLog {
    fn get_type(&self) -> ElementType {
        ElementType::TextSink
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::One(CommonFormat::Text),
            srcs: Srcs::None,
        }
    }

    fn run(
        &mut self,
        parent_datagram_receiver: Receiver<Datagram>,
    ) -> Result<(), crate::pipeline::error::Error> {
        while self.run_loop(&parent_datagram_receiver) {}
        Ok(())
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }

    fn cleanup(&mut self) -> Result<(), crate::pipeline::error::Error> {
        Ok(())
    }
}

element_def! {
    StdoutLog
}
