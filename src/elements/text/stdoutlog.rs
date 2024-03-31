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
    debug_log,
    element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sinks, Srcs},
    pipeline::{Data, Parent},
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

    fn run_loop(&self, data_receiver: &Receiver<Data>) -> bool {
        while let Some(res) = self.parent.recv_msg() {
            match res {
                Ok(msg) => {
                    match msg {
                        crate::pipeline::Message::Quit => return false,
                    }
                }
                Err(e) if e.is_empty() => break,
                Err(e) => {
                    debug_log!("{e}");
                    return false;
                }
            }
        }

        // TODO: Check also messages
        match data_receiver.recv() {
            Ok(data) => {
                if let Data::Text(s) = data {
                    print!("{s}");
                }
            }
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
            sinks: Sinks::One(CommonFormat::Text),
            srcs: Srcs::None,
        }
    }

    fn run(
        &mut self,
        data_receiver: Option<Receiver<Data>>,
    ) -> Result<(), crate::pipeline::error::Error> {
        if let Some(data_receiver) = data_receiver {
            while self.run_loop(&data_receiver) {}
            debug_log!("Finished");
        }


        Ok(())
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }
}
