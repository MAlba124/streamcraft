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
    pipeline::{Data, Parent, SinkPipe},
};

use crossbeam_channel::{bounded, unbounded, Receiver};

pub struct TextTestSrc {
    sink: SinkPipe,
    parent: Parent,
}

impl TextTestSrc {
    pub fn new() -> Self {
        Self {
            sink: SinkPipe::default(),
            parent: Parent::default(),
        }
    }

    // TODO: Check if sink element is valid
    pub fn link_sink_element(&mut self, sink: impl Element + 'static) {
        self.sink.set_element(sink);
    }

    fn run_loop(&self) -> bool {
        while let Some(res) = self.parent.recv_msg() {
            match res {
                Ok(_msg) => {
                    debug_log!("TEXTTESTSRC: Got message from parent");
                }
                Err(e) if e.is_empty() => break,
                Err(e) => {
                    debug_log!("TEXTTESTSRC: {e}");
                    return false;
                }
            }
        }
        if let Some(receiver) = &self.sink.msg_receiver {
            loop {
                match receiver.try_recv() {
                    Ok(_) => debug_log!("TEXTTESTSRC: Received emssage from sink"),
                    Err(e) if e.is_empty() => break,
                    Err(e) => {
                        debug_log!("TEXTTESTSRC: Failed to receive msg from sink: {e}");
                        return false;
                    }
                }
            }
        }

        if let Some(sender) = &self.sink.data_sender {
            if let Err(e) = sender.send(Data::Text(String::from("Test\n"))) {
                debug_log!("TEXTTESTSRC: {e}");
                return false;
            }
        }

        true
    }
}

impl Element for TextTestSrc {
    fn get_type(&self) -> ElementType {
        ElementType::TextSrc
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sinks: Sinks::None,
            srcs: Srcs::One(CommonFormat::Text),
        }
    }

    fn run(
        &mut self,
        _data_receiver: Option<Receiver<Data>>,
    ) -> Result<(), crate::pipeline::error::Error> {
        let (data_sender, data_receiver) = bounded(0);
        let (my_msg_sender, msg_receiver) = unbounded();
        let (msg_sender, my_msg_receiver) = unbounded();
        let parent = Parent::new(msg_sender, msg_receiver.clone());
        let mut sink_element = self.sink.element.take().unwrap(); // TODO: handle `None`
        sink_element.set_parent(parent);
        let data_receiver_clone = data_receiver.clone();

        self.sink.thread_handle = Some(std::thread::spawn(move || {
            match sink_element.run(Some(data_receiver_clone)) {
                Ok(_) => {}
                Err(e) => debug_log!("Error occurred running sink element: {e}"),
            }
        }));
        self.sink.msg_sender = Some(my_msg_sender);
        self.sink.msg_receiver = Some(my_msg_receiver);
        self.sink.data_sender = Some(data_sender);

        let mut i = 0;
        while self.run_loop() {
            i += 1;
        }

        debug_log!("TEXTTESTSRC: Finished iterations={i}");
        // TODO: Join sink thread

        Ok(())
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }
}
