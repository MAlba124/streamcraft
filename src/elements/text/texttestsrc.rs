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
    element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sink, Srcs},
    pipeline::{error::Error, Data, Datagram, Parent, SinkPipe},
};

use crossbeam_channel::{bounded, unbounded, Receiver};

pub struct TextTestSrc {
    sink: SinkPipe,
    parent: Parent,
    iter_mode: bool,
}

impl TextTestSrc {
    pub fn new() -> Self {
        Self {
            sink: SinkPipe::default(),
            parent: Parent::default(),
            iter_mode: false,
        }
    }

    // TODO: Check if sink element is valid
    pub fn link_sink_element(&mut self, sink: impl Element + 'static) {
        self.sink.set_element(sink);
    }

    fn run_loop(&mut self, datagram_receiver: &Receiver<Datagram>) -> bool {
        if let Some(receiver) = &self.sink.msg_receiver {
            loop {
                match receiver.try_recv() {
                    Ok(_) => debug_log!("Received emssage from sink"),
                    Err(e) if e.is_empty() => break,
                    Err(e) => {
                        debug_log!("Failed to receive msg from sink: {e}");
                        return false;
                    }
                }
            }
        }

        if let Some(sender) = &self.sink.datagram_sender {
            if let Err(e) = sender.send(Datagram::Data(Data::Text(String::from("Test\n")))) {
                debug_log!("{e}");
                return false;
            }
        }

        loop {
            match datagram_receiver.try_recv() {
                Ok(datagram) => match datagram {
                    Datagram::Message(msg) => match msg {
                        crate::pipeline::Message::Quit => return false,
                        crate::pipeline::Message::Start => self.iter_mode = false,
                        crate::pipeline::Message::Iter => self.iter_mode = true,
                        _ => println!("Received Finished from pipeline... hmm"),
                    },
                    Datagram::Data(_) => {}
                },
                Err(e) if e.is_empty() => break,
                Err(_) => return false,
            }
        }

        true
    }

    fn init(&mut self) -> Result<(), Error> {
        let (datagram_sender, datagram_receiver) = bounded(0);
        let (msg_sender, my_msg_receiver) = unbounded();
        let parent = Parent::new(msg_sender);
        let mut sink_element = match self.sink.element.take() {
            Some(elm) => elm,
            None => return Err(Error::NoSinkElement),
        };
        sink_element.set_parent(parent);
        let datagram_receiver_clone = datagram_receiver.clone();

        self.sink.thread_handle = Some(std::thread::spawn(move || {
            match sink_element.run(datagram_receiver_clone) {
                Ok(_) => {}
                Err(e) => debug_log!("Error occurred running sink element: {e}"),
            }
        }));
        self.sink.msg_receiver = Some(my_msg_receiver);
        self.sink.datagram_sender = Some(datagram_sender);

        Ok(())
    }

    fn quit(&mut self) -> Result<(), Error> {
        self.sink.send_quit()?;
        self.sink.drop_data_sender();

        self.sink.join_thread()?;

        Ok(())
    }
}

impl Element for TextTestSrc {
    fn get_type(&self) -> ElementType {
        ElementType::TextSrc
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::None,
            srcs: Srcs::One(CommonFormat::Text),
        }
    }

    // TODO: Perform one iteration when receiving Iter message
    fn run(&mut self, parent_datagram_receiver: Receiver<Datagram>) -> Result<(), Error> {
        self.init()?;

        // TODO: Make function to wait for a speciffic message
        while let Ok(datagram) = parent_datagram_receiver.recv() {
            match datagram {
                Datagram::Message(msg) => match msg {
                    crate::pipeline::Message::Start => {
                        self.iter_mode = false;
                        break;
                    }
                    crate::pipeline::Message::Iter => {
                        self.iter_mode = true;
                        break;
                    }
                    crate::pipeline::Message::Quit => return self.quit(),
                    _ => {}
                },
                Datagram::Data(_) => todo!(),
            }
        }

        while self.run_loop(&parent_datagram_receiver) {
            if self.iter_mode {
                while let Ok(datagram) = parent_datagram_receiver.recv() {
                    match datagram {
                        Datagram::Message(msg) => match msg {
                            crate::pipeline::Message::Start => {
                                self.iter_mode = false;
                                break;
                            }
                            crate::pipeline::Message::Iter => break,
                            crate::pipeline::Message::Quit => return self.quit(),
                            _ => {}
                        },
                        Datagram::Data(_) => todo!(),
                    }
                }
            }
        }

        self.parent.send_finished();

        self.quit()
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }
}
