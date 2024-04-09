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
    element_def, element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sink, Srcs}, error, pipeline::{error::Error, Datagram, Message, Parent, SinkPipe}
};

use crossbeam_channel::{bounded, unbounded, Receiver};

///```text
/// +-----------------+
/// |            _____|
/// |  TestSrc  | src |----> ????
/// |            ^^^^^|
/// +-----------------+
///```
pub struct TestSrc {
    sink: SinkPipe,
    parent: Parent,
    sink_element_type: ElementType,
    sink_format: CommonFormat,
    index: usize,
    datagrams: Vec<Datagram>,
}

impl TestSrc {
    pub fn new(
        sink_element_type: ElementType,
        sink_format: CommonFormat,
        datagrams: Vec<Datagram>,
    ) -> Self {
        Self {
            sink: SinkPipe::default(),
            parent: Parent::default(),
            sink_element_type,
            sink_format,
            datagrams: datagrams,
            index: 0,
        }
    }

    /// Link the sink element.
    pub fn link_sink_element(&mut self, sink: impl Element + 'static) -> Result<(), Error> {
        if sink.get_sink_type() != self.sink_element_type {
            return Err(Error::InvalidSinkType);
        }

        if let Sink::One(format) = sink.get_architecture().sink {
            if format == self.sink_format {
                self.sink.set_element(sink);
                return Ok(());
            }
        }

        Err(Error::InvalidSinkType)
    }

    fn run_loop(&mut self, datagram: Datagram) -> bool {
        if let Err(e) = self.sink.send_datagram(datagram) {
            error!("{e}");
            return false;
        }

        true
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
}

impl Element for TestSrc {
    fn get_sink_type(&self) -> ElementType {
        // TODO: change to some custom shit
        ElementType::TextSrc
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::None,
            srcs: Srcs::One(self.sink_format.clone()),
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
                        if self.index >= self.datagrams.len() {
                            break;
                        }
                        if !self.run_loop(self.datagrams[self.index].clone()) {
                            break;
                        }
                        self.index += 1;
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
    TestSrc,
    "testsrc"
}

#[cfg(test)]
mod tests {
    use crate::{
        elements::misc::testsink::TestSink,
        pipeline::{self, Data, Pipeline},
    };

    use super::*;

    #[test]
    fn test_basic() {
        let testsink = TestSink::new(
            ElementType::TextSink,
            CommonFormat::Text,
            |_, _| true,
            |n, data| {
                match n {
                    1 => assert_eq!(data, Data::Text(String::from("Hello"))),
                    2 => assert_eq!(data, Data::Text(String::from("World"))),
                    _ => unreachable!(),
                }
                true
            },
        );
        let datagrams = vec![
            Datagram::Data(Data::Text(String::from("Hello"))),
            Datagram::Data(Data::Text(String::from("Hello"))),
        ];
        let mut testsrc = TestSrc::new(ElementType::TextSink, CommonFormat::Text, datagrams);
        testsrc.link_sink_element(testsink).unwrap();

        let mut pipeline = Pipeline::new(testsrc);
        pipeline.init().unwrap();

        for _ in 0..2 {
            pipeline.iter().unwrap();
        }
    }
}
