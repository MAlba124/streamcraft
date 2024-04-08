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
    debug, element_def,
    element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sink, Srcs},
    error,
    pipeline::{error::Error, Data, Datagram, Message, Parent, SinkPipe},
};

use crossbeam_channel::{bounded, unbounded, Receiver};

/// Text src that sends a [`Data::Text`] packet to the src.
///
///```text
/// +--------------------+
/// |               _____|
/// |  TextTestSrc | src |----> Text
/// |               ^^^^^|
/// +--------------------+
///```
///
/// # Example
/// ```
///use streamcraft::{
///    elements::text::{stdoutlog::StdoutLog, texttestsrc::TextTestSrc},
///    pipeline::Pipeline,
///};
///
///let stdoutlog = StdoutLog::new();
///let mut texttest = TextTestSrc::new();
///texttest.link_sink_element(stdoutlog).unwrap();
///texttest.set_text_to_send("Texttestsrc example".to_string());
///
///let mut pipeline = Pipeline::new(texttest);
///pipeline.init().unwrap();
///
///for _ in 0..3 {
///    pipeline.iter().unwrap(); // This will print "Texttestsrc example" to stdout
///}
/// ```
pub struct TextTestSrc {
    sink: SinkPipe,
    parent: Parent,
    text_to_send: String,
}

impl Default for TextTestSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl TextTestSrc {
    pub fn new() -> Self {
        Self {
            sink: SinkPipe::default(),
            parent: Parent::default(),
            text_to_send: String::from("Test\n"),
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

    /// Set the text to send to sink
    pub fn set_text_to_send(&mut self, text: String) {
        self.text_to_send = text;
    }

    fn run_loop(&mut self) -> bool {
        if let Some(sender) = &self.sink.datagram_sender {
            if let Err(e) = sender.send(Datagram::Data(Data::Text(self.text_to_send.clone()))) {
                error!("{e}");
                return false;
            }
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
                Err(e) => debug!("Error occurred running sink element: {e}"),
            }
        }));
        self.sink.msg_receiver = Some(my_msg_receiver);
        self.sink.datagram_sender = Some(datagram_sender);

        Ok(())
    }
}

impl Element for TextTestSrc {
    fn get_sink_type(&self) -> ElementType {
        ElementType::TextSrc
    }

    fn get_architecture(&self) -> ElementArchitecture {
        ElementArchitecture {
            sink: Sink::None,
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
                    Message::Iter => {
                        if !self.run_loop() {
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
    TextTestSrc,
    "texttestsrc"
}

#[cfg(test)]
mod tests {
    use crate::{elements::misc::testsink, pipeline::Pipeline};

    use super::*;

    #[test]
    fn test_basic() {
        let test_text_data = "Test";
        let testsink = testsink::TestSink::new(
            ElementType::TextSink,
            CommonFormat::Text,
            |_, _| {
                true
            },
            |_, data| {
                match data {
                    Data::Text(text) => assert_eq!(text, String::from("Test")),
                    _ => {}
                }
                true
            }
        );
        let mut textsrc = TextTestSrc::new();
        textsrc.set_text_to_send(String::from(test_text_data));
        textsrc.link_sink_element(testsink).unwrap();

        let mut pipeline = Pipeline::new(textsrc);
        pipeline.init().unwrap();
        pipeline.iter().unwrap();
    }
}
