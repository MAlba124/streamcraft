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

use std::thread::JoinHandle;

use crate::element_traits::Element;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

pub mod error;

use error::Error;

pub enum Data {
    Text(String),
    None,
}

pub enum Message {
    Start,
    Iter,
    Quit,
    Finished,
}

pub enum Datagram {
    Message(Message),
    Data(Data),
}

#[derive(Default)]
pub struct Parent {
    #[allow(dead_code)]
    msg_sender: Option<Sender<Message>>,
}

impl Parent {
    pub fn new(msg_sender: Sender<Message>) -> Self {
        Self {
            msg_sender: Some(msg_sender),
        }
    }

    // TODO: Return error
    pub fn send_finished(&self) {
        if let Some(msg_sender) = &self.msg_sender {
            msg_sender.send(Message::Finished).unwrap();
        }
    }
}

#[derive(Default)]
pub struct SinkPipe {
    pub element: Option<Box<dyn Element>>,
    pub thread_handle: Option<JoinHandle<()>>,
    pub datagram_sender: Option<Sender<Datagram>>,
    pub msg_receiver: Option<Receiver<Message>>,
}

impl SinkPipe {
    pub fn set_element(&mut self, element: impl Element + 'static) {
        self.element = Some(Box::new(element));
    }

    pub fn send_quit(&self) -> Result<(), Error> {
        match self.datagram_sender.as_ref() {
            Some(msg_sender) => msg_sender
                .send(Datagram::Message(Message::Quit))
                .map_err(|_| Error::MessageSinkFailed),
            None => Err(Error::NoSinkMessageSender),
        }
    }

    pub fn join_thread(&mut self) -> Result<(), Error> {
        match self.thread_handle.take() {
            Some(join_handle) => join_handle.join().map_err(|_| Error::FailedToJoinThread),
            None => Err(Error::NoThreadHandle),
        }
    }

    pub fn drop_data_sender(&mut self) {
        self.datagram_sender.take();
    }
}

pub struct Pipeline {
    head: SinkPipe,
}

impl Pipeline {
    pub fn new(element: impl Element + 'static) -> Self {
        let mut head = SinkPipe::default();
        head.set_element(element);
        Self { head }
    }

    pub fn init(&mut self) -> Result<(), Error> {
        let (datagram_sender, datagram_receiver) = bounded(0);
        let (msg_sender, my_msg_receiver) = unbounded();
        let parent = Parent::new(msg_sender);
        let mut sink_element = match self.head.element.take() {
            Some(elm) => elm,
            None => {
                return Err(Error::NoSinkElement);
            }
        };
        sink_element.set_parent(parent);
        let datagram_receiver_clone = datagram_receiver.clone();

        self.head.thread_handle = Some(std::thread::spawn(move || {
            match sink_element.run(datagram_receiver_clone) {
                Ok(_) => {}
                Err(e) => println!("PIPELINE: Error occurred running sink element: {e}"),
            }
        }));
        self.head.msg_receiver = Some(my_msg_receiver);
        self.head.datagram_sender = Some(datagram_sender);

        Ok(())
    }

    pub fn run(&mut self) -> Result<(), Error> {
        if self.head.thread_handle.is_none() {
            return Err(Error::PipelineNotReady);
        }

        if let Some(datagram_sender) = &self.head.datagram_sender {
            datagram_sender.send(Datagram::Message(Message::Start)).unwrap();
        }

        if let Some(msg_receiver) = &self.head.msg_receiver {
            loop {
                // TODO: Handle error
                match msg_receiver.recv().unwrap() {
                    Message::Finished => break,
                    _ => println!("Got invalid message from sink"),
                }
            }
        }

        Ok(())
    }

    pub fn iter(&self) -> Result<(), Error> {
        if self.head.thread_handle.is_none() {
            return Err(Error::PipelineNotReady);
        }

        if let Some(datagram_sender) = &self.head.datagram_sender {
            datagram_sender.send(Datagram::Message(Message::Iter)).unwrap(); // TODO: Handle error
        } else {
            return Err(Error::NoSinkDatagramSender);
        }

        Ok(())
    }

    pub fn de_init(&mut self) -> Result<(), Error> {
        self.head.send_quit()?;
        self.head.join_thread()
    }
}

