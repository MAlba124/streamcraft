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

use crate::{debug, define_log_info, element_traits::Element, error};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

pub mod error;

use error::Error;

#[derive(PartialEq, Debug, Clone)]
pub enum Data {
    Text(String),
    Bytes(Vec<u8>),
    None,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Message {
    Iter,
    IterFin,
    Quit,
    Finished,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Datagram {
    Message(Message),
    Data(Data),
}

#[derive(Default)]
pub struct Parent {
    msg_sender: Option<Sender<Message>>,
}

impl Parent {
    pub fn new(msg_sender: Sender<Message>) -> Self {
        Self {
            msg_sender: Some(msg_sender),
        }
    }

    fn send_msg(&self, msg: Message) -> Result<(), Error> {
        match self.msg_sender.as_ref() {
            Some(msg_sender) => msg_sender.send(msg).map_err(|_| Error::MessageParentFailed),
            None => Err(Error::NoParentMessageSender),
        }
    }

    pub fn send_finished(&self) -> Result<(), Error> {
        self.send_msg(Message::Finished)
    }

    pub fn send_iter_fin(&self) -> Result<(), Error> {
        self.send_msg(Message::IterFin)
    }
}

#[derive(Default)]
pub struct SinkPipe {
    element: Option<Box<dyn Element>>,
    pub thread_handle: Option<JoinHandle<()>>,
    pub datagram_sender: Option<Sender<Datagram>>,
    pub msg_receiver: Option<Receiver<Message>>,
}

impl SinkPipe {
    pub fn set_element(&mut self, element: impl Element + 'static) {
        self.element = Some(Box::new(element));
    }

    pub fn take_element(&mut self) -> Result<Box<dyn Element>, Error> {
        match self.element.take() {
            Some(elem) => Ok(elem),
            None => Err(Error::NoSinkElement),
        }
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

    pub fn try_recv_msg(&self) -> Result<Option<Message>, Error> {
        match &self.msg_receiver {
            Some(receiver) => match receiver.try_recv() {
                Ok(msg) => Ok(Some(msg)),
                Err(e) if e.is_empty() => Ok(None),
                Err(_e) => Err(Error::ReceiveFromSinkFailed),
            },
            None => Err(Error::NoSinkMessageReceiver),
        }
    }

    pub fn send_datagram(&mut self, datagram: Datagram) -> Result<(), Error> {
        match &self.datagram_sender {
            Some(datagram_sender) => datagram_sender
                .send(datagram)
                .map_err(|_| Error::FailedToSendDatagramToSink)?,
            None => return Err(Error::NoSinkDatagramSender),
        }

        Ok(())
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
        let mut sink_element = self.head.take_element()?;
        sink_element.set_parent(parent);
        let datagram_receiver_clone = datagram_receiver.clone();

        self.head.thread_handle = Some(std::thread::spawn(move || {
            match sink_element.run(datagram_receiver_clone) {
                Ok(_) => {}
                Err(e) => error!("PIPELINE: Error occurred running sink element: {e}"),
            }
        }));
        self.head.msg_receiver = Some(my_msg_receiver);
        self.head.datagram_sender = Some(datagram_sender);

        Ok(())
    }

    pub fn iter(&self) -> Result<(), Error> {
        if self.head.thread_handle.is_none() {
            return Err(Error::PipelineNotReady);
        }

        if let Some(datagram_sender) = &self.head.datagram_sender {
            datagram_sender
                .send(Datagram::Message(Message::Iter))
                .map_err(|_| Error::MessageSinkFailed)?;
        } else {
            return Err(Error::NoSinkDatagramSender);
        }

        if let Some(message_receiver) = &self.head.msg_receiver {
            match message_receiver
                .recv()
                .map_err(|_| Error::ReceiveFromSinkFailed)?
            {
                Message::IterFin => Ok(()),
                Message::Finished => {
                    debug!("Finished");
                    Ok(())
                }
                _ => Err(Error::ReceivedInvalidDatagramFromSink),
            }
        } else {
            Err(Error::NoSinkMessageReceiver)
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if let Err(e) = self.head.send_quit() {
            error!("{}", e);
        }

        if let Err(e) = self.head.join_thread() {
            error!("{}", e);
        }
    }
}

define_log_info! {
    "pipeline"
}
