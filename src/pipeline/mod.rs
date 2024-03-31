use std::thread::JoinHandle;

use crate::element_traits::Element;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};

pub mod error;

use error::Error;

pub enum Data {
    Text(String),
    None,
}

pub enum Message {}

#[derive(Default)]
pub struct Parent {
    #[allow(dead_code)]
    msg_sender: Option<Sender<Message>>,
    msg_receiver: Option<Receiver<Message>>,
}

impl Parent {
    pub fn new(msg_sender: Sender<Message>, msg_receiver: Receiver<Message>) -> Self {
        Self {
            msg_sender: Some(msg_sender),
            msg_receiver: Some(msg_receiver),
        }
    }

    pub fn recv_msg(&self) -> Option<Result<Message, TryRecvError>> {
        if let Some(receiver) = &self.msg_receiver {
            return Some(receiver.try_recv());
        }
        None
    }
}

#[derive(Default)]
pub struct SinkPipe {
    pub element: Option<Box<dyn Element>>,
    pub thread_handle: Option<JoinHandle<()>>,
    pub data_sender: Option<Sender<Data>>,
    pub msg_sender: Option<Sender<Message>>,
    pub msg_receiver: Option<Receiver<Message>>,
}

impl SinkPipe {
    pub fn set_element(&mut self, element: impl Element + 'static) {
        self.element = Some(Box::new(element));
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

    // TODO: Move the thread spawning to a different function (perhaps `init()`) and later send a
    // message to the head giving a "start" signal?
    pub fn run(&mut self) -> Result<(), Error> {
        let (_data_sender, data_receiver) = bounded(0);
        let (_my_msg_sender, msg_receiver) = unbounded();
        let (msg_sender, _my_msg_receiver) = unbounded();
        let parent = Parent::new(msg_sender, msg_receiver.clone());
        let mut sink_element = self.head.element.take().unwrap(); // TODO: handle `None`
        sink_element.set_parent(parent);
        let data_receiver_clone = data_receiver.clone();

        self.head.thread_handle = Some(std::thread::spawn(move || {
            match sink_element.run(Some(data_receiver_clone)) {
                Ok(_) => {}
                Err(e) => println!("Error occured running sink element: {e}"),
            }
        }));

        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(())
    }
}
