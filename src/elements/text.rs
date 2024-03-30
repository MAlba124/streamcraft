use crate::{
    element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sinks, Srcs},
    pipeline::{Data, Parent, SinkPipe},
};

use crossbeam_channel::{bounded, unbounded, Receiver};

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
                    println!("STDOUTLOG: Got message from parent");
                }
                Err(e) if e.is_empty() => break,
                Err(e) => {
                    println!("STDOUTLOG: {e}");
                    return false;
                }
            }
        }

        match data_receiver.recv() {
            Ok(data) => {
                match data {
                    Data::Text(s) => print!("STDOUTLOG: {s}"),
                    _ => {}
                }
            }
            Err(e) => {
                println!("STDOUTLOG: Failed to receive data from src: {e}");
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
        }

        println!("STDOUTLOG: Finished");
        Ok(())
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }
}

/* ************************************************************************************* */

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
                Ok(msg) => {
                    println!("TEXTTESTSRC: Got message from parent");
                }
                Err(e) if e.is_empty() => break,
                Err(e) => {
                    println!("TEXTTESTSRC: {e}");
                    return false;
                }
            }
        }
        if let Some(receiver) = &self.sink.msg_receiver {
            loop {
                match receiver.try_recv() {
                    Ok(_) => println!("TEXTTESTSRC: Received emssage from sink"),
                    Err(e) if e.is_empty() => break,
                    Err(e) => {
                        println!("TEXTTESTSRC: Failed to receive msg from sink: {e}");
                        return false;
                    }
                }
            }
        }

        if let Some(sender) = &self.sink.data_sender {
            match sender.send(Data::Text(String::from("Test\n"))) {
                Err(e) => {
                    println!("TEXTTESTSRC: {e}");
                    return false;
                }
                _ => {}
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
                Err(e) => println!("Error occured running sink element: {e}"),
            }
        }));
        self.sink.msg_sender = Some(my_msg_sender);
        self.sink.msg_receiver = Some(my_msg_receiver);
        self.sink.data_sender = Some(data_sender);

        let mut i = 0;
        while self.run_loop() { i += 1; }

        println!("TEXTTESTSRC: Finished iterations={i}");
        // TODO: Join sink thread

        Ok(())
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }
}
