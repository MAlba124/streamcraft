use crate::{
    debug_log,
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
                    debug_log!("Got message prom parent");
                }
                Err(e) if e.is_empty() => break,
                Err(e) => {
                    debug_log!("{e}");
                    return false;
                }
            }
        }

        match data_receiver.recv() {
            Ok(data) => match data {
                Data::Text(s) => print!("{s}"),
                _ => {}
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

        debug_log!("Finished");
        Ok(())
    }

    fn set_parent(&mut self, parent: Parent) {
        self.parent = parent;
    }
}
