use std::collections::LinkedList;

use crate::element_traits::{sink_is_compatible_with_src, Data, Element};

pub struct Pipeline {
    elements: LinkedList<Box<dyn Element>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            elements: LinkedList::new(),
        }
    }

    // TODO: Return Result<>
    pub fn link_element(&mut self, element: impl Element + 'static) {
        let arch = element.get_architecture();
        if self.elements.is_empty() && !arch.sinks.has_none() {
            panic!("First element is not a source!");
        }

        if let Some(back_element) = self.elements.back() {
            let back_arch = back_element.get_architecture();
            if back_arch.srcs.has_none() {
                panic!("Last element in pipeline does not have any sources!");
            }

            if !sink_is_compatible_with_src(arch.sinks, back_arch.srcs) {
                panic!("Element is incompatible with the current pipeline!");
            }
        }

        self.elements.push_back(Box::new(element));
    }

    pub fn run(&self) {
        for _ in 0..3 {
            let mut dat = Data::None;
            for element in self.elements.iter() {
                dat = element.run(dat);
            }
        }
    }
}
