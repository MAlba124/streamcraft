use crate::element_traits::{CommonFormat, Element, ElementArchitecture, ElementType, Sinks, Srcs};

pub struct StdoutLog {}

impl StdoutLog {
    pub fn new() -> Self {
        Self {}
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
}

pub struct TextTestSrc {}

impl TextTestSrc {
    pub fn new() -> Self {
        Self {}
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
}
