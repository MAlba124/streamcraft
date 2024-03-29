#[derive(PartialEq)]
pub enum CommonFormat {
    Text,
}

#[derive(PartialEq)]
pub enum Sinks {
    One(CommonFormat),
    None,
}

impl Sinks {
    pub fn has_none(&self) -> bool {
        matches!(self, Sinks::None)
    }
}

#[derive(PartialEq)]
pub enum Srcs {
    One(CommonFormat),
    None,
}

impl Srcs {
    pub fn has_none(&self) -> bool {
        matches!(self, Srcs::None)
    }
}

pub fn sink_is_compatible_with_src(sink: Sinks, src: Srcs) -> bool {
    match sink {
        Sinks::One(format) => {
            src == Srcs::One(format);
        }
        _ => false,
    }
}

pub struct ElementArchitecture {
    pub sinks: Sinks,
    pub srcs: Srcs,
}

#[derive(PartialEq)]
pub enum ElementType {
    TextSink,
    TextSrc,
}

pub enum Data {
    Text(String),
    None,
}

pub trait Element {
    fn get_type(&self) -> ElementType;
    fn get_architecture(&self) -> ElementArchitecture;
    fn run(&self, input: Data) -> Data;
}
