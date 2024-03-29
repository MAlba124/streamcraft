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
        match self {
            Sinks::None => true,
            _ => false,
        }
    }
}

#[derive(PartialEq)]
pub enum Srcs {
    One(CommonFormat),
    None,
}

impl Srcs {
    pub fn has_none(&self) -> bool {
        match self {
            Srcs::None => true,
            _ => false,
        }
    }
}

pub fn sink_is_compatible_with_src(sink: Sinks, src: Srcs) -> bool {
    match sink {
        Sinks::One(format) => {
            return src == Srcs::One(format);
        }
        Sinks::None => unreachable!(),
    }

    false
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

pub trait Element {
    fn get_type(&self) -> ElementType;
    fn get_architecture(&self) -> ElementArchitecture;
}
