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

use crossbeam_channel::Receiver;

use crate::pipeline::{self, Datagram, Parent};

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
        Sinks::One(format) => src == Srcs::One(format),
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

// TODO: Separate to `error.rs`
#[derive(Debug)]
pub enum Error {
    NoSources,
    Incompatible,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::NoSources => "Last element in pipeline does not have any sources",
                Self::Incompatible => "Element is incompatible with the current pipeline",
            }
        )
    }
}

pub trait Element: Sync + Send {
    fn get_type(&self) -> ElementType;
    fn get_architecture(&self) -> ElementArchitecture;
    fn run(&mut self, parent_datagram_receiver: Receiver<Datagram>) -> Result<(), pipeline::error::Error>;
    fn set_parent(&mut self, parent: Parent);
}
