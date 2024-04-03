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
pub enum Sink {
    One(CommonFormat),
    None,
}

impl Sink {
    pub fn has_none(&self) -> bool {
        matches!(self, Sink::None)
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

pub fn sink_is_compatible_with_src(sink: Sink, src: Srcs) -> bool {
    match sink {
        Sink::One(format) => src == Srcs::One(format),
        _ => false,
    }
}

pub struct ElementArchitecture {
    pub sink: Sink,
    pub srcs: Srcs,
}

#[derive(PartialEq)]
pub enum ElementType {
    TextSink,
    TextSrc,
}

pub trait Element: Sync + Send {
    fn get_type(&self) -> ElementType;
    fn get_architecture(&self) -> ElementArchitecture;
    fn run(&mut self, parent_datagram_receiver: Receiver<Datagram>) -> Result<(), pipeline::error::Error>;
    fn set_parent(&mut self, parent: Parent);
}
