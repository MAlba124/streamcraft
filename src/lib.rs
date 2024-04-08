// StreamCraft - general purpose data/multimedia pipeline framework
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

//! # Streamcraft
//!
//! # Examples
//!
//! Here is a simple pipeline to print "hello world".
//!```
//!use streamcraft::{
//!    elements::text::{stdoutlog::StdoutLog, texttestsrc::TextTestSrc},
//!    pipeline::Pipeline,
//!};
//!
//!// Create text sink element that only prints text to stdout
//!let stdoutlog = StdoutLog::new();
//!
//!let mut texttest = TextTestSrc::new();
//!// Link the printing element with out text src
//!texttest.link_sink_element(stdoutlog).unwrap();
//!// Set the text we wish to print
//!texttest.set_text_to_send("Hello, World!\n".to_string());
//!
//!let mut pipeline = Pipeline::new(texttest);
//!pipeline.init().unwrap();
//!
//!// Perform one iteration of the pipeline
//!pipeline.iter().unwrap();
//!```


pub mod element_traits;
pub mod elements;
pub mod log;
pub mod pipeline;
