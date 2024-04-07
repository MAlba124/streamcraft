// Copyright (C) 2024  MAlba124 <marlhan@proton.me>
//
// Use StreamCraft to print the contents of a text file

use streamcraft::{
    elements::conversion::bytes2text::Bytes2Text, elements::io::filesrc::FileSrc,
    elements::text::stdoutlog::StdoutLog, pipeline::Pipeline,
};

fn main() {
    let stdoutlog = StdoutLog::new();
    let mut bytes2text = Bytes2Text::new();
    bytes2text.link_sink_element(stdoutlog).unwrap();

    let f = std::fs::File::open("README.md").unwrap();
    let mut filesrc = FileSrc::new(f);
    filesrc.link_sink_element(bytes2text).unwrap();

    let mut pipeline = Pipeline::new(filesrc);
    pipeline.init().unwrap();

    while pipeline.iter().is_ok() {}
}
