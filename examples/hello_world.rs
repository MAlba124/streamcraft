// Copyright (C) 2024  MAlba124 <marlhan@proton.me>
//
// Use StreamCraft to print hello world to stdout.

use streamcraft::{
    elements::text::{stdoutlog::StdoutLog, texttestsrc::TextTestSrc},
    pipeline::Pipeline,
};

fn main() {
    let stdoutlog = StdoutLog::new();
    let mut texttest = TextTestSrc::new();
    texttest.link_sink_element(stdoutlog).unwrap();
    texttest.set_text_to_send("Hello, World!\n".to_string());

    let mut pipeline = Pipeline::new(texttest);
    pipeline.init().unwrap();

    pipeline.iter().unwrap();
}
