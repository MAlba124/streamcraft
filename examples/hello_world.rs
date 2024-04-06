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
    texttest.link_sink_element(stdoutlog);

    let mut pipeline = Pipeline::new(texttest);

    pipeline.run().expect("Error occurred running pipeline");
}
