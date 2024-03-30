use streamcraft::{elements::text::{StdoutLog, TextTestSrc}, pipeline::Pipeline};

fn main() {
    let stdoutlog = StdoutLog::new();

    let mut texttest = TextTestSrc::new();
    texttest.link_sink_element(stdoutlog);

    let mut pipeline = Pipeline::new(texttest);

    pipeline.run().expect("Error occured running pipeline");
}
