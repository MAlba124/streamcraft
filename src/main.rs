use streamcraft::{elements::text::{StdoutLog, TextTestSrc}, pipeline::Pipeline};

fn main() {
    let mut pipeline = Pipeline::new();

    let texttest = TextTestSrc::new();
    let stdoutlog = StdoutLog::new();

    pipeline.link_element(texttest);
    pipeline.link_element(stdoutlog);
}
