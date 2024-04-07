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

use streamcraft::{
    elements::text::{stdoutlog::StdoutLog, texttestsrc::TextTestSrc},
    pipeline::Pipeline,
};

fn main() {
    let stdoutlog = StdoutLog::new();

    let mut texttest = TextTestSrc::new();
    texttest
        .link_sink_element(stdoutlog)
        .expect("Failed to add sink element");
    texttest.set_text_to_send("Custom set text to send\n".to_string());

    let mut pipeline = Pipeline::new(texttest);
    pipeline.init().expect("Failed to initialize pipeline");

    for _ in 0..3 {
        pipeline.iter().expect("Failed to iterate pipeline");
    }
}
