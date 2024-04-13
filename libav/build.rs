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

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=avformat");
    println!("cargo:rustc-link-lib=avcodec");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("av_find_input_format")
        .allowlist_function("avformat_open_input")
        .allowlist_function("avformat_version")
        .allowlist_function("av_read_frame")
        .allowlist_function("avformat_close_input")
        .allowlist_function("avformat_alloc_context")
        .allowlist_function("av_packet_alloc")
        .allowlist_function("av_packet_unref")
        .allowlist_function("av_find_best_stream")
        .allowlist_function("avcodec_find_decoder")
        .allowlist_function("avcodec_alloc_context3")
        .allowlist_function("avcodec_find_decoder")
        .allowlist_function("avcodec_free_context")
        .allowlist_function("avcodec_parameters_to_context")
        .allowlist_function("avcodec_open2")
        .allowlist_type("AVInputFormat")
        .allowlist_type("AVFormatContext")
        .allowlist_type("AVPacket")
        .allowlist_type("AVCodec")
        .allowlist_type("AVMediaType")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
