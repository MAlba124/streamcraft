//! Download a file over HTTP: `HttpSrc ! FileSink`.
//!
//! Usage: `cargo run -p sc-http --example download -- <http-url> [out-path]`
//! (plain `http://` only; https needs the future TLS transport.)

use sc_http::HttpSrc;
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::FileSink;

fn main() {
    let mut args = std::env::args().skip(1);
    let url = args.next().unwrap_or_else(|| "http://example.com/".to_string());
    let out = args.next().unwrap_or_else(|| "/tmp/sc_download.out".to_string());

    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(&url));
    let sink = p.add(FileSink::new(&out));
    p.link((src, "src"), (sink, "sink")).expect("link");

    match p.run() {
        Ok(()) => {
            let n = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
            let src_c = p.counters(src);
            println!("OK: {url} -> {out} ({n} bytes, {} buffers)", src_c.buffers_out);
        }
        Err(e) => {
            eprintln!("FAILED: {url}: {e:?}");
            std::process::exit(1);
        }
    }
}
