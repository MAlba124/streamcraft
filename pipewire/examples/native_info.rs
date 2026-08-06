//! Validate the from-scratch native PipeWire client against a live daemon: connect over the
//! `pipewire-0` socket, complete the handshake (no libpipewire), print the server identity, and
//! enumerate the registry globals. This is the human-facing counterpart to the unit tests in
//! `src/native/` — it exercises the real wire format end-to-end.
//!
//! Run (needs a running PipeWire daemon):
//!   cargo run -p pf-pipewire --example native_info

use pf_pipewire::native::PwClient;

fn main() {
    let mut client = match PwClient::connect("streamcraft-native-info") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not connect to PipeWire: {e}");
            eprintln!("(is a daemon running? check $XDG_RUNTIME_DIR/pipewire-0)");
            std::process::exit(1);
        }
    };

    let info = client.server_info();
    println!("connected to PipeWire daemon (native protocol, no libpipewire)");
    println!("  name      : {}", info.name);
    println!("  version   : {}", info.version);
    println!("  user@host : {}@{}", info.user_name, info.host_name);
    println!("  cookie    : {:#x}", info.cookie);

    match client.registry_globals() {
        Ok(globals) => {
            println!("\nregistry: {} global object(s)", globals.len());
            for g in globals {
                // The short interface name (last path segment) keeps the dump readable.
                let short = g.type_.rsplit(':').next().unwrap_or(&g.type_);
                let label = g
                    .prop("node.description")
                    .or_else(|| g.prop("node.nick"))
                    .or_else(|| g.prop("node.name"))
                    .or_else(|| g.prop("factory.name"))
                    .unwrap_or("");
                println!("  {:>4}  {:<10} v{:<2} {}", g.id, short, g.version, label);
            }
        }
        Err(e) => {
            eprintln!("registry enumeration failed: {e}");
            std::process::exit(1);
        }
    }
}
