//! `vaapi-probe` — print the VA-API decode capability table on this machine.
//!
//! ```text
//! cargo run -p sc-vaapi --example vaapi-probe
//! SC_NO_VAAPI=1 cargo run -p sc-vaapi --example vaapi-probe   # forces "no device"
//! SC_VAAPI_DEVICE=/dev/dri/renderD129 cargo run … --example vaapi-probe
//! ```
//!
//! Exits 0 either way: a machine with no usable device prints a clear message
//! rather than failing, so the example is a safe smoke test in CI.

use streamcraft_core::registry::Registry;

fn main() {
    println!("== sc-vaapi capability probe ==");

    let Some(caps) = sc_vaapi::probe() else {
        println!("no VA-API device (probe returned None)");
        if std::env::var("SC_NO_VAAPI").as_deref() == Ok("1") {
            println!("  (SC_NO_VAAPI=1 is set — probing is disabled)");
        }
        return;
    };

    println!("device : {}", caps.device.display());
    println!("va-api : {}.{}", caps.version.0, caps.version.1);
    println!("vendor : {}", caps.vendor);

    println!("\ndecode profiles (VLD entrypoint):");
    for (name, has_vld) in &caps.profiles {
        println!("  {:<28} VLD={}", name, if *has_vld { "yes" } else { "no" });
    }

    println!("\nstreamcraft decode families:");
    if caps.decode_families.is_empty() {
        println!("  (none)");
    } else {
        for f in &caps.decode_families {
            println!("  {f}");
        }
    }

    // Build a registry and register the gated element(s); list what registered.
    let mut registry = Registry::new();
    sc_vaapi::register(&mut registry);
    println!("\nregistered elements:");
    let names = registry.names();
    if names.is_empty() {
        println!("  (none registered — no supported decode family)");
    } else {
        for n in names {
            println!("  {n}");
        }
    }
}
