//! Hardware VA-API *encode capability* tests. Like `hw_decode.rs` these require a
//! real device and skip cleanly (print + return) without one, so the suite stays
//! green on machines with no hardware.
//!
//! These tests verify the capability layer the encoder elements
//! (`vaapih264enc`/`vaapih265enc`/`vaapivp8enc`) stand on: the probe reports
//! encode entrypoints honestly, and the driver can actually *allocate* an encode
//! pipeline (config + NV12 surfaces + context). Allocation is the step queries
//! cannot vouch for — a read-only DRM fd, for instance, answers every query and
//! then fails exactly here (the decode-path lesson). The full encode round trips
//! live in `hw_encode_roundtrip.rs`.

use pf_vaapi::ffi;
use pf_vaapi::va::{Config, Context, Display, Surfaces};

/// The H.264 encode profiles worth trying, most capable first.
const H264_PROFILES: [ffi::VAProfile; 3] = [
    ffi::VAProfileH264High,
    ffi::VAProfileH264Main,
    ffi::VAProfileH264ConstrainedBaseline,
];

#[test]
fn probe_reports_encode_entrypoints() {
    let Some(caps) = pf_vaapi::probe() else {
        eprintln!("skip probe_reports_encode_entrypoints: no VA-API device");
        return;
    };
    eprintln!(
        "device={} encode_families={:?}",
        caps.device.display(),
        caps.encode_families
    );
    for p in &caps.profiles {
        eprintln!("  {:<28} vld={} enc={} enc_lp={}", p.name, p.vld, p.enc, p.enc_lp);
    }
    // The profile table and the family list must agree: any profile with an
    // encode entrypoint implies its family is in encode_families and vice versa.
    let any_enc = caps.profiles.iter().any(|p| p.enc || p.enc_lp);
    assert_eq!(
        any_enc,
        !caps.encode_families.is_empty(),
        "profile table and encode_families disagree"
    );
}

/// If the probe says H.264 encode exists, the driver must be able to stand up a
/// real encode pipeline: config at an encode entrypoint, NV12 input surfaces, and
/// a context over them. This is the allocation-level proof `vaQueryConfigEntrypoints`
/// alone does not give.
#[test]
fn h264_encode_context_allocates() {
    let Some(caps) = pf_vaapi::probe() else {
        eprintln!("skip h264_encode_context_allocates: no VA-API device");
        return;
    };
    if !caps.supports_encode("h264/annexb") {
        eprintln!("skip h264_encode_context_allocates: no H.264 encode entrypoint");
        return;
    }

    let display = Display::open(&caps.device).expect("probe said the device opens");

    // Try each profile × entrypoint the driver might expose; the probe promised
    // at least one combination works, so exhausting them all is a real failure.
    let mut last_err = None;
    for profile in H264_PROFILES {
        for entrypoint in [ffi::VAEntrypointEncSlice, ffi::VAEntrypointEncSliceLP] {
            let config = match Config::new_encode(&display, profile, entrypoint) {
                Ok(c) => c,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };
            // 1280x720 is MB-aligned; 4 surfaces ≈ current + one reference + slack.
            let surfaces =
                Surfaces::new_nv12(&display, 1280, 720, 4).expect("encode input surfaces");
            let context = Context::new(&display, &config, 1280, 720, &surfaces)
                .expect("encode context over NV12 surfaces");
            eprintln!(
                "encode context OK: profile={profile} entrypoint={entrypoint} ctx={}",
                context.id()
            );
            return;
        }
    }
    panic!(
        "probe advertised h264/annexb encode but no profile/entrypoint combination \
         created a config: {last_err:?}"
    );
}
