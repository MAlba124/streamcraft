//! VA-API capability probe. Opens a DRM render node, initializes a display, and
//! reports which decode families the driver can accelerate — the gate that decides
//! whether [`crate::register`] adds `vaapih264dec` and what
//! [`crate::video_decoder_for`] can hand back.
//!
//! The probe is cached in a process-lifetime [`OnceLock`]: opening a display is
//! non-trivial and the answer never changes within a run. Environment overrides:
//! `SC_NO_VAAPI=1` forces "no device" (honored before any hardware is touched);
//! `SC_VAAPI_DEVICE=/dev/dri/renderDNNN` pins the node instead of scanning.

use std::path::PathBuf;
use std::sync::OnceLock;

use crate::ffi;
use crate::va::Display;

/// The decode capabilities of the selected VA-API device.
#[derive(Debug, Clone)]
pub struct VaCaps {
    /// The DRM render node the probe opened.
    pub device: PathBuf,
    /// Driver vendor banner (`vaQueryVendorString`), e.g. the iHD driver string.
    pub vendor: String,
    /// VA-API version reported by `vaInitialize` (major, minor).
    pub version: (i32, i32),
    /// streamcraft decode families the driver advertises a VLD entrypoint for, in a
    /// stable order (e.g. `["h264/annexb", "h265/annexb", "vp9"]`).
    pub decode_families: Vec<&'static str>,
    /// The raw `(profile_name, has_vld)` table, for the diagnostic example.
    pub profiles: Vec<(&'static str, bool)>,
}

impl VaCaps {
    /// Whether a decode family is accelerated here.
    pub fn supports(&self, family: &str) -> bool {
        self.decode_families.iter().any(|f| *f == family)
    }
}

static CAPS: OnceLock<Option<VaCaps>> = OnceLock::new();

/// Probe the VA-API device once and cache the result. `None` means no usable
/// device (disabled via env, none present, or none initialized).
pub fn probe() -> Option<&'static VaCaps> {
    CAPS.get_or_init(probe_uncached).as_ref()
}

/// The candidate DRM render nodes, honoring `SC_VAAPI_DEVICE`, else scanning
/// `/dev/dri/renderD*` in ascending order (128, 129, …).
fn candidate_nodes() -> Vec<PathBuf> {
    if let Ok(dev) = std::env::var("SC_VAAPI_DEVICE") {
        if !dev.is_empty() {
            return vec![PathBuf::from(dev)];
        }
    }
    let mut nodes = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/dev/dri") {
        for entry in rd.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(num) = name.strip_prefix("renderD") {
                if num.chars().all(|c| c.is_ascii_digit()) {
                    nodes.push(entry.path());
                }
            }
        }
    }
    nodes.sort();
    nodes
}

fn probe_uncached() -> Option<VaCaps> {
    // Env kill switch, before any device is opened.
    if std::env::var("SC_NO_VAAPI").map(|v| v == "1").unwrap_or(false) {
        return None;
    }
    for node in candidate_nodes() {
        if let Ok(display) = Display::open(&node) {
            return Some(query_caps(&display, node));
        }
    }
    None
}

/// Map a VA profile to a streamcraft decode family (only the ones this plugin can
/// route). Unknown/uninteresting profiles map to `None`.
fn family_of(profile: ffi::VAProfile) -> Option<&'static str> {
    match profile {
        ffi::VAProfileH264Main | ffi::VAProfileH264High | ffi::VAProfileH264ConstrainedBaseline => {
            Some("h264/annexb")
        }
        ffi::VAProfileHEVCMain | ffi::VAProfileHEVCMain10 => Some("h265/annexb"),
        ffi::VAProfileVP9Profile0 | ffi::VAProfileVP9Profile2 => Some("vp9"),
        ffi::VAProfileAV1Profile0 => Some("av1"),
        _ => None,
    }
}

/// A short human name for a profile, for the diagnostic profile table.
fn profile_name(profile: ffi::VAProfile) -> &'static str {
    match profile {
        ffi::VAProfileH264ConstrainedBaseline => "H264ConstrainedBaseline",
        ffi::VAProfileH264Main => "H264Main",
        ffi::VAProfileH264High => "H264High",
        ffi::VAProfileHEVCMain => "HEVCMain",
        ffi::VAProfileHEVCMain10 => "HEVCMain10",
        ffi::VAProfileVP9Profile0 => "VP9Profile0",
        ffi::VAProfileVP9Profile2 => "VP9Profile2",
        ffi::VAProfileAV1Profile0 => "AV1Profile0",
        _ => "other",
    }
}

fn query_caps(display: &Display, device: PathBuf) -> VaCaps {
    let mut decode_families: Vec<&'static str> = Vec::new();
    let mut profiles: Vec<(&'static str, bool)> = Vec::new();

    // vaQueryConfigProfiles → for each profile, vaQueryConfigEntrypoints → VLD?
    // SAFETY: display is live; buffers are sized by the driver-declared maxima.
    unsafe {
        let max_profiles = ffi::vaMaxNumProfiles(display.raw()).max(0) as usize;
        let mut profile_list = vec![ffi::VAProfileNone; max_profiles];
        let mut num_profiles: std::ffi::c_int = 0;
        if ffi::vaQueryConfigProfiles(display.raw(), profile_list.as_mut_ptr(), &mut num_profiles)
            != ffi::VA_STATUS_SUCCESS
        {
            return VaCaps {
                device,
                vendor: display.vendor(),
                version: display.version(),
                decode_families,
                profiles,
            };
        }
        profile_list.truncate(num_profiles.max(0) as usize);

        let max_entry = ffi::vaMaxNumEntrypoints(display.raw()).max(0) as usize;
        for &profile in &profile_list {
            let Some(family) = family_of(profile) else { continue };
            let mut entry_list = vec![0i32; max_entry];
            let mut num_entries: std::ffi::c_int = 0;
            let has_vld = if ffi::vaQueryConfigEntrypoints(
                display.raw(),
                profile,
                entry_list.as_mut_ptr(),
                &mut num_entries,
            ) == ffi::VA_STATUS_SUCCESS
            {
                entry_list[..num_entries.max(0) as usize]
                    .iter()
                    .any(|&e| e == ffi::VAEntrypointVLD)
            } else {
                false
            };
            profiles.push((profile_name(profile), has_vld));
            if has_vld && !decode_families.contains(&family) {
                decode_families.push(family);
            }
        }
    }

    VaCaps {
        device,
        vendor: display.vendor(),
        version: display.version(),
        decode_families,
        profiles,
    }
}
