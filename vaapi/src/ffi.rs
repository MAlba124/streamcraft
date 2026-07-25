//! Hand-rolled FFI to libva (`libva.so`) + libva-drm (`libva-drm.so`), transcribed
//! field-by-field from the devshell's libva 2.23 headers
//! (`va/va.h`, `va/va_drm.h`). This is the **only** module with `extern "C"`
//! declarations and the `#[repr(C)]` structs that back them; every `unsafe` call
//! either lives here or in [`crate::va`] (the RAII wrappers directly above it).
//!
//! Layout is locked by the `size_of`/`align_of` assertions at the bottom: if a
//! future libva bump reshuffles a struct, the crate fails to compile rather than
//! feeding the driver a mis-aligned buffer. All sizes are the documented x86-64
//! (LP64) sizes computed from the header field order.
//!
//! Bitfield unions (`seq_fields`, `pic_fields`) are modelled as their `uint32_t
//! value` union arm — the header declares exactly that arm — and the individual
//! flags are packed by hand in [`crate::h264dec`]. This avoids depending on Rust
//! bitfield layout (unspecified) while matching the C ABI byte-for-byte.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::{c_char, c_int, c_void};

// --- Opaque handles / typedefs (va.h) -------------------------------------------------
/// `typedef void* VADisplay;` (va.h:260) — opaque driver-connection handle.
pub type VADisplay = *mut c_void;
/// `typedef int VAStatus;` (va.h:262). `VA_STATUS_SUCCESS == 0` (va.h:264).
pub type VAStatus = c_int;
/// `typedef unsigned int VAGenericID;` and its aliases (Config/Context/Surface/Buffer/Image).
pub type VAGenericID = u32;
pub type VAConfigID = VAGenericID;
pub type VAContextID = VAGenericID;
pub type VASurfaceID = VAGenericID;
pub type VABufferID = VAGenericID;
pub type VAImageID = VAGenericID;
/// The VA-API enums (`VAProfile`, `VAEntrypoint`, …) are C `enum`s == `int`.
pub type VAProfile = c_int;
pub type VAEntrypoint = c_int;
pub type VAConfigAttribType = c_int;
pub type VABufferType = c_int;
pub type VAGenericValueType = c_int;
pub type VASurfaceAttribType = c_int;

pub const VA_STATUS_SUCCESS: VAStatus = 0;
/// `#define VA_INVALID_ID 0xffffffff` (va.h:1647); `VA_INVALID_SURFACE` aliases it.
pub const VA_INVALID_ID: VAGenericID = 0xffff_ffff;
pub const VA_INVALID_SURFACE: VASurfaceID = VA_INVALID_ID;

// --- VAProfile values (va.h:505–540) --------------------------------------------------
pub const VAProfileNone: VAProfile = -1;
pub const VAProfileH264Main: VAProfile = 6;
pub const VAProfileH264High: VAProfile = 7;
pub const VAProfileH264ConstrainedBaseline: VAProfile = 13;
pub const VAProfileHEVCMain: VAProfile = 17;
pub const VAProfileHEVCMain10: VAProfile = 18;
pub const VAProfileVP9Profile0: VAProfile = 19;
pub const VAProfileVP9Profile2: VAProfile = 21;
pub const VAProfileAV1Profile0: VAProfile = 32;

// --- VAEntrypoint (va.h:553) ----------------------------------------------------------
pub const VAEntrypointVLD: VAEntrypoint = 1;
/// Full-featured slice-level encode (va.h:558) — the PAK+ENC engine.
pub const VAEntrypointEncSlice: VAEntrypoint = 6;
/// Low-power fixed-function encode (va.h:568) — Intel VDEnc.
pub const VAEntrypointEncSliceLP: VAEntrypoint = 8;

// --- VAConfigAttribType + RTFormat values (va.h:620, 1083) ----------------------------
pub const VAConfigAttribRTFormat: VAConfigAttribType = 0;
pub const VA_RT_FORMAT_YUV420: u32 = 0x0000_0001;

// --- VABufferType (va.h:2045–2050) ----------------------------------------------------
pub const VAPictureParameterBufferType: VABufferType = 0;
pub const VAIQMatrixBufferType: VABufferType = 1;
pub const VASliceParameterBufferType: VABufferType = 4;
pub const VASliceDataBufferType: VABufferType = 5;

// --- VAGenericValueType (va.h:1652) ---------------------------------------------------
pub const VAGenericValueTypeInteger: VAGenericValueType = 1;

// --- VASurfaceAttribType + flags (va.h:1689–1746) -------------------------------------
// Enum begins at VASurfaceAttribNone = 0, so PixelFormat = 1, MemoryType = 6.
pub const VASurfaceAttribPixelFormat: VASurfaceAttribType = 1;
pub const VASurfaceAttribMemoryType: VASurfaceAttribType = 6;
pub const VA_SURFACE_ATTRIB_SETTABLE: u32 = 0x0000_0002;
pub const VA_SURFACE_ATTRIB_MEM_TYPE_VA: u32 = 0x0000_0001;

// --- FourCC / picture flags / slice-data flags ----------------------------------------
/// `#define VA_FOURCC_NV12 0x3231564E` (va.h:4420).
pub const VA_FOURCC_NV12: u32 = 0x3231_564E;
/// `#define VA_PROGRESSIVE 0x1` (va.h:1920) — the flag passed to vaCreateContext.
pub const VA_PROGRESSIVE: c_int = 0x1;
/// `#define VA_SLICE_DATA_FLAG_ALL 0x00` (va.h:3080) — whole slice in the buffer.
pub const VA_SLICE_DATA_FLAG_ALL: u32 = 0x00;

// VAPictureH264 flags (va.h:3583–3587).
pub const VA_PICTURE_H264_INVALID: u32 = 0x0000_0001;
pub const VA_PICTURE_H264_TOP_FIELD: u32 = 0x0000_0002;
pub const VA_PICTURE_H264_BOTTOM_FIELD: u32 = 0x0000_0004;
pub const VA_PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x0000_0008;
pub const VA_PICTURE_H264_LONG_TERM_REFERENCE: u32 = 0x0000_0010;

/// `#define VA_PADDING_LOW 4` (va.h:360).
pub const VA_PADDING_LOW: usize = 4;
/// `#define VA_PADDING_MEDIUM 8` (va.h:361).
pub const VA_PADDING_MEDIUM: usize = 8;

// --- Structs (#[repr(C)], exact field order from va.h) --------------------------------

/// `VAConfigAttrib` (va.h) — `{ VAConfigAttribType type; uint32_t value; }`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAConfigAttrib {
    pub type_: VAConfigAttribType,
    pub value: u32,
}

/// The value arm of `union { int32_t i; float f; void* p; VAGenericFunc fn; }`
/// (va.h:1666). Pointer arm forces 8-byte size + alignment on LP64.
#[repr(C)]
#[derive(Clone, Copy)]
pub union VAGenericValueUnion {
    pub i: i32,
    pub f: f32,
    pub p: *mut c_void,
}

/// `VAGenericValue` (va.h:1662) — `{ VAGenericValueType type; union value; }`.
/// The 8-byte union's alignment inserts a 4-byte pad after `type_`, so size == 16.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAGenericValue {
    pub type_: VAGenericValueType,
    pub value: VAGenericValueUnion,
}

/// `VASurfaceAttrib` (va.h:1742) — `{ VASurfaceAttribType type; uint32_t flags;
/// VAGenericValue value; }`. type+flags = 8, then the 8-aligned value → 24 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VASurfaceAttrib {
    pub type_: VASurfaceAttribType,
    pub flags: u32,
    pub value: VAGenericValue,
}

/// `VAImageFormat` (va.h:4712) — 8×`uint32_t` fields + `va_reserved[4]` = 48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAImageFormat {
    pub fourcc: u32,
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAImage` (va.h:4729) — the load-bearing readback descriptor. Computed size 120:
/// 4 (image_id) + 48 (format) + 4 (buf) + 2+2 (w/h) + 4 (data_size) + 4 (num_planes)
/// + 12 (pitches) + 12 (offsets) + 4+4 (palette ints) + 4 (component_order) + 16 (rsv).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAImage {
    pub image_id: VAImageID,
    pub format: VAImageFormat,
    pub buf: VABufferID,
    pub width: u16,
    pub height: u16,
    pub data_size: u32,
    pub num_planes: u32,
    pub pitches: [u32; 3],
    pub offsets: [u32; 3],
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAPictureH264` (va.h:3572) — `{ VASurfaceID picture_id; uint32_t frame_idx;
/// uint32_t flags; int32_t TopFieldOrderCnt; int32_t BottomFieldOrderCnt;
/// uint32_t va_reserved[4]; }` = 5×4 + 16 = 36 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAPictureH264 {
    pub picture_id: VASurfaceID,
    pub frame_idx: u32,
    pub flags: u32,
    pub TopFieldOrderCnt: i32,
    pub BottomFieldOrderCnt: i32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

impl VAPictureH264 {
    /// An "empty DPB slot": INVALID flag, invalid surface. Every unused
    /// `ReferenceFrames`/`RefPicList` entry must look like this (per §8.2.4.2, the
    /// driver stops at the first INVALID entry).
    pub fn invalid() -> Self {
        VAPictureH264 {
            picture_id: VA_INVALID_SURFACE,
            frame_idx: 0,
            flags: VA_PICTURE_H264_INVALID,
            TopFieldOrderCnt: 0,
            BottomFieldOrderCnt: 0,
            va_reserved: [0; VA_PADDING_LOW],
        }
    }
}

/// `VAPictureParameterBufferH264` (va.h:3594). Bitfield unions modelled as their
/// `uint32_t value` arm (seq_fields / pic_fields). Field order + `va_reserved[8]`
/// match the header; the C size is locked by the assert below.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAPictureParameterBufferH264 {
    pub CurrPic: VAPictureH264,
    pub ReferenceFrames: [VAPictureH264; 16],
    pub picture_width_in_mbs_minus1: u16,
    pub picture_height_in_mbs_minus1: u16,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub num_ref_frames: u8,
    pub seq_fields: u32,
    pub num_slice_groups_minus1: u8,
    pub slice_group_map_type: u8,
    pub slice_group_change_rate_minus1: u16,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    pub pic_fields: u32,
    pub frame_num: u16,
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

/// `VAIQMatrixBufferH264` (va.h:3648) — `ScalingList4x4[6][16]` (96) +
/// `ScalingList8x8[2][64]` (128) + `va_reserved[4]` (16) = 240 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAIQMatrixBufferH264 {
    pub ScalingList4x4: [[u8; 16]; 6],
    pub ScalingList8x8: [[u8; 64]; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VASliceParameterBufferH264` (va.h:3659). Weight tables are `[i16;32]` /
/// `[[i16;2];32]` exactly as the header declares. Size locked by the assert.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VASliceParameterBufferH264 {
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    pub slice_data_flag: u32,
    pub slice_data_bit_offset: u16,
    pub first_mb_in_slice: u16,
    pub slice_type: u8,
    pub direct_spatial_mv_pred_flag: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    pub RefPicList0: [VAPictureH264; 32],
    pub RefPicList1: [VAPictureH264; 32],
    pub luma_log2_weight_denom: u8,
    pub chroma_log2_weight_denom: u8,
    pub luma_weight_l0_flag: u8,
    pub luma_weight_l0: [i16; 32],
    pub luma_offset_l0: [i16; 32],
    pub chroma_weight_l0_flag: u8,
    pub chroma_weight_l0: [[i16; 2]; 32],
    pub chroma_offset_l0: [[i16; 2]; 32],
    pub luma_weight_l1_flag: u8,
    pub luma_weight_l1: [i16; 32],
    pub luma_offset_l1: [i16; 32],
    pub chroma_weight_l1_flag: u8,
    pub chroma_weight_l1: [[i16; 2]; 32],
    pub chroma_offset_l1: [[i16; 2]; 32],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

// --- extern fns (va.h / va_drm.h) -----------------------------------------------------
#[link(name = "va")]
#[link(name = "va-drm")]
extern "C" {
    // va_drm.h
    pub fn vaGetDisplayDRM(fd: c_int) -> VADisplay;

    // va.h — lifecycle + query
    pub fn vaInitialize(dpy: VADisplay, major: *mut c_int, minor: *mut c_int) -> VAStatus;
    pub fn vaTerminate(dpy: VADisplay) -> VAStatus;
    pub fn vaErrorStr(error_status: VAStatus) -> *const c_char;
    pub fn vaQueryVendorString(dpy: VADisplay) -> *const c_char;
    pub fn vaMaxNumProfiles(dpy: VADisplay) -> c_int;
    pub fn vaMaxNumEntrypoints(dpy: VADisplay) -> c_int;
    pub fn vaQueryConfigProfiles(
        dpy: VADisplay,
        profile_list: *mut VAProfile,
        num_profiles: *mut c_int,
    ) -> VAStatus;
    pub fn vaQueryConfigEntrypoints(
        dpy: VADisplay,
        profile: VAProfile,
        entrypoint_list: *mut VAEntrypoint,
        num_entrypoints: *mut c_int,
    ) -> VAStatus;
    pub fn vaGetConfigAttributes(
        dpy: VADisplay,
        profile: VAProfile,
        entrypoint: VAEntrypoint,
        attrib_list: *mut VAConfigAttrib,
        num_attribs: c_int,
    ) -> VAStatus;

    // va.h — config / surfaces / context
    pub fn vaCreateConfig(
        dpy: VADisplay,
        profile: VAProfile,
        entrypoint: VAEntrypoint,
        attrib_list: *mut VAConfigAttrib,
        num_attribs: c_int,
        config_id: *mut VAConfigID,
    ) -> VAStatus;
    pub fn vaDestroyConfig(dpy: VADisplay, config_id: VAConfigID) -> VAStatus;
    pub fn vaCreateSurfaces(
        dpy: VADisplay,
        format: u32,
        width: u32,
        height: u32,
        surfaces: *mut VASurfaceID,
        num_surfaces: u32,
        attrib_list: *mut VASurfaceAttrib,
        num_attribs: u32,
    ) -> VAStatus;
    pub fn vaDestroySurfaces(
        dpy: VADisplay,
        surfaces: *mut VASurfaceID,
        num_surfaces: c_int,
    ) -> VAStatus;
    pub fn vaCreateContext(
        dpy: VADisplay,
        config_id: VAConfigID,
        picture_width: c_int,
        picture_height: c_int,
        flag: c_int,
        render_targets: *mut VASurfaceID,
        num_render_targets: c_int,
        context: *mut VAContextID,
    ) -> VAStatus;
    pub fn vaDestroyContext(dpy: VADisplay, context: VAContextID) -> VAStatus;

    // va.h — buffers + the decode command sequence
    pub fn vaCreateBuffer(
        dpy: VADisplay,
        context: VAContextID,
        type_: VABufferType,
        size: u32,
        num_elements: u32,
        data: *mut c_void,
        buf_id: *mut VABufferID,
    ) -> VAStatus;
    pub fn vaDestroyBuffer(dpy: VADisplay, buffer_id: VABufferID) -> VAStatus;
    pub fn vaBeginPicture(
        dpy: VADisplay,
        context: VAContextID,
        render_target: VASurfaceID,
    ) -> VAStatus;
    pub fn vaRenderPicture(
        dpy: VADisplay,
        context: VAContextID,
        buffers: *mut VABufferID,
        num_buffers: c_int,
    ) -> VAStatus;
    pub fn vaEndPicture(dpy: VADisplay, context: VAContextID) -> VAStatus;
    pub fn vaSyncSurface(dpy: VADisplay, render_target: VASurfaceID) -> VAStatus;

    // va.h — image readback
    pub fn vaDeriveImage(dpy: VADisplay, surface: VASurfaceID, image: *mut VAImage) -> VAStatus;
    pub fn vaCreateImage(
        dpy: VADisplay,
        format: *mut VAImageFormat,
        width: c_int,
        height: c_int,
        image: *mut VAImage,
    ) -> VAStatus;
    pub fn vaGetImage(
        dpy: VADisplay,
        surface: VASurfaceID,
        x: c_int,
        y: c_int,
        width: u32,
        height: u32,
        image: VAImageID,
    ) -> VAStatus;
    pub fn vaMapBuffer(dpy: VADisplay, buf_id: VABufferID, pbuf: *mut *mut c_void) -> VAStatus;
    pub fn vaUnmapBuffer(dpy: VADisplay, buf_id: VABufferID) -> VAStatus;
    pub fn vaDestroyImage(dpy: VADisplay, image: VAImageID) -> VAStatus;
}

// --- Layout guards: documented LP64 sizes; a libva reshuffle fails compilation --------
#[cfg(test)]
mod layout {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn struct_sizes_match_libva_2_23() {
        assert_eq!(size_of::<VAConfigAttrib>(), 8, "VAConfigAttrib");
        assert_eq!(size_of::<VAGenericValueUnion>(), 8, "VAGenericValueUnion");
        assert_eq!(align_of::<VAGenericValueUnion>(), 8, "VAGenericValueUnion align");
        assert_eq!(size_of::<VAGenericValue>(), 16, "VAGenericValue");
        assert_eq!(size_of::<VASurfaceAttrib>(), 24, "VASurfaceAttrib");
        assert_eq!(size_of::<VAImageFormat>(), 48, "VAImageFormat");
        assert_eq!(size_of::<VAImage>(), 120, "VAImage");
        assert_eq!(size_of::<VAPictureH264>(), 36, "VAPictureH264");
        // 36 (CurrPic) + 576 (ReferenceFrames[16]) = 612; +4 (w/h u16×2) = 616;
        // +3 (bit depths + num_ref u8) = 619 → pad to 620 for seq_fields u32 → 624;
        // +2 (slice_group u8×2) = 626; +2 (change_rate u16) = 628; +4 (qp i8×4) =
        // 632; +4 (pic_fields u32) = 636; +2 (frame_num u16) = 638 → pad to 640 for
        // va_reserved[8] u32 → 672.
        assert_eq!(
            size_of::<VAPictureParameterBufferH264>(),
            672,
            "VAPictureParameterBufferH264"
        );
        assert_eq!(size_of::<VAIQMatrixBufferH264>(), 240, "VAIQMatrixBufferH264");
        // Computed LP64 layout (repr(C) padding accounted for): 12 (data u32×3) +
        // 4 (bit_offset/first_mb u16×2) + 9 (u8/i8 block) → pad to 28 for RefPicList
        // (VAPictureH264 align 4); 2×32×36 = 2304 → 2332; 3 u8 flags/denoms → 2335;
        // pad to 2336 for the i16 weight arrays; four [i16;32]=64 (2336→2400→2464)
        // + flag(1)+pad(1) + two [[i16;2];32]=128 (2466→2594→2722) for l0; the same
        // for l1 (2722→3110); va_reserved[4]=16 after a 2-byte pad (3110→3112→3128).
        assert_eq!(
            size_of::<VASliceParameterBufferH264>(),
            3128,
            "VASliceParameterBufferH264"
        );
        assert_eq!(align_of::<VASliceParameterBufferH264>(), 4);
    }
}
