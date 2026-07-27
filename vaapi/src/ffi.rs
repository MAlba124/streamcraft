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
/// `VAProfileVP8Version0_3 = 14` (va.h:519) — the single VP8 profile (versions 0–3).
pub const VAProfileVP8Version0_3: VAProfile = 14;
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
/// `VAConfigAttribRateControl = 5` (va.h:625) — encode rate-control modes bitmask.
pub const VAConfigAttribRateControl: VAConfigAttribType = 5;
/// `VAConfigAttribEncPackedHeaders = 10` (va.h:694) — which packed headers the app
/// may/must submit alongside encode parameter buffers.
pub const VAConfigAttribEncPackedHeaders: VAConfigAttribType = 10;
/// `VAConfigAttribEncMaxRefFrames = 13` (va.h:714) — max L0/L1 references
/// (L0 in the low 16 bits, L1 in the high 16).
pub const VAConfigAttribEncMaxRefFrames: VAConfigAttribType = 13;
pub const VA_RT_FORMAT_YUV420: u32 = 0x0000_0001;
/// `#define VA_ATTRIB_NOT_SUPPORTED 0x80000000` (va.h) — vaGetConfigAttributes'
/// "driver does not support this attribute" sentinel.
pub const VA_ATTRIB_NOT_SUPPORTED: u32 = 0x8000_0000;

// Attribute values for VAConfigAttribRateControl (va.h:1105–1118).
pub const VA_RC_CBR: u32 = 0x0000_0002;
pub const VA_RC_VBR: u32 = 0x0000_0004;
pub const VA_RC_CQP: u32 = 0x0000_0010;

// Attribute values for VAConfigAttribEncPackedHeaders (va.h:1196–1230).
pub const VA_ENC_PACKED_HEADER_NONE: u32 = 0x0000_0000;
pub const VA_ENC_PACKED_HEADER_SEQUENCE: u32 = 0x0000_0001;
pub const VA_ENC_PACKED_HEADER_PICTURE: u32 = 0x0000_0002;
pub const VA_ENC_PACKED_HEADER_SLICE: u32 = 0x0000_0004;
pub const VA_ENC_PACKED_HEADER_MISC: u32 = 0x0000_0008;
pub const VA_ENC_PACKED_HEADER_RAW_DATA: u32 = 0x0000_0010;

// --- VABufferType (va.h:2045–2067) ----------------------------------------------------
pub const VAPictureParameterBufferType: VABufferType = 0;
pub const VAIQMatrixBufferType: VABufferType = 1;
pub const VASliceParameterBufferType: VABufferType = 4;
pub const VASliceDataBufferType: VABufferType = 5;
/// `VAQMatrixBufferType = 11` (va.h:2056) — codec-specific quantization matrices
/// (the VP8 encoder's `VAQMatrixBufferVP8` rides this type).
pub const VAQMatrixBufferType: VABufferType = 11;
// Encode buffer types (va.h:2061–2067).
pub const VAEncCodedBufferType: VABufferType = 21;
pub const VAEncSequenceParameterBufferType: VABufferType = 22;
pub const VAEncPictureParameterBufferType: VABufferType = 23;
pub const VAEncSliceParameterBufferType: VABufferType = 24;
pub const VAEncPackedHeaderParameterBufferType: VABufferType = 25;
pub const VAEncPackedHeaderDataBufferType: VABufferType = 26;
pub const VAEncMiscParameterBufferType: VABufferType = 27;

// --- VAEncPackedHeaderType (va.h:2421–2434) -------------------------------------------
pub const VAEncPackedHeaderSequence: u32 = 1;
pub const VAEncPackedHeaderPicture: u32 = 2;
pub const VAEncPackedHeaderSlice: u32 = 3;
pub const VAEncPackedHeaderRawData: u32 = 4;

// --- VAEncMiscParameterType (va.h:2380–2387) ------------------------------------------
pub const VAEncMiscParameterTypeFrameRate: u32 = 0;
pub const VAEncMiscParameterTypeRateControl: u32 = 1;
pub const VAEncMiscParameterTypeHRD: u32 = 5;

// --- VAGenericValueType (va.h:1652) ---------------------------------------------------
pub const VAGenericValueTypeInteger: VAGenericValueType = 1;

// --- VASurfaceAttribType + flags (va.h:1689–1746) -------------------------------------
// Enum begins at VASurfaceAttribNone = 0, so PixelFormat = 1, MemoryType = 6.
pub const VASurfaceAttribPixelFormat: VASurfaceAttribType = 1;
pub const VASurfaceAttribMemoryType: VASurfaceAttribType = 6;
pub const VA_SURFACE_ATTRIB_SETTABLE: u32 = 0x0000_0002;
pub const VA_SURFACE_ATTRIB_MEM_TYPE_VA: u32 = 0x0000_0001;

// --- DMA-BUF surface export (va_drmcommon.h) ------------------------------------------
// The zero-copy display path (`h265dec::VaapiH265Dec::new_zerocopy` / the player's EGL
// backend): instead of reading the decoded NV12 surface back to the CPU, export it as a
// set of DMA-BUF fds the GUI's GLES context imports with `EGL_LINUX_DMA_BUF_EXT`. The
// descriptor + `vaExportSurfaceHandle` are transcribed from libva's `va/va_drmcommon.h`
// (the `VADRMPRIMESurfaceDescriptor` / `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2` family,
// libva ≥ 1.1 / the "PRIME 2" export added in 2.1). Layouts locked by the asserts at the
// bottom, exactly like the decode/encode structs.

/// `#define VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2 0x40000000` (va_drmcommon.h) — the
/// memory type `vaExportSurfaceHandle` fills a [`VADRMPRIMESurfaceDescriptor`] for. (The
/// older `..._DRM_PRIME` = 0x20000000 fills the legacy `VADRMPRIMESurfaceDescriptor`
/// without per-layer DRM format modifiers; we use PRIME_2, the modern form iHD/radeonsi
/// both implement.)
pub const VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2: u32 = 0x4000_0000;

/// `#define VA_EXPORT_SURFACE_READ_ONLY 0x0001` (va.h) — the exported fds are only read
/// (the GUI samples them; it never writes the surface).
pub const VA_EXPORT_SURFACE_READ_ONLY: u32 = 0x0001;
/// `#define VA_EXPORT_SURFACE_WRITE_ONLY 0x0002` (va.h).
pub const VA_EXPORT_SURFACE_WRITE_ONLY: u32 = 0x0002;
/// `#define VA_EXPORT_SURFACE_SEPARATE_LAYERS 0x0004` (va.h) — one layer per plane (Y in
/// layer 0, CbCr in layer 1 for NV12), each with its own `drm_format`.
pub const VA_EXPORT_SURFACE_SEPARATE_LAYERS: u32 = 0x0004;
/// `#define VA_EXPORT_SURFACE_COMPOSED_LAYERS 0x0008` (va.h) — a single layer whose
/// `drm_format` is the whole surface's FourCC (`DRM_FORMAT_NV12`) and whose `num_planes`
/// covers every plane. This is what an `EGL_LINUX_DMA_BUF_EXT` NV12 import wants (one
/// image with per-plane fd/offset/pitch), so it is the flag the export path passes.
pub const VA_EXPORT_SURFACE_COMPOSED_LAYERS: u32 = 0x0008;

/// The max objects/layers/planes `VADRMPRIMESurfaceDescriptor` statically sizes its arrays
/// to (va_drmcommon.h `#define VA_DRM_PRIME_SURFACE_MAX_...`). 4 comfortably covers NV12
/// (1–2 objects, 1 composed layer, 2 planes) and any 4:2:0/4:4:4 planar layout.
pub const VA_DRM_PRIME_MAX_OBJECTS: usize = 4;
pub const VA_DRM_PRIME_MAX_LAYERS: usize = 4;
pub const VA_DRM_PRIME_MAX_PLANES: usize = 4;

/// One exported DMA-BUF object: `{ int fd; uint32_t size; uint64_t drm_format_modifier; }`
/// (va_drmcommon.h). The `u64` modifier forces 8-byte alignment → a 4-byte pad after
/// `size`, so the struct is 24 bytes on LP64 (locked below). `fd` is owned by the caller
/// after export and must be `close()`d (or handed to an importer that `dup`s / closes it).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VADRMPRIMESurfaceDescriptorObject {
    pub fd: c_int,
    pub size: u32,
    pub drm_format_modifier: u64,
}

/// One exported layer: `{ uint32_t drm_format; uint32_t num_planes; uint32_t
/// object_index[4]; uint32_t offset[4]; uint32_t pitch[4]; }` (va_drmcommon.h). For a
/// COMPOSED NV12 export there is one layer whose `drm_format` is `DRM_FORMAT_NV12`, with
/// `num_planes == 2` (Y then CbCr) indexing into the descriptor's `objects[]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VADRMPRIMESurfaceDescriptorLayer {
    pub drm_format: u32,
    pub num_planes: u32,
    pub object_index: [u32; VA_DRM_PRIME_MAX_PLANES],
    pub offset: [u32; VA_DRM_PRIME_MAX_PLANES],
    pub pitch: [u32; VA_DRM_PRIME_MAX_PLANES],
}

/// `VADRMPRIMESurfaceDescriptor` (va_drmcommon.h) — the whole export: the surface FourCC +
/// coded geometry, the DMA-BUF `objects[]` (the GPU-memory handles), and the `layers[]`
/// (which object/offset/pitch each plane lives at). `vaExportSurfaceHandle` fills this when
/// asked for `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VADRMPRIMESurfaceDescriptor {
    /// The surface FourCC (`VA_FOURCC_NV12`), NOT the DRM fourcc (that is per-layer).
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub num_objects: u32,
    pub objects: [VADRMPRIMESurfaceDescriptorObject; VA_DRM_PRIME_MAX_OBJECTS],
    pub num_layers: u32,
    pub layers: [VADRMPRIMESurfaceDescriptorLayer; VA_DRM_PRIME_MAX_LAYERS],
}

// --- FourCC / picture flags / slice-data flags ----------------------------------------
/// `#define VA_FOURCC_NV12 0x3231564E` (va.h:4420).
pub const VA_FOURCC_NV12: u32 = 0x3231_564E;
/// `#define DRM_FORMAT_NV12 fourcc_code('N','V','1','2')` (drm/drm_fourcc.h) — the DRM
/// FourCC an EGL NV12 import declares. Byte-identical to `VA_FOURCC_NV12` (both `'N','V',
/// '1','2'` little-endian), but named separately at its point of use (the DRM/EGL side).
pub const DRM_FORMAT_NV12: u32 = 0x3231_564E;
/// `#define DRM_FORMAT_MOD_INVALID` (drm/drm_fourcc.h) — "the driver did not report a
/// modifier"; a COMPOSED export from a linear/tiled surface may leave the modifier at this
/// sentinel, in which case the EGL import must omit the `EGL_DMA_BUF_PLANE*_MODIFIER_*`
/// attributes (passing INVALID is an error on some drivers).
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
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
/// `#define VA_PADDING_HIGH 16` (va.h:362).
pub const VA_PADDING_HIGH: usize = 16;

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

// --- Encode structs (va.h / va_enc_h264.h / va_enc_hevc.h / va_enc_vp8.h) -------------
//
// Same transcription rules as the decode structs above: field order verbatim from the
// devshell's libva 2.23 headers, bitfield unions modelled as their `uint32_t value`
// arm (the individual flags are packed by hand in the encoder elements), layouts
// locked by the size assertions at the bottom.

/// `VACodedBufferSegment` (va.h:3940) — one link of the coded-output list a mapped
/// `VAEncCodedBufferType` buffer points at. 16 (4×u32) + 16 (2 ptrs) + 16 (reserved).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VACodedBufferSegment {
    pub size: u32,
    pub bit_offset: u32,
    pub status: u32,
    pub reserved: u32,
    pub buf: *mut c_void,
    pub next: *mut c_void,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncPackedHeaderParameterBuffer` (va.h:2447) — describes the packed-header
/// *data* buffer that follows it (type, bit length, whether emulation-prevention
/// bytes are already inserted).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncPackedHeaderParameterBuffer {
    pub type_: u32,
    pub bit_length: u32,
    pub has_emulation_bytes: u8,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncMiscParameterRateControl` (va.h:2500) — the RC misc payload (bitrate,
/// target %, window, QP bounds). Rides inside a `VAEncMiscParameterBuffer` whose
/// leading u32 is `VAEncMiscParameterTypeRateControl`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncMiscParameterRateControl {
    pub bits_per_second: u32,
    pub target_percentage: u32,
    pub window_size: u32,
    pub initial_qp: u32,
    pub min_qp: u32,
    pub basic_unit_size: u32,
    pub rc_flags: u32,
    pub ICQ_quality_factor: u32,
    pub max_qp: u32,
    pub quality_factor: u32,
    pub target_frame_size: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncMiscParameterFrameRate` (va.h:2617) — fps as `den << 16 | num`
/// (denominator zero means 1, i.e. the low half is integer fps).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncMiscParameterFrameRate {
    pub framerate: u32,
    pub framerate_flags: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncMiscParameterHRD` (va.h:2633) — the hypothetical-reference-decoder
/// buffer model: `buffer_size` bits of decoder buffer, starting at
/// `initial_buffer_fullness`. Bounding this bounds worst-case frame burst (and
/// so end-to-end latency) under bitrate control.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncMiscParameterHRD {
    pub initial_buffer_fullness: u32,
    pub buffer_size: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncSequenceParameterBufferH264` (va_enc_h264.h:187). `seq_fields`/`vui_fields`
/// are the u32 union arms; see the header for the bit order (packed by the element).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncSequenceParameterBufferH264 {
    pub seq_parameter_set_id: u8,
    pub level_idc: u8,
    pub intra_period: u32,
    pub intra_idr_period: u32,
    pub ip_period: u32,
    pub bits_per_second: u32,
    pub max_num_ref_frames: u32,
    pub picture_width_in_mbs: u16,
    pub picture_height_in_mbs: u16,
    pub seq_fields: u32,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub num_ref_frames_in_pic_order_cnt_cycle: u8,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offset_for_ref_frame: [i32; 256],
    pub frame_cropping_flag: u8,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    pub vui_parameters_present_flag: u8,
    pub vui_fields: u32,
    pub aspect_ratio_idc: u8,
    pub sar_width: u32,
    pub sar_height: u32,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncPictureParameterBufferH264` (va_enc_h264.h:344).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncPictureParameterBufferH264 {
    pub CurrPic: VAPictureH264,
    pub ReferenceFrames: [VAPictureH264; 16],
    pub coded_buf: VABufferID,
    pub pic_parameter_set_id: u8,
    pub seq_parameter_set_id: u8,
    pub last_picture: u8,
    pub frame_num: u16,
    pub pic_init_qp: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    pub pic_fields: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncSliceParameterBufferH264` (va_enc_h264.h:501) — including the (unused by
/// us, but ABI-load-bearing) explicit weighted-prediction tables.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncSliceParameterBufferH264 {
    pub macroblock_address: u32,
    pub num_macroblocks: u32,
    pub macroblock_info: VABufferID,
    pub slice_type: u8,
    pub pic_parameter_set_id: u8,
    pub idr_pic_id: u16,
    pub pic_order_cnt_lsb: u16,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub direct_spatial_mv_pred_flag: u8,
    pub num_ref_idx_active_override_flag: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
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
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i8,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAPictureHEVC` (va.h:5280) — surface + POC + RPS-membership flags.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAPictureHEVC {
    pub picture_id: VASurfaceID,
    pub pic_order_cnt: i32,
    pub flags: u32,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

// VAPictureHEVC flags (va.h:5305–5331).
pub const VA_PICTURE_HEVC_INVALID: u32 = 0x0000_0001;
pub const VA_PICTURE_HEVC_LONG_TERM_REFERENCE: u32 = 0x0000_0008;
pub const VA_PICTURE_HEVC_RPS_ST_CURR_BEFORE: u32 = 0x0000_0010;
pub const VA_PICTURE_HEVC_RPS_ST_CURR_AFTER: u32 = 0x0000_0020;
pub const VA_PICTURE_HEVC_RPS_LT_CURR: u32 = 0x0000_0040;

impl VAPictureHEVC {
    /// An "empty slot" entry: invalid surface + INVALID flag, the fill value every
    /// unused `reference_frames`/`ref_pic_list` element must carry.
    pub fn invalid() -> Self {
        VAPictureHEVC {
            picture_id: VA_INVALID_SURFACE,
            pic_order_cnt: 0,
            flags: VA_PICTURE_HEVC_INVALID,
            va_reserved: [0; VA_PADDING_LOW],
        }
    }
}

/// `VAPictureParameterBufferHEVC` (va_dec_hevc.h:57) — the once-per-frame HEVC
/// **decode** picture parameters: SPS/PPS-derived geometry + coding-tool flags, the
/// DPB `ReferenceFrames[15]`, and the fields the driver needs to parse slice segment
/// headers itself (the *long slice format*). `pic_fields` and `slice_parsing_fields`
/// are the two `uint32_t value` bitfield-union arms (packed by hand in
/// [`crate::h265dec`], exactly like the H.264 `seq_fields`/`pic_fields`).
///
/// Range-extension / screen-content decode uses the wider `…HEVCExtension` variant
/// that embeds this as `base`; Main 8-bit 4:2:0 (all this crate wires) uses this
/// struct alone with buffer type `VAPictureParameterBufferType`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAPictureParameterBufferHEVC {
    pub CurrPic: VAPictureHEVC,
    pub ReferenceFrames: [VAPictureHEVC; 15],
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
    /// `pic_fields` union arm (va_dec_hevc.h:71–100).
    pub pic_fields: u32,
    pub sps_max_dec_pic_buffering_minus1: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_transform_block_size_minus2: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub init_qp_minus26: i8,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub log2_parallel_merge_level_minus2: u8,
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub column_width_minus1: [u16; 19],
    pub row_height_minus1: [u16; 21],
    /// `slice_parsing_fields` union arm (va_dec_hevc.h:139–164).
    pub slice_parsing_fields: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub num_short_term_ref_pic_sets: u8,
    pub num_long_term_ref_pic_sps: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub num_extra_slice_header_bits: u8,
    pub st_rps_bits: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM],
}

// VAPictureParameterBufferHEVC.pic_fields bit positions (LSB-first; va_dec_hevc.h:71).
pub const HEVC_PIC_CHROMA_FORMAT_IDC_SHIFT: u32 = 0; // 2 bits
pub const HEVC_PIC_SEPARATE_COLOUR_PLANE_FLAG: u32 = 1 << 2;
pub const HEVC_PIC_PCM_ENABLED_FLAG: u32 = 1 << 3;
pub const HEVC_PIC_SCALING_LIST_ENABLED_FLAG: u32 = 1 << 4;
pub const HEVC_PIC_TRANSFORM_SKIP_ENABLED_FLAG: u32 = 1 << 5;
pub const HEVC_PIC_AMP_ENABLED_FLAG: u32 = 1 << 6;
pub const HEVC_PIC_STRONG_INTRA_SMOOTHING_ENABLED_FLAG: u32 = 1 << 7;
pub const HEVC_PIC_SIGN_DATA_HIDING_ENABLED_FLAG: u32 = 1 << 8;
pub const HEVC_PIC_CONSTRAINED_INTRA_PRED_FLAG: u32 = 1 << 9;
pub const HEVC_PIC_CU_QP_DELTA_ENABLED_FLAG: u32 = 1 << 10;
pub const HEVC_PIC_WEIGHTED_PRED_FLAG: u32 = 1 << 11;
pub const HEVC_PIC_WEIGHTED_BIPRED_FLAG: u32 = 1 << 12;
pub const HEVC_PIC_TRANSQUANT_BYPASS_ENABLED_FLAG: u32 = 1 << 13;
pub const HEVC_PIC_TILES_ENABLED_FLAG: u32 = 1 << 14;
pub const HEVC_PIC_ENTROPY_CODING_SYNC_ENABLED_FLAG: u32 = 1 << 15;
pub const HEVC_PIC_PPS_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: u32 = 1 << 16;
pub const HEVC_PIC_LOOP_FILTER_ACROSS_TILES_ENABLED_FLAG: u32 = 1 << 17;
pub const HEVC_PIC_PCM_LOOP_FILTER_DISABLED_FLAG: u32 = 1 << 18;
pub const HEVC_PIC_NO_PIC_REORDERING_FLAG: u32 = 1 << 19;
pub const HEVC_PIC_NO_BI_PRED_FLAG: u32 = 1 << 20;

// VAPictureParameterBufferHEVC.slice_parsing_fields bit positions (va_dec_hevc.h:139).
pub const HEVC_SLP_LISTS_MODIFICATION_PRESENT_FLAG: u32 = 1 << 0;
pub const HEVC_SLP_LONG_TERM_REF_PICS_PRESENT_FLAG: u32 = 1 << 1;
pub const HEVC_SLP_SPS_TEMPORAL_MVP_ENABLED_FLAG: u32 = 1 << 2;
pub const HEVC_SLP_CABAC_INIT_PRESENT_FLAG: u32 = 1 << 3;
pub const HEVC_SLP_OUTPUT_FLAG_PRESENT_FLAG: u32 = 1 << 4;
pub const HEVC_SLP_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG: u32 = 1 << 5;
pub const HEVC_SLP_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG: u32 = 1 << 6;
pub const HEVC_SLP_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG: u32 = 1 << 7;
pub const HEVC_SLP_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG: u32 = 1 << 8;
pub const HEVC_SLP_PPS_DISABLE_DEBLOCKING_FILTER_FLAG: u32 = 1 << 9;
pub const HEVC_SLP_SLICE_SEGMENT_HEADER_EXTENSION_PRESENT_FLAG: u32 = 1 << 10;
pub const HEVC_SLP_RAP_PIC_FLAG: u32 = 1 << 11;
pub const HEVC_SLP_IDR_PIC_FLAG: u32 = 1 << 12;
pub const HEVC_SLP_INTRA_PIC_FLAG: u32 = 1 << 13;

/// `VASliceParameterBufferHEVC` (va_dec_hevc.h:358) — the once-per-slice **decode**
/// parameters for the *long slice format*: reference-list indices into
/// `VAPictureParameterBufferHEVC.ReferenceFrames[]`, the `LongSliceFlags` union arm,
/// QP/deblock/weight fields. Accompanied by a `VASliceDataBufferType` buffer holding
/// the raw slice NAL (start-code + emulation bytes intact).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VASliceParameterBufferHEVC {
    pub slice_data_size: u32,
    pub slice_data_offset: u32,
    pub slice_data_flag: u32,
    pub slice_data_byte_offset: u32,
    pub slice_segment_address: u32,
    /// `RefPicList[list][idx]` — index into `ReferenceFrames[]`, `0xFF` = invalid.
    pub RefPicList: [[u8; 15]; 2],
    /// `LongSliceFlags` union arm (va_dec_hevc.h:390–419).
    pub long_slice_flags: u32,
    pub collocated_ref_idx: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub slice_qp_delta: i8,
    pub slice_cb_qp_offset: i8,
    pub slice_cr_qp_offset: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_tc_offset_div2: i8,
    pub luma_log2_weight_denom: u8,
    pub delta_chroma_log2_weight_denom: i8,
    pub delta_luma_weight_l0: [i8; 15],
    pub luma_offset_l0: [i8; 15],
    pub delta_chroma_weight_l0: [[i8; 2]; 15],
    pub ChromaOffsetL0: [[i8; 2]; 15],
    pub delta_luma_weight_l1: [i8; 15],
    pub luma_offset_l1: [i8; 15],
    pub delta_chroma_weight_l1: [[i8; 2]; 15],
    pub ChromaOffsetL1: [[i8; 2]; 15],
    pub five_minus_max_num_merge_cand: u8,
    pub num_entry_point_offsets: u16,
    pub entry_offset_to_subset_array: u16,
    pub slice_data_num_emu_prevn_bytes: u16,
    pub va_reserved: [u32; VA_PADDING_LOW - 2],
}

// VASliceParameterBufferHEVC.long_slice_flags bit positions (va_dec_hevc.h:390).
pub const HEVC_LSF_LAST_SLICE_OF_PIC: u32 = 1 << 0;
pub const HEVC_LSF_DEPENDENT_SLICE_SEGMENT_FLAG: u32 = 1 << 1;
pub const HEVC_LSF_SLICE_TYPE_SHIFT: u32 = 2; // 2 bits
pub const HEVC_LSF_COLOR_PLANE_ID_SHIFT: u32 = 4; // 2 bits
pub const HEVC_LSF_SLICE_SAO_LUMA_FLAG: u32 = 1 << 6;
pub const HEVC_LSF_SLICE_SAO_CHROMA_FLAG: u32 = 1 << 7;
pub const HEVC_LSF_MVD_L1_ZERO_FLAG: u32 = 1 << 8;
pub const HEVC_LSF_CABAC_INIT_FLAG: u32 = 1 << 9;
pub const HEVC_LSF_SLICE_TEMPORAL_MVP_ENABLED_FLAG: u32 = 1 << 10;
pub const HEVC_LSF_SLICE_DEBLOCKING_FILTER_DISABLED_FLAG: u32 = 1 << 11;
pub const HEVC_LSF_COLLOCATED_FROM_L0_FLAG: u32 = 1 << 12;
pub const HEVC_LSF_SLICE_LOOP_FILTER_ACROSS_SLICES_ENABLED_FLAG: u32 = 1 << 13;

/// `VAIQMatrixBufferHEVC` (va_dec_hevc.h:561) — scaling lists, sent once per frame
/// only when `scaling_list_enabled_flag == 1`. For streams that do not signal custom
/// scaling lists the driver applies the flat default and no IQ buffer is submitted.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAIQMatrixBufferHEVC {
    pub ScalingList4x4: [[u8; 16]; 6],
    pub ScalingList8x8: [[u8; 64]; 6],
    pub ScalingList16x16: [[u8; 64]; 6],
    pub ScalingList32x32: [[u8; 64]; 2],
    pub ScalingListDC16x16: [u8; 6],
    pub ScalingListDC32x32: [u8; 2],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncSequenceParameterBufferHEVC` (va_enc_hevc.h:107).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncSequenceParameterBufferHEVC {
    pub general_profile_idc: u8,
    pub general_level_idc: u8,
    pub general_tier_flag: u8,
    pub intra_period: u32,
    pub intra_idr_period: u32,
    pub ip_period: u32,
    pub bits_per_second: u32,
    pub pic_width_in_luma_samples: u16,
    pub pic_height_in_luma_samples: u16,
    pub seq_fields: u32,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_transform_block_size_minus2: u8,
    pub log2_diff_max_min_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub pcm_sample_bit_depth_luma_minus1: u32,
    pub pcm_sample_bit_depth_chroma_minus1: u32,
    pub log2_min_pcm_luma_coding_block_size_minus3: u32,
    pub log2_max_pcm_luma_coding_block_size_minus3: u32,
    pub vui_parameters_present_flag: u8,
    pub vui_fields: u32,
    pub aspect_ratio_idc: u8,
    pub sar_width: u32,
    pub sar_height: u32,
    pub vui_num_units_in_tick: u32,
    pub vui_time_scale: u32,
    pub min_spatial_segmentation_idc: u16,
    pub max_bytes_per_pic_denom: u8,
    pub max_bits_per_min_cu_denom: u8,
    pub scc_fields: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM - 1],
}

/// `VAEncPictureParameterBufferHEVC` (va_enc_hevc.h:295).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncPictureParameterBufferHEVC {
    pub decoded_curr_pic: VAPictureHEVC,
    pub reference_frames: [VAPictureHEVC; 15],
    pub coded_buf: VABufferID,
    pub collocated_ref_pic_index: u8,
    pub last_picture: u8,
    pub pic_init_qp: u8,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub column_width_minus1: [u8; 19],
    pub row_height_minus1: [u8; 21],
    pub log2_parallel_merge_level_minus2: u8,
    pub ctu_max_bitsize_allowed: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub slice_pic_parameter_set_id: u8,
    pub nal_unit_type: u8,
    pub pic_fields: u32,
    pub hierarchical_level_plus1: u8,
    pub va_byte_reserved: u8,
    pub scc_fields: u16,
    pub va_reserved: [u32; VA_PADDING_HIGH - 1],
}

/// `VAEncSliceParameterBufferHEVC` (va_enc_hevc.h:472).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncSliceParameterBufferHEVC {
    pub slice_segment_address: u32,
    pub num_ctu_in_slice: u32,
    pub slice_type: u8,
    pub slice_pic_parameter_set_id: u8,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub ref_pic_list0: [VAPictureHEVC; 15],
    pub ref_pic_list1: [VAPictureHEVC; 15],
    pub luma_log2_weight_denom: u8,
    pub delta_chroma_log2_weight_denom: i8,
    pub delta_luma_weight_l0: [i8; 15],
    pub luma_offset_l0: [i8; 15],
    pub delta_chroma_weight_l0: [[i8; 2]; 15],
    pub chroma_offset_l0: [[i8; 2]; 15],
    pub delta_luma_weight_l1: [i8; 15],
    pub luma_offset_l1: [i8; 15],
    pub delta_chroma_weight_l1: [[i8; 2]; 15],
    pub chroma_offset_l1: [[i8; 2]; 15],
    pub max_num_merge_cand: u8,
    pub slice_qp_delta: i8,
    pub slice_cb_qp_offset: i8,
    pub slice_cr_qp_offset: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_tc_offset_div2: i8,
    pub slice_fields: u32,
    pub pred_weight_table_bit_offset: u32,
    pub pred_weight_table_bit_length: u32,
    pub va_reserved: [u32; VA_PADDING_MEDIUM - 2],
}

/// `VAEncSequenceParameterBufferVP8` (va_enc_vp8.h:51).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncSequenceParameterBufferVP8 {
    pub frame_width: u32,
    pub frame_height: u32,
    pub frame_width_scale: u32,
    pub frame_height_scale: u32,
    pub error_resilient: u32,
    pub kf_auto: u32,
    pub kf_min_dist: u32,
    pub kf_max_dist: u32,
    pub bits_per_second: u32,
    pub intra_period: u32,
    pub reference_frames: [VASurfaceID; 4],
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAEncPictureParameterBufferVP8` (va_enc_vp8.h:106). `ref_flags`/`pic_flags` are
/// the u32 union arms (bit order in the header, packed by the element).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAEncPictureParameterBufferVP8 {
    pub reconstructed_frame: VASurfaceID,
    pub ref_last_frame: VASurfaceID,
    pub ref_gf_frame: VASurfaceID,
    pub ref_arf_frame: VASurfaceID,
    pub coded_buf: VABufferID,
    pub ref_flags: u32,
    pub pic_flags: u32,
    pub loop_filter_level: [i8; 4],
    pub ref_lf_delta: [i8; 4],
    pub mode_lf_delta: [i8; 4],
    pub sharpness_level: u8,
    pub clamp_qindex_high: u8,
    pub clamp_qindex_low: u8,
    pub va_reserved: [u32; VA_PADDING_LOW],
}

/// `VAQMatrixBufferVP8` (va_enc_vp8.h:306) — per-segment base quantizer indices +
/// the five per-coefficient-class deltas (VP8 RFC 6386 §9.6).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VAQMatrixBufferVP8 {
    pub quantization_index: [u16; 4],
    pub quantization_index_delta: [i16; 5],
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

    // va.h — raster upload (the encode input path's PutImage fallback when the
    // driver refuses vaDeriveImage on an input surface).
    pub fn vaPutImage(
        dpy: VADisplay,
        surface: VASurfaceID,
        image: VAImageID,
        src_x: c_int,
        src_y: c_int,
        src_width: u32,
        src_height: u32,
        dest_x: c_int,
        dest_y: c_int,
        dest_width: u32,
        dest_height: u32,
    ) -> VAStatus;

    // va.h — DMA-BUF surface export (the zero-copy display path). `mem_type` is
    // `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`, `flags` an OR of the
    // `VA_EXPORT_SURFACE_*` bits, and `descriptor` a `*mut VADRMPRIMESurfaceDescriptor`
    // the driver fills (the surface must be synced first — `vaSyncSurface`). On success
    // the caller owns each `objects[i].fd` and must `close()` it (directly, or via an
    // importer that dups then closes).
    pub fn vaExportSurfaceHandle(
        dpy: VADisplay,
        surface_id: VASurfaceID,
        mem_type: u32,
        flags: u32,
        descriptor: *mut c_void,
    ) -> VAStatus;
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

    #[test]
    fn encode_struct_sizes_match_libva_2_23() {
        // 16 (4×u32) + 16 (2 ptrs, 8-aligned) + 16 (va_reserved) = 48.
        assert_eq!(size_of::<VACodedBufferSegment>(), 48, "VACodedBufferSegment");
        assert_eq!(align_of::<VACodedBufferSegment>(), 8);
        // 4+4+1 → pad to 12 for va_reserved[4] = 28.
        assert_eq!(
            size_of::<VAEncPackedHeaderParameterBuffer>(),
            28,
            "VAEncPackedHeaderParameterBuffer"
        );
        // 11×u32 + va_reserved[4] = 60.
        assert_eq!(
            size_of::<VAEncMiscParameterRateControl>(),
            60,
            "VAEncMiscParameterRateControl"
        );
        assert_eq!(size_of::<VAEncMiscParameterFrameRate>(), 24, "VAEncMiscParameterFrameRate");
        assert_eq!(size_of::<VAEncMiscParameterHRD>(), 24, "VAEncMiscParameterHRD");
        // 2 u8 → pad 4; 5×u32 (4..24); 2×u16 (24..28); seq_fields (28..32); 3 u8
        // (32..35) → pad 36; 2×i32 (36..44); [i32;256] (44..1068); crop flag u8 →
        // pad 1072; 4×u32 (1072..1088); vui flag u8 → pad 1092; vui_fields
        // (1092..1096); aspect u8 → pad 1100; 4×u32 (1100..1116); va_reserved[4]
        // (1116..1132).
        assert_eq!(
            size_of::<VAEncSequenceParameterBufferH264>(),
            1132,
            "VAEncSequenceParameterBufferH264"
        );
        // 36 + 576 = 612 (pics); coded_buf (612..616); 3 u8 (616..619) → pad 620;
        // frame_num u16 (620..622); 3 u8 + 2 i8 (622..627) → pad 628; pic_fields
        // (628..632); va_reserved[4] (632..648).
        assert_eq!(
            size_of::<VAEncPictureParameterBufferH264>(),
            648,
            "VAEncPictureParameterBufferH264"
        );
        // 12 (3×u32); 2 u8 + 2×u16 (12..18) → pad 20; i32×3 (20..32); 4 u8
        // (32..36); RefPicLists 2×1152 (36..2340); 3 u8 (2340..2343) → pad 2344;
        // [i16;32]×2 (2344..2472); flag u8 → pad 2474; [[i16;2];32]×2 (2474..2730);
        // flag u8 → pad 2732; ×2 (2732..2860); flag u8 → pad 2862; ×2 (2862..3118);
        // 5 u8/i8 (3118..3123) → pad 3124; va_reserved[4] (3124..3140).
        assert_eq!(
            size_of::<VAEncSliceParameterBufferH264>(),
            3140,
            "VAEncSliceParameterBufferH264"
        );
        assert_eq!(size_of::<VAPictureHEVC>(), 28, "VAPictureHEVC");
        // HEVC decode buffers (va_dec_hevc.h). CurrPic 28 + ReferenceFrames 15×28=420
        // → 448; 2×u16 (448..452); pic_fields (452..456); 20 u8/i8 (456..476);
        // column_width[19] u16 (476..514); row_height[21] u16 (514..556);
        // slice_parsing_fields (556..560); 8 u8/i8 (560..568); st_rps_bits (568..572);
        // va_reserved[8] (572..604).
        assert_eq!(
            size_of::<VAPictureParameterBufferHEVC>(),
            604,
            "VAPictureParameterBufferHEVC"
        );
        // 5×u32 (0..20); RefPicList[2][15] (20..50); long_slice_flags 4-aligned
        // (52..56); 10 u8/i8 (56..66); weight arrays 15+15+30+30 ×2 = 180 (66..246);
        // five_minus_max_num_merge_cand u8 (246..247); 3 u16 2-aligned (248..254);
        // va_reserved[2] 4-aligned (256..264).
        assert_eq!(
            size_of::<VASliceParameterBufferHEVC>(),
            264,
            "VASliceParameterBufferHEVC"
        );
        // 6×16 + 6×64 + 6×64 + 2×64 + 6 + 2 = 96+384+384+128+6+2 = 1000;
        // va_reserved[4] (1000..1016).
        assert_eq!(size_of::<VAIQMatrixBufferHEVC>(), 1016, "VAIQMatrixBufferHEVC");
        // 3 u8 → pad 4; 4×u32 (4..20); 2×u16 (20..24); seq_fields (24..28); 6 u8
        // (28..34) → pad 36; 4×u32 (36..52); vui flag u8 → pad 56; vui_fields
        // (56..60); aspect u8 → pad 64; 4×u32 (64..80); u16 + 2 u8 (80..84);
        // scc_fields (84..88); va_reserved[7] (88..116).
        assert_eq!(
            size_of::<VAEncSequenceParameterBufferHEVC>(),
            116,
            "VAEncSequenceParameterBufferHEVC"
        );
        // 28 + 15×28 = 448 (pics); coded_buf (448..452); 8 u8/i8 (452..460);
        // [u8;19]+[u8;21] (460..500); 6 u8 (500..506) → pad 508; pic_fields
        // (508..512); 2 u8 (512..514); scc u16 (514..516); va_reserved[15]
        // (516..576).
        assert_eq!(
            size_of::<VAEncPictureParameterBufferHEVC>(),
            576,
            "VAEncPictureParameterBufferHEVC"
        );
        // 8 (2×u32); 4 u8 (8..12); lists 2×420 (12..852); 2 (852..854); weight
        // arrays 15+15+30+30 ×2 = 180 (854..1034); 6 u8/i8 (1034..1040);
        // slice_fields (1040..1044); 2×u32 (1044..1052); va_reserved[6]
        // (1052..1076).
        assert_eq!(
            size_of::<VAEncSliceParameterBufferHEVC>(),
            1076,
            "VAEncSliceParameterBufferHEVC"
        );
        // 10×u32 + [u32;4] + va_reserved[4] = 72.
        assert_eq!(
            size_of::<VAEncSequenceParameterBufferVP8>(),
            72,
            "VAEncSequenceParameterBufferVP8"
        );
        // 5×u32 + 2×u32 flags + 12 i8 + 3 u8 (40..43) → pad 44; va_reserved[4]
        // (44..60).
        assert_eq!(
            size_of::<VAEncPictureParameterBufferVP8>(),
            60,
            "VAEncPictureParameterBufferVP8"
        );
        // [u16;4] (8) + [i16;5] (8..18) → pad 20; va_reserved[4] (20..36).
        assert_eq!(size_of::<VAQMatrixBufferVP8>(), 36, "VAQMatrixBufferVP8");
    }

    #[test]
    fn drm_prime_export_struct_sizes_match_va_drmcommon() {
        // Object: fd i32 (0..4) + size u32 (4..8) → pad 8 for the u64 modifier (8..16).
        assert_eq!(
            size_of::<VADRMPRIMESurfaceDescriptorObject>(),
            16,
            "VADRMPRIMESurfaceDescriptorObject"
        );
        assert_eq!(align_of::<VADRMPRIMESurfaceDescriptorObject>(), 8);
        // Layer: drm_format + num_planes (8) + object_index[4] + offset[4] + pitch[4]
        // (3×16 = 48) = 56, all u32 → align 4.
        assert_eq!(
            size_of::<VADRMPRIMESurfaceDescriptorLayer>(),
            56,
            "VADRMPRIMESurfaceDescriptorLayer"
        );
        assert_eq!(align_of::<VADRMPRIMESurfaceDescriptorLayer>(), 4);
        // Descriptor: fourcc/width/height/num_objects (16); objects[4] 8-aligned
        // (16 already 8-aligned) → 16 + 4×16 = 80; num_layers u32 (80..84); layers[4]
        // 4-aligned → 84 + 4×56 = 84 + 224 = 308. Whole struct align 8 (the u64 in
        // objects), 308 rounds up to 312.
        assert_eq!(
            size_of::<VADRMPRIMESurfaceDescriptor>(),
            312,
            "VADRMPRIMESurfaceDescriptor"
        );
        assert_eq!(align_of::<VADRMPRIMESurfaceDescriptor>(), 8);
    }
}
