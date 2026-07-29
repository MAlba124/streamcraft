//! SPA constants and param-object POD builders used by the ClientNode audio path (see
//! `REFERENCES.md`; values from the PipeWire 1.6 `spa-0.2` headers — `spa/utils/type.h`,
//! `spa/param/*.h`, `spa/node/io.h`, `spa/buffer/buffer.h`, and `pw_node_activation` in
//! `src/pipewire/private.h`). These are ABI values shared with the daemon, so they are exact.

#![allow(clippy::disallowed_methods)] // control-path param serialization, not a frame loop

use crate::native::pod::PodBuilder;

// --- SPA basic object type ids (`enum spa_type`, object range base 0x40000) ----------------

pub const TYPE_OBJECT_PROPS: u32 = 0x4_0002;
pub const TYPE_OBJECT_FORMAT: u32 = 0x4_0003;
pub const TYPE_OBJECT_PARAM_BUFFERS: u32 = 0x4_0004;
pub const TYPE_OBJECT_PARAM_META: u32 = 0x4_0005;
pub const TYPE_OBJECT_PARAM_IO: u32 = 0x4_0006;
/// `SPA_TYPE_COMMAND_Node` — the object type of node commands (Start/Pause/Suspend).
pub const TYPE_COMMAND_NODE: u32 = 0x3_0002;

// --- SPA_PARAM ids (`enum spa_param_type`) -------------------------------------------------

pub const PARAM_PROP_INFO: u32 = 1;
pub const PARAM_PROPS: u32 = 2;
pub const PARAM_ENUM_FORMAT: u32 = 3;
pub const PARAM_FORMAT: u32 = 4;
pub const PARAM_BUFFERS: u32 = 5;
pub const PARAM_META: u32 = 6;
pub const PARAM_IO: u32 = 7;

/// `enum spa_param_info` flags for the port's advertised param list.
pub const PARAM_INFO_READ: u32 = 1 << 1;
pub const PARAM_INFO_WRITE: u32 = 1 << 2;
pub const PARAM_INFO_RW: u32 = PARAM_INFO_READ | PARAM_INFO_WRITE;

// --- Format object keys (`enum spa_format`) + media/audio enums ----------------------------

pub const FORMAT_MEDIA_TYPE: u32 = 1;
pub const FORMAT_MEDIA_SUBTYPE: u32 = 2;
pub const FORMAT_AUDIO_FORMAT: u32 = 0x1_0001;
pub const FORMAT_AUDIO_RATE: u32 = 0x1_0003;
pub const FORMAT_AUDIO_CHANNELS: u32 = 0x1_0004;

pub const MEDIA_TYPE_AUDIO: u32 = 1;
pub const MEDIA_SUBTYPE_RAW: u32 = 1;
pub const MEDIA_SUBTYPE_DSP: u32 = 2;

/// `SPA_AUDIO_FORMAT_DSP_F32` (== `F32P`) — mono 32-bit float, the format of a PipeWire audio
/// device's per-channel DSP ports. A DSP node links 1:1 to those ports with no conversion.
pub const AUDIO_FORMAT_DSP_F32: u32 = 0x206;

// `enum spa_audio_format` (interleaved), little-endian variants (native on x86-64).
pub const AUDIO_FORMAT_U8: u32 = 0x102;
pub const AUDIO_FORMAT_S16_LE: u32 = 0x103;
pub const AUDIO_FORMAT_S32_LE: u32 = 0x10b;
pub const AUDIO_FORMAT_S24_LE: u32 = 0x10f;
pub const AUDIO_FORMAT_F32_LE: u32 = 0x11b;

// --- ParamBuffers object keys (`enum spa_param_buffers`) -----------------------------------

pub const BUFFERS_BUFFERS: u32 = 1;
pub const BUFFERS_BLOCKS: u32 = 2;
pub const BUFFERS_SIZE: u32 = 3;
pub const BUFFERS_STRIDE: u32 = 4;
pub const BUFFERS_DATA_TYPE: u32 = 6;

// --- POD Choice types (`enum spa_choice_type`) ---------------------------------------------

pub const CHOICE_RANGE: u32 = 1; // { default, min, max }
pub const CHOICE_FLAGS: u32 = 4; // first value is a flags mask

// --- spa_io ids + spa_io_buffers status (`spa/node/io.h`) ----------------------------------

pub const IO_BUFFERS: u32 = 1;
pub const IO_POSITION: u32 = 7;

pub const STATUS_NEED_DATA: i32 = 1 << 0;
pub const STATUS_HAVE_DATA: i32 = 1 << 1;

// --- spa_data types (`enum spa_data_type`) -------------------------------------------------

pub const DATA_MEM_PTR: u32 = 1;
pub const DATA_MEM_FD: u32 = 2;
pub const DATA_MEM_ID: u32 = 4;

// --- node change masks / flags (`spa/node/node.h`) + directions ----------------------------

pub const NODE_CHANGE_MASK_FLAGS: u64 = 1 << 0;
pub const NODE_CHANGE_MASK_PARAMS: u64 = 1 << 2;
pub const NODE_FLAG_RT: u64 = 1 << 0;

pub const PORT_CHANGE_MASK_FLAGS: u64 = 1 << 0;
pub const PORT_CHANGE_MASK_RATE: u64 = 1 << 1;
pub const PORT_CHANGE_MASK_PARAMS: u64 = 1 << 3;

pub const DIRECTION_OUTPUT: u32 = 1;

/// `SPA_ID_INVALID` — the "no buffer" sentinel in `spa_io_buffers.buffer_id`.
pub const ID_INVALID: u32 = 0xffff_ffff;

// --- node commands (`enum spa_node_command`) -----------------------------------------------

pub const NODE_COMMAND_SUSPEND: u32 = 0;
pub const NODE_COMMAND_PAUSE: u32 = 1;
pub const NODE_COMMAND_START: u32 = 2;

// --- pw_node_activation layout (offsets into the shared struct; private.h) -----------------
//
// Only the stable head of the struct is touched from the RT path — these offsets are
// unambiguous (start-of-struct) and ABI-frozen for `PW_VERSION_NODE_ACTIVATION`:
//
//   0  uint32 status                    (NOT_TRIGGERED/TRIGGERED/AWAKE/FINISHED/INACTIVE)
//   8  state[0].status  (int32)
//   12 state[0].required(int32)
//   16 state[0].pending (int32)         <- decremented to trigger a peer
//   ...
//   560 spa_io_position position; -> +96 clock.duration (u64) -> abs 656

pub const ACT_STATUS: usize = 0;
pub const ACT_STATE0_PENDING: usize = 16;
/// `position.clock.duration` (frames this cycle). Read with a sanity fallback — a wrong offset
/// only yields a bad value, never an out-of-bounds read (the mapped activation is far larger).
pub const ACT_CLOCK_DURATION: usize = 656;

pub const ACTIVATION_NOT_TRIGGERED: u32 = 0;
pub const ACTIVATION_TRIGGERED: u32 = 1;
pub const ACTIVATION_AWAKE: u32 = 2;
pub const ACTIVATION_FINISHED: u32 = 3;

/// `sizeof(struct spa_chunk)` — the per-data-block `{offset,size,stride,flags}` descriptor.
pub const CHUNK_SIZE: usize = 16;

/// The five interleaved PCM formats the sink handles, paired with their SPA id and byte width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    U8,
    S16,
    S24,
    S32,
    F32,
}

impl SampleFormat {
    pub fn from_name(name: &str) -> Option<SampleFormat> {
        Some(match name {
            "u8" => SampleFormat::U8,
            "s16" => SampleFormat::S16,
            "s24" => SampleFormat::S24,
            "s32" => SampleFormat::S32,
            "f32" => SampleFormat::F32,
            _ => return None,
        })
    }
    /// The `enum spa_audio_format` id.
    pub fn spa(self) -> u32 {
        match self {
            SampleFormat::U8 => AUDIO_FORMAT_U8,
            SampleFormat::S16 => AUDIO_FORMAT_S16_LE,
            SampleFormat::S24 => AUDIO_FORMAT_S24_LE,
            SampleFormat::S32 => AUDIO_FORMAT_S32_LE,
            SampleFormat::F32 => AUDIO_FORMAT_F32_LE,
        }
    }
    /// Bytes per sample.
    pub fn bytes(self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::S16 => 2,
            SampleFormat::S24 => 3,
            SampleFormat::S32 | SampleFormat::F32 => 4,
        }
    }
}

// --- param object builders (appended into the message's arg struct) ------------------------

/// A fixed `SPA_TYPE_OBJECT_Format` describing interleaved raw audio, tagged with `param_id`
/// (`EnumFormat` when advertising, `Format` when confirming a negotiated format).
pub fn build_format(b: &mut PodBuilder, param_id: u32, fmt: SampleFormat, rate: u32, channels: u32) {
    b.push_object(TYPE_OBJECT_FORMAT, param_id);
    b.property(FORMAT_MEDIA_TYPE, 0).id(MEDIA_TYPE_AUDIO);
    b.property(FORMAT_MEDIA_SUBTYPE, 0).id(MEDIA_SUBTYPE_RAW);
    b.property(FORMAT_AUDIO_FORMAT, 0).id(fmt.spa());
    b.property(FORMAT_AUDIO_RATE, 0).int(rate as i32);
    b.property(FORMAT_AUDIO_CHANNELS, 0).int(channels as i32);
    b.pop();
}

/// A DSP-port `Format`: mono `DSP_F32`, no rate/channels (they come from the graph). Matches an
/// audio device's `playback_FL`/`FR` ports so we link 1:1 with no adapter/conversion.
pub fn build_format_dsp(b: &mut PodBuilder, param_id: u32) {
    b.push_object(TYPE_OBJECT_FORMAT, param_id);
    b.property(FORMAT_MEDIA_TYPE, 0).id(MEDIA_TYPE_AUDIO);
    b.property(FORMAT_MEDIA_SUBTYPE, 0).id(MEDIA_SUBTYPE_DSP);
    b.property(FORMAT_AUDIO_FORMAT, 0).id(AUDIO_FORMAT_DSP_F32);
    b.pop();
}

/// A `SPA_TYPE_OBJECT_ParamBuffers` declaring our buffer constraints: a block of `size` bytes
/// per buffer, `stride`-aligned, with a small pool (the server reconciles with the sink and
/// allocates the memfds). `size` should be ~one quantum of PCM.
pub fn build_buffers(b: &mut PodBuilder, size: u32, stride: u32) {
    b.push_object(TYPE_OBJECT_PARAM_BUFFERS, PARAM_BUFFERS);
    // A generous pool so the RT cycle always has a free buffer (round-robin, no recycle races).
    b.property(BUFFERS_BUFFERS, 0).choice_int(CHOICE_RANGE, &[8, 2, 32]);
    b.property(BUFFERS_BLOCKS, 0).int(1);
    b.property(BUFFERS_SIZE, 0).choice_int(CHOICE_RANGE, &[size as i32, size as i32, (size * 4) as i32]);
    b.property(BUFFERS_STRIDE, 0).int(stride as i32);
    // Cross-process buffers must be shareable memory: advertise MemFd (and MemPtr) or the
    // daemon cannot allocate the link's buffers ("alloc buffers: Invalid argument").
    let data_types = (1i32 << DATA_MEM_FD) | (1i32 << DATA_MEM_PTR);
    b.property(BUFFERS_DATA_TYPE, 0).choice_int(CHOICE_FLAGS, &[data_types]);
    b.pop();
}
