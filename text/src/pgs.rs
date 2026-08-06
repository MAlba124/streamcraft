//! PGS (HDMV Presentation Graphics Stream) decode — the image-based BluRay subtitle format
//! (spec: RFC 9559 §12.7 Matroska `S_HDMV/PGS` mapping; the widely-documented "Blu-ray
//! SUP/PGS" segment grammar; the HDMV run-length scheme in US patent 7,634,739 "Method and
//! apparatus for reproducing a data recorded on a recording medium"). Clean-room from the
//! format grammar — no libavcodec `pgssubdec` was consulted.
//!
//! # What a PGS stream is
//!
//! A PGS subtitle stream is a sequence of **Display Sets**. A Display Set is one caption's
//! worth of state: which bitmap objects are on screen, where, and with what colour palette.
//! Each Display Set is a run of **segments**, every segment framed as:
//!
//! ```text
//!   segment_type  u8      (0x14 PDS, 0x15 ODS, 0x16 PCS, 0x17 WDS, 0x80 END)
//!   segment_size  u16 BE  (length of the payload that follows)
//!   payload       [segment_size] bytes
//! ```
//!
//! A Display Set always opens with a **PCS** (Presentation Composition Segment) and closes
//! with an **END** segment. The segment kinds:
//!
//! * **PCS** `0x16` — the composition: the reference video size, a *composition state*
//!   (`0x80` epoch-start, `0x40` acquisition-point, `0x00` normal), a palette id, and a list
//!   of *composition objects* placing object bitmaps at `(x, y)`. **Zero composition objects
//!   is a clear** — erase whatever caption is currently shown.
//! * **WDS** `0x17` — Window Definition: the on-screen rectangles ("windows") captions live
//!   in. We read them for completeness but placement comes from the PCS composition object.
//! * **PDS** `0x14` — Palette Definition: up to 256 entries, each `entry_id, Y, Cr, Cb, A`.
//!   **PGS palettes are Y′CrCb + alpha, not RGB** — and note the channel order is Y, then
//!   **Cr**, then **Cb** (the opposite of the usual CbCr grouping). We convert to RGBA with
//!   the BT.709 matrix (these are 1080p BluRay sources — Rec. 709 limited range).
//! * **ODS** `0x15` — Object Definition: an object id, a *sequence flag* (first / last /
//!   both fragment), a 3-byte data length, the object `width`/`height` (only on the first
//!   fragment), and the **RLE-encoded** indexed bitmap. A large object may be split across
//!   several ODS fragments (first, then middle(s), then last) which we concatenate before
//!   decoding.
//! * **END** `0x80` — end of the Display Set. Zero-length payload.
//!
//! # The HDMV run-length scheme (`decode_rle`)
//!
//! Each row of the object bitmap is a sequence of runs over 8-bit palette indices, terminated
//! by an end-of-line marker. The escape byte is `0x00`:
//!
//! ```text
//!   CCCCCCCC                      one pixel, colour C   (C != 0)
//!   00000000 00000000             end of line
//!   00000000 00LLLLLL             L pixels (1..63)   of colour 0
//!   00000000 01LLLLLL LLLLLLLL    L pixels (64..16383) of colour 0
//!   00000000 10LLLLLL CCCCCCCC    L pixels (3..63)   of colour C
//!   00000000 11LLLLLL LLLLLLLL CCCCCCCC   L pixels (64..16383) of colour C
//! ```
//!
//! (A single non-zero byte is one pixel of that colour; the two top bits after the `0x00`
//! escape select the run form.) We decode into an `width*height` index plane, then map each
//! index through the palette to RGBA. Every length is bounds-checked against the row/plane so
//! a malformed run warns-and-drops the Display Set rather than over-writing or panicking
//! (spec: untrusted input is a P0).
//!
//! # SUP vs. mkv framing (the caller's concern, documented here)
//!
//! A raw `.sup` file (what `ffmpeg -c copy` writes) prefixes **every** segment with a 10-byte
//! header: the 2-byte magic `PG`, a 32-bit presentation timestamp and a 32-bit decode
//! timestamp (both 90 kHz). Inside an mkv, the demuxer has already stripped that — an mkv
//! Block carries the bare segments of **one Display Set**, and the timing rides the Block's
//! PTS + `BlockDuration`. This module parses the bare-segment form ([`parse_display_set`]);
//! the `.sup` header is peeled by [`strip_sup_header`], used only by the offline test/validation
//! path. `pgsdec` feeds it mkv Blocks and never sees a `PG` header.
//!
//! # v1 scope
//!
//! v1 handles the mainstream **show / clear** display sets every BluRay rip uses (an object
//! composition, then a later empty composition that clears it). Palette-update-only display
//! sets (`palette_update_flag` with no new objects — used for fades), forced-subtitle flags,
//! object cropping rectangles, and multi-object compositions beyond the first are **not**
//! specially handled: cropping is ignored, extra objects are composited in order, and a
//! palette-only update is treated as a no-op refresh (a `warn` is the caller's, see
//! `pgsdec`). These are documented follow-ups, not silent data loss.

// COLD: PGS decode runs once per sparse subtitle caption (a Display Set every few seconds via
// `pgsdec`, never per video frame). The RGBA/index/accumulator Vecs are the decoded-caption
// payload — bounded and one-per-caption, not steady-state heap. (`pgsdec` sends output through
// the pool; the overlay composites from `ctx`-owned buffers, not these directly.)
#![allow(clippy::disallowed_methods)]

/// A decoded PGS Display Set ready to composite: the RGBA pixels of the (first) composition
/// object, its size, and its top-left placement in the reference video frame. A **clear**
/// Display Set (no composition objects) decodes to [`DisplaySet::clear`] — `rgba` empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplaySet {
    /// Straight-alpha RGBA8888, row-major, `width*height*4` bytes. Empty for a clear.
    pub rgba: Vec<u8>,
    /// Object bitmap dimensions in pixels.
    pub width: u32,
    pub height: u32,
    /// Top-left placement of the object in the reference video frame (from the PCS
    /// composition object). `(0, 0)` for a clear.
    pub x: u32,
    pub y: u32,
    /// The composition's reference video size (PCS width/height) — the coordinate space
    /// `(x, y)` is expressed in. The overlay scales this to the actual decoded frame size.
    pub video_width: u32,
    pub video_height: u32,
}

impl DisplaySet {
    /// A clear Display Set (a composition with no objects): erase the current caption.
    pub fn clear(video_width: u32, video_height: u32) -> Self {
        DisplaySet {
            rgba: Vec::new(),
            width: 0,
            height: 0,
            x: 0,
            y: 0,
            video_width,
            video_height,
        }
    }

    /// True when this Display Set clears the current caption (no bitmap).
    pub fn is_clear(&self) -> bool {
        self.rgba.is_empty()
    }
}

/// A parse error in a PGS Display Set. The caller (`pgsdec`) warns-and-drops the Display Set;
/// none of these is stream-fatal (spec: untrusted input is a P0 — malformed data never panics).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PgsError {
    /// A segment header or payload ran out of bytes before a declared field.
    Truncated(&'static str),
    /// A structurally impossible value (a run length past the row, a bad segment ordering).
    Malformed(&'static str),
    /// The Display Set never carried a PCS (nothing to compose).
    NoComposition,
}

impl PgsError {
    /// A `&'static` one-line reason, for a `log!` field (which takes `&'static str`, not an
    /// owned `String`). The `Truncated`/`Malformed` payload is already `&'static`.
    pub fn reason(&self) -> &'static str {
        match self {
            PgsError::Truncated(what) => what,
            PgsError::Malformed(what) => what,
            PgsError::NoComposition => "display set has no PCS",
        }
    }
}

/// The PGS segment type bytes (the BluRay HDMV segment kinds).
const SEG_PDS: u8 = 0x14;
const SEG_ODS: u8 = 0x15;
const SEG_PCS: u8 = 0x16;
const SEG_WDS: u8 = 0x17;
const SEG_END: u8 = 0x80;

/// The `PG` magic prefixing every segment in a raw `.sup` file.
const SUP_MAGIC: [u8; 2] = [0x50, 0x47];

/// A 256-entry PGS palette: each index → (Y′, Cr, Cb, A). Index 0 defaults to fully
/// transparent (the standard "no colour" background), the rest to opaque black until a PDS
/// entry overrides them.
#[derive(Clone)]
struct Palette {
    /// `[Y, Cr, Cb, A]` per entry, 256 entries.
    entries: [[u8; 4]; 256],
}

impl Palette {
    fn new() -> Self {
        // Default: transparent everywhere (A=0). A PDS fills in the used entries; any index
        // the RLE references that the PDS never defined stays transparent — the safe default.
        Palette { entries: [[0, 128, 128, 0]; 256] }
    }

    /// Apply a PDS payload: `palette_id(1) palette_version(1)` then N × `id(1) Y(1) Cr(1)
    /// Cb(1) A(1)`. Ignores the id/version (v1 tracks a single palette per Display Set).
    fn apply_pds(&mut self, payload: &[u8]) -> Result<(), PgsError> {
        if payload.len() < 2 {
            return Err(PgsError::Truncated("PDS header"));
        }
        let body = &payload[2..];
        // Trailing bytes that don't form a full 5-byte entry are malformed framing.
        if !body.len().is_multiple_of(5) {
            return Err(PgsError::Malformed("PDS entry array not a multiple of 5"));
        }
        for e in body.as_chunks::<5>().0 {
            let idx = e[0] as usize;
            self.entries[idx] = [e[1], e[2], e[3], e[4]];
        }
        Ok(())
    }

    /// Map an 8-bit palette index to straight-alpha RGBA, converting Y′CrCb→RGB with the
    /// BT.709 limited-range matrix (ITU-R BT.709; these are 1080p BluRay sources). The
    /// alpha channel is passed through unchanged (PGS alpha is already straight).
    #[inline]
    fn rgba(&self, index: u8) -> [u8; 4] {
        let [y, cr, cb, a] = self.entries[index as usize];
        let [r, g, b] = ycrcb_to_rgb_bt709(y, cr, cb);
        [r, g, b, a]
    }
}

/// Convert a limited-range BT.709 Y′CrCb sample to full-range 8-bit RGB (ITU-R BT.709
/// §3.2/§3.3; limited-range coding: Y′ ∈ [16, 235], Cb/Cr ∈ [16, 240] centred at 128).
///
/// ```text
///   R = 1.16438·(Y-16)                    + 1.79274·(Cr-128)
///   G = 1.16438·(Y-16) − 0.21325·(Cb-128) − 0.53291·(Cr-128)
///   B = 1.16438·(Y-16) + 2.11240·(Cb-128)
/// ```
///
/// Coefficients are the Rec. 709 luma weights (Kr=0.2126, Kb=0.0722) inverted for the
/// limited→full range expansion. Fixed-point would do, but a subtitle palette is ≤256
/// entries decoded once per Display Set, so float is fine and clearer.
#[inline]
fn ycrcb_to_rgb_bt709(y: u8, cr: u8, cb: u8) -> [u8; 3] {
    let yf = (y as f32 - 16.0) * 1.164_383_6;
    let crf = cr as f32 - 128.0;
    let cbf = cb as f32 - 128.0;
    let r = yf + 1.792_741_1 * crf;
    let g = yf - 0.213_248_6 * cbf - 0.532_909_3 * crf;
    let b = yf + 2.112_401_7 * cbf;
    [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        (v + 0.5) as u8
    }
}

/// A composition object placement from the PCS: which object, and where.
struct CompObject {
    object_id: u16,
    x: u32,
    y: u32,
}

/// The parsed PCS composition: reference video size and the placed objects (empty = clear).
struct Composition {
    video_width: u32,
    video_height: u32,
    objects: Vec<CompObject>,
}

/// Parse a PCS payload into a [`Composition`]. Layout:
/// `width u16, height u16, frame_rate u8, composition_number u16, composition_state u8,
/// palette_update_flag u8, palette_id u8, number_of_composition_objects u8`, then per object:
/// `object_id u16, window_id u8, object_cropped_flag u8, x u16, y u16` (+ an 8-byte crop
/// rectangle when the flag's `0x40` bit is set — read and ignored in v1).
fn parse_pcs(payload: &[u8]) -> Result<Composition, PgsError> {
    let mut r = Reader::new(payload);
    let video_width = r.u16("PCS width")? as u32;
    let video_height = r.u16("PCS height")? as u32;
    r.skip(1, "PCS frame_rate")?;
    r.skip(2, "PCS composition_number")?;
    r.skip(1, "PCS composition_state")?;
    r.skip(1, "PCS palette_update_flag")?;
    r.skip(1, "PCS palette_id")?;
    let num_objects = r.u8("PCS number_of_composition_objects")?;
    let mut objects = Vec::with_capacity(num_objects as usize);
    for _ in 0..num_objects {
        let object_id = r.u16("PCS object_id")?;
        r.skip(1, "PCS window_id")?;
        let cropped = r.u8("PCS object_cropped_flag")?;
        let x = r.u16("PCS x")? as u32;
        let y = r.u16("PCS y")? as u32;
        // A cropped object carries an extra 8-byte crop rectangle we skip (v1 ignores cropping).
        if cropped & 0x40 != 0 {
            r.skip(8, "PCS crop rectangle")?;
        }
        objects.push(CompObject { object_id, x, y });
    }
    Ok(Composition { video_width, video_height, objects })
}

/// An object being reassembled across ODS fragments: its declared size and the concatenated
/// RLE bytes (the first fragment carries width/height, all fragments carry RLE).
struct ObjectAccum {
    width: u32,
    height: u32,
    rle: Vec<u8>,
    complete: bool,
}

/// Feed one ODS payload into the per-object accumulator map. Layout:
/// `object_id u16, object_version u8, sequence_flag u8, object_data_length u24`, and on the
/// **first** fragment (`sequence_flag & 0x80`) also `width u16, height u16` before the RLE.
/// `sequence_flag & 0x40` marks the last fragment. The 24-bit data length counts the 4
/// width/height bytes plus the RLE of the whole object (so it spans fragments).
fn parse_ods(payload: &[u8], objects: &mut Vec<(u16, ObjectAccum)>) -> Result<(), PgsError> {
    let mut r = Reader::new(payload);
    let object_id = r.u16("ODS object_id")?;
    r.skip(1, "ODS object_version")?;
    let seq = r.u8("ODS sequence_flag")?;
    let is_first = seq & 0x80 != 0;
    let is_last = seq & 0x40 != 0;
    if is_first {
        // data_length (u24) counts the width/height + RLE of the whole object; we validate
        // by reassembly length, not this field, so we only skip it.
        r.skip(3, "ODS object_data_length")?;
        let width = r.u16("ODS width")? as u32;
        let height = r.u16("ODS height")? as u32;
        let rle = r.rest().to_vec();
        // Start a fresh accumulator for this object id (a new first fragment supersedes).
        upsert(objects, object_id, ObjectAccum { width, height, rle, complete: is_last });
    } else {
        // A continuation fragment: still has the 3-byte data_length field, then raw RLE.
        r.skip(3, "ODS object_data_length (continuation)")?;
        let more = r.rest();
        match objects.iter_mut().find(|(id, _)| *id == object_id) {
            Some((_, acc)) => {
                acc.rle.extend_from_slice(more);
                acc.complete |= is_last;
            }
            // A continuation with no first fragment is malformed framing — drop the Display Set.
            None => return Err(PgsError::Malformed("ODS continuation without a first fragment")),
        }
    }
    Ok(())
}

/// Insert or replace an object accumulator keyed by object id.
fn upsert(objects: &mut Vec<(u16, ObjectAccum)>, id: u16, acc: ObjectAccum) {
    match objects.iter_mut().find(|(oid, _)| *oid == id) {
        Some(slot) => slot.1 = acc,
        None => objects.push((id, acc)),
    }
}

/// Parse one **Display Set** (a run of bare segments, no `.sup` `PG` headers) into a decoded
/// [`DisplaySet`]. `mkvdemux` hands one Block = one Display Set; this is `pgsdec`'s core call.
///
/// The Display Set opens with a PCS and ends with an END; PDS updates the palette, ODS
/// fragments assemble object bitmaps. A composition with no objects (or one whose objects have
/// no bitmap data) is a **clear**. Returns [`PgsError`] on malformed framing (the caller
/// warns-and-drops) — never panics on adversarial input.
pub fn parse_display_set(data: &[u8]) -> Result<DisplaySet, PgsError> {
    let mut palette = Palette::new();
    let mut objects: Vec<(u16, ObjectAccum)> = Vec::new();
    let mut composition: Option<Composition> = None;

    let mut at = 0usize;
    while at < data.len() {
        // Segment header: type u8, size u16 BE.
        let seg_type = *data.get(at).ok_or(PgsError::Truncated("segment type"))?;
        let size = u16::from_be_bytes([
            *data.get(at + 1).ok_or(PgsError::Truncated("segment size hi"))?,
            *data.get(at + 2).ok_or(PgsError::Truncated("segment size lo"))?,
        ]) as usize;
        let body_start = at + 3;
        let body_end = body_start
            .checked_add(size)
            .ok_or(PgsError::Malformed("segment size overflow"))?;
        let payload = data
            .get(body_start..body_end)
            .ok_or(PgsError::Truncated("segment payload"))?;
        at = body_end;

        match seg_type {
            SEG_PCS => composition = Some(parse_pcs(payload)?),
            SEG_WDS => { /* window rectangles — read implicitly, placement is from the PCS */ }
            SEG_PDS => palette.apply_pds(payload)?,
            SEG_ODS => parse_ods(payload, &mut objects)?,
            SEG_END => break,
            // An unknown segment type inside a Display Set is skipped (its length framed it):
            // forward-compatible with segment kinds v1 does not model.
            _ => {}
        }
    }

    let comp = composition.ok_or(PgsError::NoComposition)?;

    // No composition objects → a clear (erase the current caption).
    if comp.objects.is_empty() {
        return Ok(DisplaySet::clear(comp.video_width, comp.video_height));
    }

    // v1 composites the first composition object (the mainstream single-object caption). If it
    // references an object with no ODS data (an odd composition), treat it as a clear.
    let first = &comp.objects[0];
    let Some((_, acc)) = objects.iter().find(|(id, _)| *id == first.object_id) else {
        return Ok(DisplaySet::clear(comp.video_width, comp.video_height));
    };

    let rgba = decode_object_rgba(acc, &palette)?;
    Ok(DisplaySet {
        rgba,
        width: acc.width,
        height: acc.height,
        x: first.x,
        y: first.y,
        video_width: comp.video_width,
        video_height: comp.video_height,
    })
}

/// Decode one assembled object's RLE into an RGBA image via the palette. Bounds-checks the
/// object size (a maliciously huge `width*height` is rejected before allocation) and every
/// run (never writes past a row). See the module docs for the HDMV RLE grammar.
fn decode_object_rgba(acc: &ObjectAccum, palette: &Palette) -> Result<Vec<u8>, PgsError> {
    let (w, h) = (acc.width as usize, acc.height as usize);
    // A zero-dimension object is degenerate but not fatal — an empty (clear-like) bitmap.
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    // Reject an object whose pixel count would overflow / balloon memory before allocating
    // (spec: untrusted input). 8K² is far above any real subtitle; anything larger is bogus.
    let pixels = w.checked_mul(h).ok_or(PgsError::Malformed("object size overflow"))?;
    if pixels > 8192 * 8192 {
        return Err(PgsError::Malformed("object size implausibly large"));
    }

    let indices = decode_rle(&acc.rle, w, h)?;
    let mut rgba = vec![0u8; pixels * 4];
    for (i, &idx) in indices.iter().enumerate() {
        rgba[i * 4..i * 4 + 4].copy_from_slice(&palette.rgba(idx));
    }
    Ok(rgba)
}

/// Decode HDMV run-length data into a `width*height` index plane (one palette index per
/// pixel). Rows are decoded independently; an end-of-line marker (`00 00`) advances to the
/// next row. Under-filled rows are zero-padded (colour 0, the transparent background);
/// over-filled rows are clamped (a malformed run cannot write past the row). See the module
/// docs for the run grammar.
fn decode_rle(rle: &[u8], width: usize, height: usize) -> Result<Vec<u8>, PgsError> {
    let mut out = vec![0u8; width * height];
    let mut i = 0usize;
    let mut row = 0usize;
    let mut col = 0usize;

    // Helper: write `count` pixels of `colour` into the current row, clamped to the row end.
    // Returns the (possibly clamped) new column.
    // Write `count` pixels of `colour` into the current row starting at `col`, **clamped** to
    // the row end (a run past the row is silently truncated — a well-formed stream always emits
    // an explicit end-of-line, so a run only ever fills up to the row edge). Returns the new
    // column. Guards `row >= height` (writes nothing past the last row).
    let put = |out: &mut [u8], row: usize, col: usize, count: usize, colour: u8| -> usize {
        if row >= height || col >= width {
            return col; // row already full / no row left — wait for the EOL marker
        }
        let start = row * width + col;
        let end = (col + count).min(width);
        let n = end - col;
        if n > 0 {
            out[start..start + n].fill(colour);
        }
        end
    };

    // Row advance is driven **only** by the end-of-line marker (`00 00`); a run that fills the
    // row does not auto-advance (the encoder emits an explicit EOL). This avoids a double
    // advance when a run exactly fills the row and is then followed by its EOL.
    while i < rle.len() && row < height {
        let b0 = rle[i];
        i += 1;
        if b0 != 0 {
            // A single pixel of colour b0.
            col = put(&mut out, row, col, 1, b0);
            continue;
        }
        // Escape 0x00 — read the run descriptor.
        let b1 = *rle.get(i).ok_or(PgsError::Truncated("RLE run byte"))?;
        i += 1;
        if b1 == 0 {
            // End of line.
            row += 1;
            col = 0;
            continue;
        }
        let run_form = b1 >> 6; // top two bits
        let (count, colour) = match run_form {
            0b00 => {
                // 1..63 pixels of colour 0.
                ((b1 & 0x3F) as usize, 0u8)
            }
            0b01 => {
                // 64..16383 pixels of colour 0 (length low 6 bits + next byte).
                let lo = *rle.get(i).ok_or(PgsError::Truncated("RLE long-run length"))?;
                i += 1;
                ((((b1 & 0x3F) as usize) << 8) | lo as usize, 0u8)
            }
            0b10 => {
                // 3..63 pixels of a colour (colour in the next byte).
                let c = *rle.get(i).ok_or(PgsError::Truncated("RLE short-colour byte"))?;
                i += 1;
                ((b1 & 0x3F) as usize, c)
            }
            // 0b11
            _ => {
                // 64..16383 pixels of a colour (length + colour in two more bytes).
                let lo = *rle.get(i).ok_or(PgsError::Truncated("RLE long-run length"))?;
                i += 1;
                let c = *rle.get(i).ok_or(PgsError::Truncated("RLE long-colour byte"))?;
                i += 1;
                ((((b1 & 0x3F) as usize) << 8) | lo as usize, c)
            }
        };
        col = put(&mut out, row, col, count, colour);
    }
    Ok(out)
}

/// Peel the raw `.sup` framing off a whole SUP file into the concatenated bare segments of
/// each Display Set, grouped by the segments' presentation timestamp. Each SUP record is a
/// 10-byte header (`PG` magic, PTS u32, DTS u32, both 90 kHz) followed by one segment
/// (`type u8, size u16, payload`). A Display Set runs from a PCS up to and including its END.
///
/// Returns `(pts_90k, dts_90k, display_set_bytes)` per Display Set — the bare-segment form
/// [`parse_display_set`] consumes. This is the **offline test/validation path only**; the mkv
/// element never sees a `PG` header (the demuxer strips it — see the module docs). Malformed
/// records stop the scan (best-effort), returning what parsed cleanly.
pub fn split_sup(mut data: &[u8]) -> Vec<(u32, u32, Vec<u8>)> {
    let mut sets = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut cur_pts = 0u32;
    let mut cur_dts = 0u32;
    let mut have_pcs = false;
    while data.len() >= 13 {
        if data[..2] != SUP_MAGIC {
            break; // lost sync — stop, keep what we have
        }
        let pts = u32::from_be_bytes([data[2], data[3], data[4], data[5]]);
        let dts = u32::from_be_bytes([data[6], data[7], data[8], data[9]]);
        let seg_type = data[10];
        let size = u16::from_be_bytes([data[11], data[12]]) as usize;
        let record_end = 13 + size;
        if data.len() < record_end {
            break; // truncated tail
        }
        let seg = &data[10..record_end]; // bare segment: type, size, payload
        if seg_type == SEG_PCS {
            // A PCS opens a Display Set — stamp its timing.
            cur.clear();
            cur_pts = pts;
            cur_dts = dts;
            have_pcs = true;
        }
        if have_pcs {
            cur.extend_from_slice(seg);
        }
        if seg_type == SEG_END && have_pcs {
            sets.push((cur_pts, cur_dts, std::mem::take(&mut cur)));
            have_pcs = false;
        }
        data = &data[record_end..];
    }
    sets
}

/// Strip a single `.sup` record's 10-byte `PG`+timestamps header, returning the bare segment
/// bytes and the 90 kHz PTS. `None` if the magic is wrong or the record is truncated. Handy
/// for a targeted test; whole-file scanning uses [`split_sup`].
pub fn strip_sup_header(record: &[u8]) -> Option<(u32, &[u8])> {
    if record.len() < 13 || record[..2] != SUP_MAGIC {
        return None;
    }
    let pts = u32::from_be_bytes([record[2], record[3], record[4], record[5]]);
    let size = u16::from_be_bytes([record[11], record[12]]) as usize;
    let end = 13 + size;
    let seg = record.get(10..end)?;
    Some((pts, seg))
}

// ============================ the `subtitle/bitmap` wire format ============================
//
// A decoded [`DisplaySet`] travels from `pgsdec` to `suboverlay` as one buffer on the
// `subtitle/bitmap` family (`crate::BITMAP_FAMILY`). Geometry is *per-buffer* — every caption
// has its own size/position — so it rides in a small fixed header on the payload rather than
// in the (fieldless) negotiated caps, exactly as `subtitle/events` carries its text inline.
// Timing (the [pts, pts+dur) span, clear semantics) rides the buffer's PTS + duration.
//
//   magic 'B' 'S' 'U' 'B'  (4 bytes)
//   width         u32 BE     object bitmap width  (0 = clear)
//   height        u32 BE     object bitmap height (0 = clear)
//   x             u32 BE     placement x in the reference video frame
//   y             u32 BE     placement y
//   video_width   u32 BE     reference composition width  (the coordinate space of x/y)
//   video_height  u32 BE     reference composition height
//   rgba          [width*height*4] bytes, straight-alpha RGBA8888

/// The 4-byte magic prefixing a `subtitle/bitmap` payload.
pub const BITMAP_MAGIC: [u8; 4] = *b"BSUB";
/// Header length preceding the RGBA pixels (magic + six u32 fields).
pub const BITMAP_HEADER: usize = 4 + 6 * 4;

/// Encode a decoded [`DisplaySet`] to the `subtitle/bitmap` wire payload (header + RGBA). A
/// clear encodes to a bare header with `width == height == 0`.
pub fn encode_bitmap(ds: &DisplaySet) -> Vec<u8> {
    let mut out = Vec::with_capacity(BITMAP_HEADER + ds.rgba.len());
    out.extend_from_slice(&BITMAP_MAGIC);
    for v in [ds.width, ds.height, ds.x, ds.y, ds.video_width, ds.video_height] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    out.extend_from_slice(&ds.rgba);
    out
}

/// Decode a `subtitle/bitmap` wire payload back to a [`DisplaySet`]. `None` on a bad magic, a
/// truncated header, or an RGBA slice that doesn't match `width*height*4` (a malformed peer
/// buffer is dropped, never trusted — spec: untrusted input).
pub fn decode_bitmap(payload: &[u8]) -> Option<DisplaySet> {
    if payload.len() < BITMAP_HEADER || payload[..4] != BITMAP_MAGIC {
        return None;
    }
    let rd = |o: usize| u32::from_be_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]]);
    let width = rd(4);
    let height = rd(8);
    let x = rd(12);
    let y = rd(16);
    let video_width = rd(20);
    let video_height = rd(24);
    let want = (width as usize).checked_mul(height as usize)?.checked_mul(4)?;
    let rgba = payload.get(BITMAP_HEADER..)?;
    if rgba.len() != want {
        return None;
    }
    Some(DisplaySet {
        rgba: rgba.to_vec(),
        width,
        height,
        x,
        y,
        video_width,
        video_height,
    })
}

/// A minimal bounds-checked forward cursor over segment payload bytes — every read is fallible
/// so a truncated payload errors rather than panicking (spec: untrusted input is a P0).
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, PgsError> {
        let b = *self.data.get(self.at).ok_or(PgsError::Truncated(what))?;
        self.at += 1;
        Ok(b)
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, PgsError> {
        let hi = self.u8(what)? as u16;
        let lo = self.u8(what)? as u16;
        Ok((hi << 8) | lo)
    }

    fn skip(&mut self, n: usize, what: &'static str) -> Result<(), PgsError> {
        let end = self.at.checked_add(n).ok_or(PgsError::Malformed(what))?;
        if end > self.data.len() {
            return Err(PgsError::Truncated(what));
        }
        self.at = end;
        Ok(())
    }

    /// The remaining unread bytes (the RLE tail of an ODS fragment).
    fn rest(&self) -> &'a [u8] {
        &self.data[self.at.min(self.data.len())..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BT.709 conversion sanity: neutral chroma (128) maps grey; the two range endpoints
    /// map to black and white.
    #[test]
    fn ycrcb_bt709_endpoints() {
        // Black (Y=16, neutral chroma) → ~0; white (Y=235) → ~255.
        assert_eq!(ycrcb_to_rgb_bt709(16, 128, 128), [0, 0, 0]);
        let w = ycrcb_to_rgb_bt709(235, 128, 128);
        assert!(w.iter().all(|&c| c >= 254), "white ≈ 255: {w:?}");
        // A grey (mid Y, neutral chroma) is achromatic (R≈G≈B).
        let g = ycrcb_to_rgb_bt709(126, 128, 128);
        assert!(g[0] == g[1] && g[1] == g[2], "neutral chroma is achromatic: {g:?}");
    }

    /// The RLE grammar, run form by run form, on a tiny 8×2 bitmap.
    #[test]
    fn rle_decodes_each_run_form() {
        // Row 0: 4 px colour 0 (short zero run), then 4 px colour 5 (short colour run), EOL.
        // Row 1: one literal pixel (colour 7), then 63-cap short zero run clamps to row end.
        let rle = [
            0x00, 0x04, // 00b: 4 px of colour 0
            0x00, 0x84, 0x05, // 10b: 4 px of colour 5
            0x00, 0x00, // EOL
            0x07, // literal 1 px colour 7
            0x00, 0x0A, // 10 px colour 0 — clamps to the 7 remaining of an 8-wide row
            0x00, 0x00, // EOL
        ];
        let idx = decode_rle(&rle, 8, 2).expect("rle");
        assert_eq!(&idx[0..8], &[0, 0, 0, 0, 5, 5, 5, 5], "row 0: zeros then fives");
        assert_eq!(idx[8], 7, "row 1 first pixel is the literal 7");
        assert_eq!(&idx[9..16], &[0, 0, 0, 0, 0, 0, 0], "row 1 rest zero (clamped run)");
    }

    /// A long zero run (form 01) and a long colour run (form 11) reach past 63 pixels.
    #[test]
    fn rle_long_runs() {
        // 100 px of colour 0 across a 100-wide, 1-tall row: form 01, length 100 = 0x64.
        let rle = [0x00, 0x40, 0x64, 0x00, 0x00];
        let idx = decode_rle(&rle, 100, 1).expect("rle");
        assert!(idx.iter().all(|&c| c == 0));
        assert_eq!(idx.len(), 100);

        // 100 px of colour 9: form 11, length 0x0064, colour 9.
        let rle = [0x00, 0xC0, 0x64, 0x09, 0x00, 0x00];
        let idx = decode_rle(&rle, 100, 1).expect("rle");
        assert!(idx.iter().all(|&c| c == 9));
    }

    /// A truncated run errors rather than panicking (spec: untrusted input).
    #[test]
    fn rle_truncation_errors() {
        // Escape then a form-10 run promising a colour byte that isn't there.
        assert!(decode_rle(&[0x00, 0x83], 8, 1).is_err());
        // Escape then a form-01 run missing its second length byte.
        assert!(decode_rle(&[0x00, 0x40], 8, 1).is_err());
    }

    /// A hand-built Display Set — one PDS (two entries) + one ODS (a 2×1 RLE of two colours)
    /// → the expected RGBA at the composed position. The golden end-to-end path.
    #[test]
    fn display_set_decodes_to_expected_rgba() {
        // Palette: index 1 = opaque white (Y=235, neutral chroma, A=255),
        //          index 2 = opaque red-ish (Y=82, Cr=240, Cb=90 ≈ red, A=255).
        let pds = {
            let mut p = vec![
                SEG_PDS, 0x00, 0x00, // type, size placeholder
            ];
            let body = [
                0x00, 0x00, // palette_id, version
                0x01, 235, 128, 128, 255, // idx1 white
                0x02, 82, 240, 90, 255, // idx2 red
            ];
            let len = body.len() as u16;
            p[1..3].copy_from_slice(&len.to_be_bytes());
            p.extend_from_slice(&body);
            p
        };
        // ODS: object 0, first+last (0xC0), a 2×1 bitmap. RLE: one literal px colour 1, one
        // literal px colour 2, EOL.
        let ods = {
            let rle = [0x01u8, 0x02, 0x00, 0x00];
            let mut body = vec![
                0x00, 0x00, // object_id
                0x00, // version
                0xC0, // first+last
            ];
            let data_len = (4 + rle.len()) as u32; // 4 = width+height bytes
            body.extend_from_slice(&data_len.to_be_bytes()[1..]); // u24
            body.extend_from_slice(&2u16.to_be_bytes()); // width
            body.extend_from_slice(&1u16.to_be_bytes()); // height
            body.extend_from_slice(&rle);
            let mut o = vec![SEG_ODS, 0x00, 0x00];
            let len = body.len() as u16;
            o[1..3].copy_from_slice(&len.to_be_bytes());
            o.extend_from_slice(&body);
            o
        };
        // PCS: 1920×1080 reference, one object (id 0) at (100, 200).
        let pcs = {
            let mut body = vec![];
            body.extend_from_slice(&1920u16.to_be_bytes()); // width
            body.extend_from_slice(&1080u16.to_be_bytes()); // height
            body.push(0x10); // frame_rate
            body.extend_from_slice(&0u16.to_be_bytes()); // composition_number
            body.push(0x80); // composition_state (epoch start)
            body.push(0x00); // palette_update_flag
            body.push(0x00); // palette_id
            body.push(0x01); // number_of_composition_objects
            body.extend_from_slice(&0u16.to_be_bytes()); // object_id
            body.push(0x00); // window_id
            body.push(0x00); // object_cropped_flag
            body.extend_from_slice(&100u16.to_be_bytes()); // x
            body.extend_from_slice(&200u16.to_be_bytes()); // y
            let mut p = vec![SEG_PCS, 0x00, 0x00];
            let len = body.len() as u16;
            p[1..3].copy_from_slice(&len.to_be_bytes());
            p.extend_from_slice(&body);
            p
        };
        let end = [SEG_END, 0x00, 0x00];

        let mut ds = Vec::new();
        ds.extend_from_slice(&pcs);
        ds.extend_from_slice(&pds);
        ds.extend_from_slice(&ods);
        ds.extend_from_slice(&end);

        let out = parse_display_set(&ds).expect("display set decodes");
        assert_eq!((out.width, out.height), (2, 1));
        assert_eq!((out.x, out.y), (100, 200));
        assert_eq!((out.video_width, out.video_height), (1920, 1080));
        assert!(!out.is_clear());
        // Pixel 0 is white (idx1), pixel 1 is reddish (idx2, opaque).
        assert_eq!(&out.rgba[0..4], &[255, 255, 255, 255], "white pixel");
        assert_eq!(out.rgba[7], 255, "second pixel opaque");
        assert!(out.rgba[4] > out.rgba[5] && out.rgba[4] > out.rgba[6], "second pixel red-dominant");
    }

    /// A composition with zero objects is a clear (erase the caption).
    #[test]
    fn empty_composition_is_a_clear() {
        let pcs = {
            let mut body = vec![];
            body.extend_from_slice(&1920u16.to_be_bytes());
            body.extend_from_slice(&1080u16.to_be_bytes());
            body.push(0x10);
            body.extend_from_slice(&1u16.to_be_bytes());
            body.push(0x00); // normal state
            body.push(0x00);
            body.push(0x00);
            body.push(0x00); // zero composition objects
            let mut p = vec![SEG_PCS, 0x00, 0x00];
            let len = body.len() as u16;
            p[1..3].copy_from_slice(&len.to_be_bytes());
            p.extend_from_slice(&body);
            p
        };
        let mut ds = pcs;
        ds.extend_from_slice(&[SEG_END, 0x00, 0x00]);
        let out = parse_display_set(&ds).expect("clear parses");
        assert!(out.is_clear(), "no objects → clear");
        assert_eq!((out.video_width, out.video_height), (1920, 1080));
    }

    /// A Display Set with no PCS errors (nothing to compose).
    #[test]
    fn missing_pcs_errors() {
        let ds = [SEG_END, 0x00, 0x00];
        assert_eq!(parse_display_set(&ds), Err(PgsError::NoComposition));
    }

    /// Truncated segment headers error, never panic (spec: untrusted input is a P0).
    #[test]
    fn truncated_segments_error() {
        assert!(parse_display_set(&[SEG_PCS, 0x00]).is_err(), "truncated size field");
        // A PCS claiming 200 payload bytes but with none present.
        assert!(parse_display_set(&[SEG_PCS, 0x00, 0xC8]).is_err(), "payload runs past end");
    }

    /// The `subtitle/bitmap` wire format round-trips (header + RGBA), including a clear.
    #[test]
    fn bitmap_wire_round_trips() {
        let ds = DisplaySet {
            rgba: vec![1, 2, 3, 4, 5, 6, 7, 8],
            width: 2,
            height: 1,
            x: 100,
            y: 200,
            video_width: 1920,
            video_height: 1080,
        };
        let wire = encode_bitmap(&ds);
        assert_eq!(wire.len(), BITMAP_HEADER + 8);
        assert_eq!(decode_bitmap(&wire), Some(ds));

        // A clear round-trips as an empty-RGBA header.
        let clear = DisplaySet::clear(1920, 1080);
        let wire = encode_bitmap(&clear);
        assert_eq!(wire.len(), BITMAP_HEADER);
        let back = decode_bitmap(&wire).expect("clear decodes");
        assert!(back.is_clear());

        // A bad magic / truncated payload is rejected.
        assert!(decode_bitmap(b"XXXX").is_none());
        assert!(decode_bitmap(&wire[..4]).is_none());
        // A header whose RGBA length disagrees with width*height*4 is rejected.
        let mut bad = encode_bitmap(&DisplaySet {
            rgba: vec![0; 8],
            width: 2,
            height: 1,
            x: 0,
            y: 0,
            video_width: 1,
            video_height: 1,
        });
        bad.pop(); // now the RGBA is one byte short
        assert!(decode_bitmap(&bad).is_none());
    }

    /// The `.sup` header peel round-trips: a `PG`-framed record yields its bare segment + PTS.
    #[test]
    fn strip_sup_header_peels_the_pg_frame() {
        // PG + pts=0x000C7C68 + dts=0 + segment(END, len 0).
        let record = [
            0x50, 0x47, // PG
            0x00, 0x0C, 0x7C, 0x68, // pts
            0x00, 0x00, 0x00, 0x00, // dts
            SEG_END, 0x00, 0x00, // segment
        ];
        let (pts, seg) = strip_sup_header(&record).expect("peels");
        assert_eq!(pts, 0x000C_7C68);
        assert_eq!(seg, &[SEG_END, 0x00, 0x00]);
        // A wrong magic is rejected.
        assert!(strip_sup_header(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).is_none());
    }
}
