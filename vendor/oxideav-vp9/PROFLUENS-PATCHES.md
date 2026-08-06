# profluens patches to oxideav-vp9 0.0.12

Vendored from crates.io `oxideav-vp9 0.0.12` (wired via `[patch.crates-io]`,
same arrangement as `vendor/oxideav-h264` / `vendor/oxideav-aac` /
`vendor/oxideav-vp8`). One behavioral patch, to be offered upstream:

## Frame edge: a transform block may legally overhang `CurrFrame`

`decode_intra_frame` panicked with `index out of bounds` at
`src/intra.rs` (`Plane::set`) on **any** frame whose MI-aligned height is not a
multiple of the transform size used on the bottom superblock row. Measured with
`libvpx-vp9` keyframe-only streams (`ffmpeg … -c:v libvpx-vp9 -g 1`):

| size      | `h % 32` | 0.0.12  |
|-----------|----------|---------|
| 320x240   | 16       | panic   |
| 640x360   | 8        | panic   |
| **1280x720**  | 16   | **panic** |
| 1920x1080 | 24       | panic   |
| 64x64, 80x64, 128x96, 160x128, 352x288, 640x480, 1920x1088 | 0 | ok |

720p and 1080p are the two most common HD resolutions, so in practice the
decoder crashed on most real VP9 content.

### What the spec says

§6.4.21 `residual( )` gates a transform block on its **top-left corner only**:

```
maxx = (MiCols * 8) >> subX
maxy = (MiRows * 8) >> subY
...
if ( startX < maxx && startY < maxy ) {
    if ( !is_inter )
        predict_intra( plane, startX, startY, … )
    if ( !skip ) {
        nonzero = tokens( plane, startX, startY, txSz, blockIdx )
        reconstruct( plane, startX, startY, txSz )
    }
}
```

A block whose top-left is in range therefore *legally extends past the frame*
by up to `size - 1` samples. Both store steps are written against a `CurrFrame`
the spec never bounds, and neither is clamped:

* §8.5.1 (Intra prediction process), final step — "The current frame is updated
  as follows: `CurrFrame[ plane ][ y + i ][ x + j ]` is set equal to
  `pred[ i ][ j ]` for `i = 0..size-1` and `j = 0..size-1`."
* §8.6.2 (Reconstruct process), step 4 — "`CurrFrame[ plane ][ y + i ][ x + j ]`
  is set equal to `Clip1( CurrFrame[ plane ][ y + i ][ x + j ] + Dequant[ i ][ j ] )`
  for `i = 0..(n0-1)` and `j = 0..(n0-1)`."

Note the asymmetry that makes this safe to clamp: §8.5.1's *reads* are clamped
— `aboveRow[ i ] = CurrFrame[ plane ][ y-1 ][ Min(maxX, x+i) ]`,
`leftCol[ i ] = CurrFrame[ plane ][ Min(maxY, y+i) ][ x-1 ]` — so **nothing ever
reads a sample past `maxX` / `maxY`**. The overhang is write-only scratch.
libvpx absorbs it in the frame buffer's `VP9BORDERINPIXELS` border; the spec
simply leaves `CurrFrame` unbounded.

### The bug

Not a missing read clamp and not a wrong stride: it is an **allocation sized to
the un-padded extent**. `src/decode_frame.rs` allocates

```rust
let y_w = (mi_cols * 8) as usize;
let y_h = (mi_rows * 8) as usize;
let mut plane_y = Plane::new(y_w, y_h);
```

i.e. exactly `(maxX + 1) x (maxY + 1)` — zero room for the overhang the two
store steps produce. `src/residual.rs` then correctly passes
`max_x = maxx - 1` / `max_y = maxy - 1` (so the neighbour reads clamp fine), but
`predict_intra` and `reconstruct_block` write the full `size x size` block
unconditionally. Consequences:

- **Vertical overhang** (`MiRows * 8` not a multiple of `size`): the flat index
  `y * width + x` runs past the buffer → panic. The reported index is always
  exactly `len` — row `MiRows * 8`, column 0, the first sample of the first
  out-of-range row.
- **Horizontal overhang** (`MiCols * 8` not a multiple of `size`): the flat
  index stays in bounds but **wraps into the next row**, silently corrupting
  visible samples. This one never panicked and was never noticed.

### The patch

Both sites marked `PROFLUENS PATCH`; the fix is to bound the *store* loops (not
the prediction/transform maths) by the plane's own extents, which are exactly
`(maxX + 1, maxY + 1)`:

- `src/intra.rs` — `predict_intra`'s final store loop runs to
  `size.min(plane.height() - y)` / `size.min(plane.width() - x)`. `pred` is
  still built over the full `size x size`, because the directional recurrences
  (D207 step 5, D117 step 6, D135 step 5, D153 step 7) read cells that the
  clamped store never emits.
- `src/reconstruct.rs` — `reconstruct_block`'s step-4 add loop gets the same
  bound. Step 4 is element-wise, so skipping out-of-range positions cannot
  perturb an in-range one. The inverse transform still runs over the full
  `n0 x n0`.
- `Plane::set`'s doc comment records the new caller contract.

### Why this is bit-exact for unaffected sizes

Nothing reads the dropped samples: §8.5.1's neighbour fetches clamp to
`maxX` / `maxY`, §8.6.2 step 4 touches only positions it also writes, the §8.8
loop filter is driven with the §8.10 visible extents (`crop_w` / `crop_h` in
`decode_frame.rs`), and §8.10 crops the output to `FrameWidth x FrameHeight`.
Verified empirically: the packed planar output of `decode_intra_frame` is
**byte-identical before and after the patch** (sha256, 2 frames each) for
64x32, 64x48, 64x64, 80x64, 128x96, 160x128, 352x288, 640x480 and 1920x1088,
i.e. every size the unpatched decoder could decode at all.

Correctness of the newly decodable sizes is checked against ffmpeg's VP9
decoder as an oracle in `vp9/tests/vp9dec.rs`.
