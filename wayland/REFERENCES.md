# References

Primary sources for every non-trivial algorithm in this crate (project rule: the
implementation is reviewed *against* these texts; no other implementation is
consulted — clean-room).

- **Rec. ITU-R BT.601-7** (03/2011), *Studio encoding parameters of digital
  television for standard 4:3 and wide-screen 16:9 aspect ratios*.
  - §2.5.1: luma/colour-difference weights (Kr = 0.299, Kb = 0.114).
  - §3.5: 8-bit quantization levels (Y' 16..235; Cb/Cr 128 ± 112) — the
    studio-swing range folded into `src/convert.rs`'s integer matrix.
- **Wayland protocol specification** — vendored as `spec/wayland.xml`
  (wire format, core interfaces; cited per `interface.request/event` in
  `src/protocol.rs`/`src/wire.rs`).
- **xdg-shell protocol** — vendored as `spec/xdg-shell.xml` (window management;
  cited likewise).
