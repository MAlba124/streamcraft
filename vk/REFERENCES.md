# References

Primary sources for every non-trivial algorithm in this crate (project rule: the
implementation is reviewed *against* these texts; no other implementation —
libplacebo, mpv, or otherwise — is consulted. Clean-room).

- **Rec. ITU-R BT.601-7** (03/2011), *Studio encoding parameters of digital
  television* — §2.5.1 (Kr = 0.299, Kb = 0.114), §3.5 (8-bit quantization,
  Y' 16..235, Cb/Cr 128 ± 112). The integer matrix in `shaders/yuv2rgb.comp`
  derives these exactly as `wayland/src/convert.rs` does on the CPU; the two paths
  are asserted equal in `tests/parity.rs`.
- **GLSL 4.50 specification** §5.9 — `>>` on a signed operand is an arithmetic
  shift, matching Rust `i32 >>`; load-bearing for the CPU/GPU parity claim.
- **Vulkan 1.1 specification** (Khronos) — core API usage; external memory
  capability (promoted from VK_KHR_external_memory).
- **VK_KHR_external_memory_fd**, **VK_EXT_external_memory_dma_buf** (Khronos
  extension specs) — exportable allocations (`VkExportMemoryAllocateInfo`) and
  `vkGetMemoryFdKHR`.
- **linux-dmabuf-unstable-v1** Wayland protocol — vendored at
  `wayland/spec/linux-dmabuf-unstable-v1.xml`; the import path lives in
  `sc-wayland`'s client.
- **Linux `drm_fourcc.h`** (kernel UAPI) — `DRM_FORMAT_XRGB8888` ('XR24'),
  `DRM_FORMAT_MOD_LINEAR`.

## Cited follow-ups (not yet implemented)

- Scaling: C. E. Duchon (1979), *Lanczos Filtering in One and Two Dimensions*,
  J. Appl. Meteor. 18; D. P. Mitchell & A. N. Netravali (1988), *Reconstruction
  Filters in Computer Graphics*, SIGGRAPH '88.
- Dithering (for 10-bit → 8-bit output): B. E. Bayer (1973), *An optimum method
  for two-level rendition of continuous-tone pictures*; R. A. Ulichney (1993),
  *Void-and-cluster method for dither array generation*.
- HDR tone mapping: Rec. ITU-R BT.2100 (PQ/HLG), SMPTE ST 2084, Rep. ITU-R
  BT.2390 (EOTF handling and tone-mapping guidance).
- Zero-stall sync: VK_KHR_external_semaphore_fd + the Wayland
  linux-explicit-synchronization protocol.
