{
  description = "Profluens — ultra light weight data/multimedia streaming/processing graph framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Pinned Rust toolchain with the components used across the workspace
        # (clippy, rustfmt, rust-analyzer, and the src for tooling/loom/miri work).
        # NIGHTLY: the adopted codecs gate their portable_simd paths on nightly
        # (oxideav-vp8 `simd` uses std::simd in the transforms + loop filter;
        # vendored oxideav-h264 has a `nightly` feature for the same). The date is
        # pinned by the rust-overlay input in flake.lock — still reproducible.
        rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain
            # Native tooling. Core is dependency-free; pkg-config + the dev libs
            # below are for device elements (PipeWire, ALSA, V4L2, …) as they land.
            pkgs.pkg-config
            pkgs.cargo-nextest
            pkgs.cargo-fuzz
            pkgs.mold
            pkgs.clang # libclang, for the pipewire crate's bindgen
            pkgs.libva-utils # vainfo, for VA-API driver debugging
            pkgs.glslang # glslangValidator, for the offline SPIR-V shader bake
          ];

          # PipeWire (libpipewire-0.3 + libspa), SDL3, and libva (+ libva-drm),
          # found via pkg-config. Only pf-pipewire links libpipewire, only pf-sdl3
          # links SDL3, and only pf-vaapi links libva (spec: a device backend is
          # the one "buy, don't build"); the core stays dependency-free.
          buildInputs = [ pkgs.pipewire pkgs.sdl3 pkgs.libva ];

          # Self-contained linking: the stdenv `cc` driver with mold, so the shell
          # doesn't depend on a globally-configured linker. Plain RUSTFLAGS is used
          # deliberately — a target-specific `CARGO_TARGET_*_RUSTFLAGS` does NOT
          # override rustflags set in ~/.cargo/config.toml, but plain RUSTFLAGS does.
          CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = "cc";
          RUSTFLAGS = "-C link-arg=-fuse-ld=mold";

          # bindgen (pipewire-sys / libspa-sys) needs libclang.
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

          # So the dynamically-linked libSDL3.so.0 / libva.so resolve at run time
          # inside the pure devshell (not on the default search path here).
          LD_LIBRARY_PATH = "${pkgs.sdl3}/lib:${pkgs.libva}/lib";

          # VA-API driver discovery must be self-contained too (the host is not
          # NixOS): iHD for Intel, Mesa's for AMD. The runtime picks by probing.
          LIBVA_DRIVERS_PATH = "${pkgs.intel-media-driver}/lib/dri:${pkgs.mesa}/lib/dri";

          shellHook = ''
            echo "profluens devshell — $(rustc --version)"
          '';
        };

        # `nix fmt`
        formatter = pkgs.nixpkgs-fmt;
      });
}
