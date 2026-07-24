{
  description = "StreamCraft — ultra light weight data/multimedia streaming/processing graph framework";

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
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
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
            # `ash`'s "loaded" feature dlopen()s libvulkan.so.1 at runtime for
            # the sc-vk renderer; glslang regenerates the checked-in SPIR-V from
            # the reviewable GLSL sources (vk/shaders/*.glsl → vk/src/shaders.rs).
            # build.rs never runs a shader compiler — regeneration is a documented
            # manual step (see vk/REFERENCES.md).
            pkgs.vulkan-loader
            pkgs.glslang
          ];

          # PipeWire (libpipewire-0.3 + libspa), found via pkg-config. Only the
          # sc-pipewire plugin links it; the core stays dependency-free.
          buildInputs = [ pkgs.pipewire ];

          # Self-contained linking: the stdenv `cc` driver with mold, so the shell
          # doesn't depend on a globally-configured linker. Plain RUSTFLAGS is used
          # deliberately — a target-specific `CARGO_TARGET_*_RUSTFLAGS` does NOT
          # override rustflags set in ~/.cargo/config.toml, but plain RUSTFLAGS does.
          CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = "cc";
          RUSTFLAGS = "-C link-arg=-fuse-ld=mold";

          # bindgen (pipewire-sys / libspa-sys) needs libclang.
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

          # So `ash`'s runtime dlopen("libvulkan.so.1") resolves inside the pure
          # devshell (the loader is not on the default linker search path here).
          # The loader then discovers the system ICDs via the usual
          # /run/opengl-driver + /usr/share/vulkan/icd.d manifests.
          LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";

          shellHook = ''
            echo "streamcraft devshell — $(rustc --version)"
          '';
        };

        # `nix fmt`
        formatter = pkgs.nixpkgs-fmt;
      });
}
