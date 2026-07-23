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
            # below are for device elements (ALSA, V4L2, …) as they land.
            pkgs.pkg-config
            pkgs.cargo-nextest
            pkgs.cargo-fuzz
            pkgs.mold
          ];

          # Self-contained linking: the stdenv `cc` driver with mold, so the shell
          # doesn't depend on a globally-configured linker. Plain RUSTFLAGS is used
          # deliberately — a target-specific `CARGO_TARGET_*_RUSTFLAGS` does NOT
          # override rustflags set in ~/.cargo/config.toml, but plain RUSTFLAGS does.
          CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = "cc";
          RUSTFLAGS = "-C link-arg=-fuse-ld=mold";

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

          shellHook = ''
            echo "streamcraft devshell — $(rustc --version)"
          '';
        };

        # `nix fmt`
        formatter = pkgs.nixpkgs-fmt;
      });
}
