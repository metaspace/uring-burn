{ nixpkgs, system, rust-overlay, crane }:
let
  pkgs = import nixpkgs {
    inherit system;
    overlays = [ (import rust-overlay) ];
  };
  craneLib =
    (crane.mkLib pkgs).overrideToolchain pkgs.rust-bin.stable.latest.minimal;
in craneLib.buildPackage {
  name = "uring-burn";
  src = ./.;
}
