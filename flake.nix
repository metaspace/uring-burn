{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = { nixpkgs.follows = "nixpkgs"; };
    };
    llm-agents.url = "github:numtide/llm-agents.nix";
  };

  outputs = inputs:
    with inputs;
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        package =
          pkgs.callPackage ./. { inherit nixpkgs system crane rust-overlay; };
      in {
        packages.default = package;
        devShells.default = let
          overlays = [ (import inputs.rust-overlay) ];
          pkgs = import inputs.nixpkgs { inherit overlays system; };
        in pkgs.mkShell {
          packages = [
            pkgs.rust-bin.stable.latest.complete
            inputs.llm-agents.packages.${system}.claude-code
          ];
        };
      });
}
