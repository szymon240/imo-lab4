{ pkgs ? import <nixpkgs> {} }:
let
  fetchurl = pkgs.fetchurl;
  packageOverrides = pkgs.callPackage ./python-packages.nix { inherit pkgs fetchurl; };
  pythonCustom = pkgs.python3.override { inherit packageOverrides; };
in
pkgs.mkShell {
  packages = with pkgs; [
    python3
    python3Packages.numpy_1
    python3Packages.matplotlib
    (pythonCustom.withPackages(p: [ p.tsplib95 ]))
  ];
}