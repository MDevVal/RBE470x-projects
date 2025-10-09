let
  pkgs = import <nixpkgs> {};
in
pkgs.mkShell {
  buildInputs = [
    pkgs.python313
    pkgs.python313Packages.pygame
    pkgs.python313Packages.numpy
    pkgs.python313Packages.colorama
    pkgs.python313Packages.torch
    pkgs.SDL2
    pkgs.SDL2_image
    pkgs.libpng
    pkgs.zlib
  ];
}

