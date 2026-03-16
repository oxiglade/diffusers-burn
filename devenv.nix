{ pkgs, lib, config, inputs, ... }:

let
  libtorch = pkgs.fetchzip {
    url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.9.0.zip";
    hash = "sha256-inwzvGPPKK6KVBoMijpXVKV+V3QcmQbBhYdFlZbZ/ho=";
  };
in
{
  packages = [
    pkgs.git
  ];

  env.LIBTORCH = "${libtorch}";
  env.DYLD_LIBRARY_PATH = "${libtorch}/lib";

  languages.rust.enable = true;
  languages.rust.channel = "stable";

  enterShell = ''
    echo "diffusers-burn dev environment"
    rustc --version
    cargo --version
  '';
}
