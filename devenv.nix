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
    pkgs.pixi
  ];

  env.LIBTORCH = "${libtorch}";
  env.DYLD_LIBRARY_PATH = "${libtorch}/lib";

  languages.rust.enable = true;
  languages.rust.channel = "stable";

  enterShell = ''
    echo "diffusers-burn dev environment"
    rustc --version
    cargo --version

    # Python via pixi (conda-forge torch, unset DYLD_LIBRARY_PATH to avoid Rust libtorch conflict)
    if [ -d .pixi/envs/default ]; then
      export PATH="$PWD/.pixi/envs/default/bin:$PATH"
      python --version 2>/dev/null
    else
      echo "Python env not set up. Run: pixi install"
    fi

    # Wrapper to run Python scripts without DYLD_LIBRARY_PATH interference
    py() { env -u DYLD_LIBRARY_PATH python "$@"; }
  '';
}
