# diffusers-burn

Stable Diffusion in Rust using [Burn](https://github.com/burn-rs/burn). Supports SD 1.5 and 2.1.

Based on [diffusers-rs](https://github.com/LaurentMazare/diffusers-rs).

<div align="left" valign="middle">
<a href="https://runblaze.dev">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://www.runblaze.dev/logo_dark.png">
   <img align="right" src="https://www.runblaze.dev/logo_light.png" height="102px"/>
 </picture>
</a>

<br style="display: none;"/>

_[Blaze](https://runblaze.dev) supports this project by providing ultra-fast Apple Silicon macOS Github Action Runners. Apply the discount code `AI25` at checkout to enjoy 25% off your first year._

</div>

## Quick Start

```bash
# Using wgpu backend (default, works on most GPUs)
cargo run --release --features wgpu --example stable-diffusion -- \
  --prompt "A photo of a rusty robot on a beach"

# Using torch backend
cargo run --release --features torch --example stable-diffusion -- \
  --prompt "A photo of a rusty robot on a beach"

# SD 2.1 at 768x768
cargo run --release --features torch --example stable-diffusion -- \
  --sd-version v2-1 \
  --prompt "A majestic lion on a cliff at sunset"
```

## Backends

| Feature | Backend | Notes |
|---------|---------|-------|
| `wgpu` | WebGPU | Cross-platform GPU support |
| `torch` | LibTorch | Requires libtorch |
| `ndarray` | ndarray | CPU only, pure Rust |

## no_std Support

This crate supports `#![no_std]` with `alloc` by disabling the default `std` feature.

## Community

Join our [Discord](https://discord.gg/UHtSgF6j5J) if you want to contribute or have questions!

## License

MIT or Apache-2.0, at your option.
