//! # Diffusion pipelines and models
//!
//! This is a Rust port of Hugging Face's [diffusers](https://github.com/huggingface/diffusers) Python api using [Burn](https://github.com/burn-rs/burn)

#![cfg_attr(not(feature = "std"), no_std)]

pub mod models;
pub mod pipelines;
pub mod transformers;
pub mod utils;

extern crate alloc;

#[cfg(all(test, feature = "ndarray"))]
pub type TestBackend = burn::backend::NdArray<f32>;

#[cfg(all(test, feature = "torch"))]
pub type TestBackend = burn::backend::LibTorch<f32>;

#[cfg(all(test, feature = "wgpu"))]
pub type TestBackend = burn::backend::Wgpu;
