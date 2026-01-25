//! Weight Loading Utilities
//!
//! This module provides utilities for loading pre-trained weights from
//! SafeTensors format into Burn models.
//!
//! # Overview
//!
//! Stable Diffusion models typically come in SafeTensors format from Hugging Face.
//! This module provides helper functions to load these weights into the
//! diffusers-burn model structures.
//!
//! # Weight File Structure
//!
//! For Stable Diffusion, you typically need:
//! - `text_encoder/model.safetensors` - CLIP text encoder weights
//! - `vae/diffusion_pytorch_model.safetensors` - VAE encoder/decoder weights
//! - `unet/diffusion_pytorch_model.safetensors` - UNet denoising model weights
//!
//! # Example Usage
//!
//! ```ignore
//! use diffusers_burn::pipelines::weights::load_safetensors;
//! use diffusers_burn::pipelines::stable_diffusion::StableDiffusionConfig;
//!
//! let config = StableDiffusionConfig::v1_5(None, None, None);
//! let device = Default::default();
//!
//! // Initialize models
//! let mut pipeline = config.init::<MyBackend>(&device);
//!
//! // Load weights
//! pipeline.clip = load_safetensors(
//!     pipeline.clip,
//!     "path/to/text_encoder/model.safetensors",
//!     &device,
//! )?;
//!
//! pipeline.vae = load_safetensors(
//!     pipeline.vae,
//!     "path/to/vae/diffusion_pytorch_model.safetensors",
//!     &device,
//! )?;
//!
//! pipeline.unet = load_safetensors(
//!     pipeline.unet,
//!     "path/to/unet/diffusion_pytorch_model.safetensors",
//!     &device,
//! )?;
//! ```

use std::path::Path;

use burn::module::Module;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

/// Errors that can occur during weight loading.
#[derive(Debug, thiserror::Error)]
pub enum WeightLoadError {
    #[error("Failed to load safetensors file: {0}")]
    SafetensorsLoad(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load weights from a SafeTensors file into a Burn module.
///
/// This function uses the PyTorch adapter by default, which handles:
/// - Transposing linear layer weights (PyTorch uses [out, in], Burn uses [in, out])
/// - Renaming normalization parameters (weight->gamma, bias->beta)
///
/// # Arguments
/// * `module` - The Burn module to load weights into
/// * `path` - Path to the SafeTensors file
/// * `device` - Device to load the weights onto
///
/// # Returns
/// The module with loaded weights, or an error if loading fails.
///
/// # Example
/// ```ignore
/// let clip = config.build_clip_transformer::<Backend>(&device);
/// let clip = load_safetensors(clip, "model.safetensors", &device)?;
/// ```
pub fn load_safetensors<B, M, P>(
    module: M,
    path: P,
    device: &B::Device,
) -> Result<M, WeightLoadError>
where
    B: Backend,
    M: Module<B>,
    P: AsRef<Path>,
{
    let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let args = LoadArgs::new(path.as_ref().to_path_buf());

    let record = recorder
        .load(args, device)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    Ok(module.load_record(record))
}

/// Load weights with debug output to see key mappings.
///
/// This is useful when debugging weight loading issues to see:
/// - Original keys from the SafeTensors file
/// - How keys are mapped to Burn's naming convention
///
/// # Arguments
/// * `module` - The Burn module to load weights into
/// * `path` - Path to the SafeTensors file
/// * `device` - Device to load the weights onto
///
/// # Returns
/// The module with loaded weights, or an error if loading fails.
pub fn load_safetensors_debug<B, M, P>(
    module: M,
    path: P,
    device: &B::Device,
) -> Result<M, WeightLoadError>
where
    B: Backend,
    M: Module<B>,
    P: AsRef<Path>,
{
    let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let args = LoadArgs::new(path.as_ref().to_path_buf()).with_debug_print();

    let record = recorder
        .load(args, device)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    Ok(module.load_record(record))
}

/// Instructions for downloading Stable Diffusion weights.
///
/// Returns a string with instructions for obtaining the necessary weight files.
pub fn download_instructions() -> &'static str {
    r#"
# Downloading Stable Diffusion Weights

## Option 1: From Hugging Face Hub (Recommended)

1. Install huggingface-cli:
   pip install huggingface_hub

2. Download SD 1.5 weights:
   huggingface-cli download runwayml/stable-diffusion-v1-5 \
     text_encoder/model.safetensors \
     vae/diffusion_pytorch_model.safetensors \
     unet/diffusion_pytorch_model.safetensors \
     --local-dir ./sd-v1-5

3. Or download SD 2.1 weights:
   huggingface-cli download stabilityai/stable-diffusion-2-1 \
     text_encoder/model.safetensors \
     vae/diffusion_pytorch_model.safetensors \
     unet/diffusion_pytorch_model.safetensors \
     --local-dir ./sd-v2-1

## Option 2: Manual Download

Visit the model pages on Hugging Face:
- SD 1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5
- SD 2.1: https://huggingface.co/stabilityai/stable-diffusion-2-1

Download the .safetensors files from the "Files" tab.

## BPE Vocabulary File

You also need the BPE vocabulary file for the tokenizer:
   wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
   gunzip bpe_simple_vocab_16e6.txt.gz

## Directory Structure

Your weights directory should look like:
   model_dir/
   ├── text_encoder/
   │   └── model.safetensors
   ├── vae/
   │   └── diffusion_pytorch_model.safetensors
   ├── unet/
   │   └── diffusion_pytorch_model.safetensors
   └── bpe_simple_vocab_16e6.txt
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_instructions() {
        let instructions = download_instructions();
        assert!(instructions.contains("Hugging Face"));
        assert!(instructions.contains("safetensors"));
    }
}
