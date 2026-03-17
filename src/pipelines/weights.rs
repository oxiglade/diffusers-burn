//! Weight Loading Utilities
//!
//! This module provides utilities for loading pre-trained weights from
//! SafeTensors format into Burn models using `burn-store`.
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
//! use diffusers_burn::pipelines::weights::{load_clip_safetensors, load_vae_safetensors, load_unet_safetensors};
//! use diffusers_burn::pipelines::stable_diffusion::StableDiffusionConfig;
//!
//! let config = StableDiffusionConfig::v1_5(None, None, None);
//! let device = Default::default();
//!
//! // Initialize models
//! let clip = config.build_clip_transformer::<Backend>(&device);
//! let vae = config.build_vae::<Backend>(&device);
//! let unet = config.build_unet::<Backend>(&device, 4);
//!
//! // Load weights
//! let clip = load_clip_safetensors::<Backend, _, _>(clip, "path/to/model.safetensors", &device)?;
//! let vae = load_vae_safetensors::<Backend, _, _>(vae, "path/to/vae.safetensors", &device)?;
//! let unet = load_unet_safetensors::<Backend, _, _>(unet, "path/to/unet.safetensors", &device)?;
//! ```

use std::path::Path;

use burn::tensor::backend::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

/// Errors that can occur during weight loading.
#[derive(Debug, thiserror::Error)]
pub enum WeightLoadError {
    #[error("Failed to load safetensors file: {0}")]
    SafetensorsLoad(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Load CLIP text encoder weights from HuggingFace format.
///
/// This handles the key remapping from HuggingFace's naming convention
/// (e.g., `text_model.embeddings.token_embedding`) to Burn's convention
/// (e.g., `embeddings.token_embedding`).
///
/// Uses `PyTorchToBurnAdapter` to automatically handle:
/// - Transposing linear layer weights
/// - Renaming normalization parameters (weight->gamma, bias->beta)
pub fn load_clip_safetensors<B, M, P>(
    mut module: M,
    path: P,
    _device: &B::Device,
) -> Result<M, WeightLoadError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    P: AsRef<Path>,
{
    // Remap HuggingFace CLIP keys to our model structure
    let key_mappings: Vec<(&str, &str)> = vec![
        // Remove "text_model." prefix
        ("^text_model\\.", ""),
    ];

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    let checkpoint_path = path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    module
        .load_from(&mut store)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    Ok(module)
}

/// Load VAE weights from HuggingFace format.
///
/// This handles the key remapping from HuggingFace's naming convention to Burn's.
/// Main differences:
/// - HF uses `mid_block.resnets.0/1` but Burn uses `mid_block.resnet` (first) and
///   `mid_block.attn_resnets.0.resnet_block` (second)
/// - HF uses `mid_block.attentions.0` but Burn uses `mid_block.attn_resnets.0.attention_block`
/// - HF uses `downsamplers.0` but Burn uses `downsampler`
/// - HF uses `upsamplers.0` but Burn uses `upsampler`
///
/// Uses `PyTorchToBurnAdapter` to automatically handle:
/// - Transposing linear layer weights
/// - Renaming normalization parameters (weight->gamma, bias->beta)
pub fn load_vae_safetensors<B, M, P>(
    mut module: M,
    path: P,
    _device: &B::Device,
) -> Result<M, WeightLoadError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    P: AsRef<Path>,
{
    // Remap HuggingFace VAE keys to our model structure
    // Order matters: more specific patterns should come first
    // Note: VAE has encoder/decoder prefixes, so patterns should not use ^ anchor
    let key_mappings: Vec<(&str, &str)> = vec![
        // Mid block: first resnet (index 0) maps to standalone resnet field
        ("\\.mid_block\\.resnets\\.0\\.", ".mid_block.resnet."),
        // Mid block: second resnet (index 1) maps to attn_resnets.0.resnet_block
        (
            "\\.mid_block\\.resnets\\.1\\.",
            ".mid_block.attn_resnets.0.resnet_block.",
        ),
        // Mid block: attention maps to attn_resnets.X.attention_block
        (
            "\\.mid_block\\.attentions\\.(\\d+)\\.",
            ".mid_block.attn_resnets.$1.attention_block.",
        ),
        // VAE attention key remapping (SDXL VAE uses to_k/to_q/to_v/to_out.0)
        ("\\.to_k\\.", ".key."),
        ("\\.to_q\\.", ".query."),
        ("\\.to_v\\.", ".value."),
        ("\\.to_out\\.0\\.", ".proj_attn."),
        // Downsamplers: downsamplers.0 -> downsampler
        ("\\.downsamplers\\.0\\.", ".downsampler."),
        // Upsamplers: upsamplers.0 -> upsampler
        ("\\.upsamplers\\.0\\.", ".upsampler."),
    ];

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    let checkpoint_path = path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    module
        .load_from(&mut store)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    Ok(module)
}

/// Inspect a safetensors file to determine which down/up blocks have attention.
///
/// Returns two vectors of booleans indicating which blocks have attention:
/// - First vector: down_blocks (true if block has attention)
/// - Second vector: up_blocks (true if block has attention)
fn inspect_unet_block_types<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<bool>, Vec<bool>), WeightLoadError> {
    use std::collections::HashSet;

    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;
    let buffer = unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;
    let tensors = safetensors::SafeTensors::deserialize(&buffer)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    let keys: Vec<String> = tensors.names().into_iter().cloned().collect();

    // Find max block indices
    let mut max_down_block = 0usize;
    let mut max_up_block = 0usize;
    let mut down_blocks_with_attn = HashSet::new();
    let mut up_blocks_with_attn = HashSet::new();

    for key in keys.iter() {
        // Check for down_blocks.X pattern
        if let Some(rest) = key.strip_prefix("down_blocks.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                    max_down_block = max_down_block.max(idx);
                    // Check if this block has attentions
                    if rest[dot_pos..].starts_with(".attentions.") {
                        down_blocks_with_attn.insert(idx);
                    }
                }
            }
        }
        // Check for up_blocks.X pattern
        if let Some(rest) = key.strip_prefix("up_blocks.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                    max_up_block = max_up_block.max(idx);
                    // Check if this block has attentions
                    if rest[dot_pos..].starts_with(".attentions.") {
                        up_blocks_with_attn.insert(idx);
                    }
                }
            }
        }
    }

    // Build boolean vectors
    let down_has_attn: Vec<bool> = (0..=max_down_block)
        .map(|i| down_blocks_with_attn.contains(&i))
        .collect();
    let up_has_attn: Vec<bool> = (0..=max_up_block)
        .map(|i| up_blocks_with_attn.contains(&i))
        .collect();

    Ok((down_has_attn, up_has_attn))
}

/// Load UNet weights from HuggingFace format with smart block detection.
///
/// This function uses burn-store with `skip_enum_variants` to handle the enum-based
/// block types (UNetDownBlock, UNetUpBlock) without needing variant names in the
/// weight file keys.
///
/// Main differences between HuggingFace and Burn naming:
/// - Mid block resnets: `mid_block.resnets.0/1` → `mid_block.resnet` / `mid_block.attn_resnets.0.resnet_block`
/// - Mid block attention: `mid_block.attentions.0` → `mid_block.attn_resnets.0.spatial_transformer`
/// - Downsampler: `downsamplers.0` → `downsampler`
/// - Upsampler: `upsamplers.0` → `upsampler`
/// - Cross-attention keys: `to_k/to_q/to_v` → `key/query/value`
/// - Cross-attention output: `to_out.0` → `output`
/// - FeedForward: `ff.net.0.proj` → `ff.geglu.proj`, `ff.net.2` → `ff.linear_outer`
///
/// For blocks with cross-attention (CrossAttnDownBlock2D, CrossAttnUpBlock2D):
/// - resnets/downsamplers/upsamplers are nested under `downblock`/`upblock`
///
/// For basic blocks (DownBlock2D, UpBlock2D):
/// - resnets/downsamplers/upsamplers are at the top level
///
/// Uses `PyTorchToBurnAdapter` to automatically handle:
/// - Transposing linear layer weights
/// - Renaming normalization parameters (weight->gamma, bias->beta)
pub fn load_unet_safetensors<B, M, P>(
    mut module: M,
    path: P,
    _device: &B::Device,
) -> Result<M, WeightLoadError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    P: AsRef<Path>,
{
    // Inspect the file to determine block types
    let (down_has_attn, up_has_attn) = inspect_unet_block_types(path.as_ref())?;

    // Build key mappings for HuggingFace -> Burn structure
    // Order matters: more specific patterns should come first
    let mut key_mappings: Vec<(&str, &str)> = vec![
        // Mid block remappings
        ("^mid_block\\.resnets\\.0\\.", "mid_block.resnet."),
        (
            "^mid_block\\.resnets\\.1\\.",
            "mid_block.attn_resnets.0.resnet_block.",
        ),
        (
            "^mid_block\\.attentions\\.([0-9]+)\\.",
            "mid_block.attn_resnets.$1.spatial_transformer.",
        ),
        // Cross-attention key remapping
        ("\\.to_k\\.", ".key."),
        ("\\.to_q\\.", ".query."),
        ("\\.to_v\\.", ".value."),
        ("\\.to_out\\.0\\.", ".output."),
        // FeedForward remapping
        ("\\.ff\\.net\\.0\\.proj\\.", ".ff.geglu.proj."),
        ("\\.ff\\.net\\.2\\.", ".ff.linear_outer."),
    ];

    // Down block remappings - depends on whether block has attention
    // We need to use owned strings for dynamic patterns
    let mut dynamic_mappings: Vec<(String, String)> = Vec::new();

    for (i, has_attn) in down_has_attn.iter().enumerate() {
        if *has_attn {
            // CrossAttnDownBlock2D: resnets/downsamplers nested under downblock
            dynamic_mappings.push((
                format!("^down_blocks\\.{}\\.resnets\\.", i),
                format!("down_blocks.{}.downblock.resnets.", i),
            ));
            dynamic_mappings.push((
                format!("^down_blocks\\.{}\\.downsamplers\\.0\\.", i),
                format!("down_blocks.{}.downblock.downsampler.", i),
            ));
            // attentions stay at block level
        } else {
            // DownBlock2D: flat structure, just remap downsamplers.0 -> downsampler
            dynamic_mappings.push((
                format!("^down_blocks\\.{}\\.downsamplers\\.0\\.", i),
                format!("down_blocks.{}.downsampler.", i),
            ));
        }
    }

    for (i, has_attn) in up_has_attn.iter().enumerate() {
        if *has_attn {
            // CrossAttnUpBlock2D: resnets/upsamplers nested under upblock
            dynamic_mappings.push((
                format!("^up_blocks\\.{}\\.resnets\\.", i),
                format!("up_blocks.{}.upblock.resnets.", i),
            ));
            dynamic_mappings.push((
                format!("^up_blocks\\.{}\\.upsamplers\\.0\\.", i),
                format!("up_blocks.{}.upblock.upsampler.", i),
            ));
            // attentions stay at block level
        } else {
            // UpBlock2D: flat structure, just remap upsamplers.0 -> upsampler
            dynamic_mappings.push((
                format!("^up_blocks\\.{}\\.upsamplers\\.0\\.", i),
                format!("up_blocks.{}.upsampler.", i),
            ));
        }
    }

    // Combine static and dynamic mappings
    let dynamic_refs: Vec<(&str, &str)> = dynamic_mappings
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    key_mappings.extend(dynamic_refs);

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    let checkpoint_path = path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper)
        .skip_enum_variants(true); // This is the key: skip enum variant names when matching paths

    module
        .load_from(&mut store)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    Ok(module)
}

/// Load CLIP text encoder with projection weights from HuggingFace format.
///
/// This is used for the SDXL second text encoder (CLIPTextModelWithProjection).
/// The key remapping strips the `text_model.` prefix and remaps to `transformer.`
/// since our struct wraps a `ClipTextTransformer` inside a `transformer` field.
///
/// Uses `PyTorchToBurnAdapter` to automatically handle:
/// - Transposing linear layer weights
/// - Renaming normalization parameters (weight->gamma, bias->beta)
pub fn load_clip_with_projection_safetensors<B, M, P>(
    mut module: M,
    path: P,
    _device: &B::Device,
) -> Result<M, WeightLoadError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    P: AsRef<Path>,
{
    // Remap HuggingFace CLIP keys to our model structure
    // The HF format has "text_model.embeddings...", "text_model.encoder..."
    // Our CLIPTextModelWithProjection has a "transformer" field wrapping a ClipTextTransformer
    let key_mappings: Vec<(&str, &str)> = vec![("^text_model\\.", "transformer.")];

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    let checkpoint_path = path.as_ref().to_path_buf();
    let mut store = SafetensorsStore::from_file(checkpoint_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    module
        .load_from(&mut store)
        .map_err(|e| WeightLoadError::SafetensorsLoad(e.to_string()))?;

    Ok(module)
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
