//! Stable Diffusion XL Pipeline
//!
//! This module provides the Stable Diffusion XL pipeline for text-to-image generation.
//! SDXL uses two CLIP text encoders (one standard, one with projection), a VAE,
//! and a UNet with additional conditioning (pooled embeddings + time IDs).
//!
//! # Example (pseudocode)
//!
//! ```ignore
//! // Create configuration
//! let config = StableDiffusionXLConfig::xl(None, None);
//!
//! // Build pipeline components
//! let pipeline = config.init::<MyBackend>(&device);
//!
//! // Build scheduler
//! let scheduler = config.build_ddim_scheduler::<MyBackend>(30, &device);
//!
//! // Tokenize prompt (requires std feature)
//! let tokenizer = SimpleTokenizer::new("bpe_simple_vocab_16e6.txt", SimpleTokenizerConfig::v1_5())?;
//! let tokens = tokenizer.encode("a photo of a cat")?;
//! let uncond_tokens = tokenizer.encode("")?;
//!
//! // Generate image
//! let image = generate_image_sdxl_ddim(
//!     &pipeline,
//!     &scheduler,
//!     tokens,
//!     uncond_tokens,
//!     7.5,  // guidance_scale
//!     42,   // seed
//!     &device,
//! );
//! ```

use alloc::vec;
use alloc::vec::Vec;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use crate::models::unet_2d::{
    AddedCondKwargs, BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig,
};
use crate::models::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use crate::schedulers::{BetaSchedule, DDIMScheduler, DDIMSchedulerConfig, PredictionType};
use crate::transformers::clip::{
    CLIPTextModelWithProjection, CLIPTextModelWithProjectionConfig, ClipConfig, ClipTextTransformer,
};

/// The scaling factor for the SDXL VAE latent space.
pub const VAE_SCALE_XL: f64 = 0.13025;

/// Configuration for the Stable Diffusion XL pipeline.
#[derive(Debug, Clone)]
pub struct StableDiffusionXLConfig {
    /// Width of the generated image in pixels.
    pub width: usize,
    /// Height of the generated image in pixels.
    pub height: usize,
    /// CLIP text encoder 1 configuration.
    pub clip: ClipConfig,
    /// CLIP text encoder 2 (with projection) configuration.
    pub clip2: CLIPTextModelWithProjectionConfig,
    /// VAE configuration.
    pub vae: AutoEncoderKLConfig,
    /// UNet configuration.
    pub unet: UNet2DConditionModelConfig,
    /// Beta schedule start value.
    pub beta_start: f64,
    /// Beta schedule end value.
    pub beta_end: f64,
    /// Beta schedule type.
    pub beta_schedule: BetaSchedule,
    /// Prediction type for the scheduler.
    pub prediction_type: PredictionType,
    /// Number of training timesteps.
    pub train_timesteps: usize,
}

impl StableDiffusionXLConfig {
    /// Create a configuration for Stable Diffusion XL.
    ///
    /// Reference: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    pub fn xl(height: Option<usize>, width: Option<usize>) -> Self {
        let height = height.unwrap_or(1024);
        let width = width.unwrap_or(1024);
        assert!(height.is_multiple_of(8), "height must be divisible by 8");
        assert!(width.is_multiple_of(8), "width must be divisible by 8");

        // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json
        let unet = UNet2DConditionModelConfig {
            blocks: vec![
                BlockConfig::new(320)
                    .with_use_cross_attn(false)
                    .with_attention_head_dim(5)
                    .with_transformer_layers_per_block(1),
                BlockConfig::new(640)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(10)
                    .with_transformer_layers_per_block(2),
                BlockConfig::new(1280)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(20)
                    .with_transformer_layers_per_block(10),
            ],
            center_input_sample: false,
            cross_attention_dim: 2048,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            layers_per_block: 2,
            mid_block_scale_factor: 1.0,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size: None,
            use_linear_projection: true,
            addition_time_embed_dim: Some(256),
            projection_class_embeddings_input_dim: Some(2816),
        };

        // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/config.json
        let vae = AutoEncoderKLConfig::new()
            .with_block_out_channels(vec![128, 256, 512, 512])
            .with_layers_per_block(2)
            .with_latent_channels(4)
            .with_norm_num_groups(32);

        Self {
            width,
            height,
            clip: ClipConfig::v1_5(),
            clip2: CLIPTextModelWithProjectionConfig::sdxl2(),
            vae,
            unet,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }

    /// Initialize the first CLIP text transformer (encoder 1).
    pub fn build_clip_transformer<B: Backend>(&self, device: &B::Device) -> ClipTextTransformer<B> {
        self.clip.init_text_transformer(device)
    }

    /// Initialize the second CLIP text model with projection (encoder 2).
    pub fn build_clip_with_projection<B: Backend>(
        &self,
        device: &B::Device,
    ) -> CLIPTextModelWithProjection<B> {
        self.clip2.init(device)
    }

    /// Initialize the VAE.
    pub fn build_vae<B: Backend>(&self, device: &B::Device) -> AutoEncoderKL<B> {
        self.vae.init(3, 3, device)
    }

    /// Initialize the UNet.
    pub fn build_unet<B: Backend>(
        &self,
        device: &B::Device,
        in_channels: usize,
    ) -> UNet2DConditionModel<B> {
        self.unet.init(in_channels, 4, device)
    }

    /// Build a DDIM scheduler.
    pub fn build_ddim_scheduler<B: Backend>(
        &self,
        n_steps: usize,
        device: &B::Device,
    ) -> DDIMScheduler {
        let config = DDIMSchedulerConfig {
            beta_start: self.beta_start,
            beta_end: self.beta_end,
            beta_schedule: self.beta_schedule,
            prediction_type: self.prediction_type,
            train_timesteps: self.train_timesteps,
            ..DDIMSchedulerConfig::default()
        };
        DDIMScheduler::new::<B>(n_steps, config, device)
    }

    /// Initialize the complete Stable Diffusion XL pipeline.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StableDiffusionXL<B> {
        StableDiffusionXL {
            clip: self.build_clip_transformer(device),
            clip_with_proj: self.build_clip_with_projection(device),
            vae: self.build_vae(device),
            unet: self.build_unet(device, 4),
            width: self.width,
            height: self.height,
        }
    }
}

/// The Stable Diffusion XL pipeline.
///
/// This struct holds all the models needed for SDXL text-to-image generation,
/// including two CLIP encoders, a VAE, and a UNet.
#[derive(Module, Debug)]
pub struct StableDiffusionXL<B: Backend> {
    /// The first CLIP text encoder (standard, produces hidden states).
    pub clip: ClipTextTransformer<B>,
    /// The second CLIP text encoder with projection (produces hidden states + pooled output).
    pub clip_with_proj: CLIPTextModelWithProjection<B>,
    /// The VAE for encoding/decoding images.
    pub vae: AutoEncoderKL<B>,
    /// The UNet for denoising.
    pub unet: UNet2DConditionModel<B>,
    /// Width of the generated image.
    pub width: usize,
    /// Height of the generated image.
    pub height: usize,
}

/// Generate an image using the Stable Diffusion XL pipeline with a DDIM scheduler.
///
/// This function implements the full SDXL diffusion loop with dual CLIP encoding
/// and classifier-free guidance.
///
/// # Arguments
/// * `pipeline` - The Stable Diffusion XL pipeline with loaded models
/// * `scheduler` - The DDIM scheduler configured with the number of steps
/// * `prompt_tokens` - Tokenized prompt as a vector of token IDs
/// * `uncond_tokens` - Tokenized empty/negative prompt as a vector of token IDs
/// * `guidance_scale` - Classifier-free guidance scale (typically 7.5)
/// * `seed` - Random seed for reproducibility
/// * `device` - Device to run inference on
///
/// # Returns
/// Generated image tensor [1, 3, height, width] with values in [0, 1]
#[allow(clippy::too_many_arguments)]
pub fn generate_image_sdxl_ddim<B: Backend>(
    pipeline: &StableDiffusionXL<B>,
    scheduler: &DDIMScheduler,
    prompt_tokens: &[usize],
    prompt_tokens_2: &[usize],
    guidance_scale: f64,
    seed: u64,
    device: &B::Device,
) -> Tensor<B, 4> {
    // Seed the random number generator for reproducibility
    B::seed(device, seed);

    // Convert tokens to tensors
    let prompt_tokens: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
    let prompt_tokens_2: Vec<i64> = prompt_tokens_2.iter().map(|&x| x as i64).collect();

    let prompt_tensor: Tensor<B, 1, Int> = Tensor::from_ints(&prompt_tokens[..], device);
    let prompt_tensor: Tensor<B, 2, Int> = prompt_tensor.unsqueeze_dim(0);
    let prompt_tensor_2: Tensor<B, 1, Int> = Tensor::from_ints(&prompt_tokens_2[..], device);
    let prompt_tensor_2: Tensor<B, 2, Int> = prompt_tensor_2.unsqueeze_dim(0);

    // Encoder 1: penultimate layer output (no final_layer_norm)
    let text_emb_1 = pipeline.clip.forward_penultimate(prompt_tensor);

    // Encoder 2: penultimate hidden states + pooled projection
    let (text_emb_2, text_pooled) = pipeline.clip_with_proj.forward(prompt_tensor_2);

    // Concatenate hidden states from both encoders: [batch, 77, 768+1280]
    let text_embeddings = Tensor::cat(vec![text_emb_1, text_emb_2], 2);

    // SDXL uses zeroed-out negative embeddings (force_zeros_for_empty_prompt=True)
    let uncond_embeddings: Tensor<B, 3> = Tensor::zeros(text_embeddings.dims(), device);
    let uncond_pooled: Tensor<B, 2> = Tensor::zeros(text_pooled.dims(), device);

    // Combine for guidance: [2, 77, 2048]
    let combined_emb = Tensor::cat(vec![uncond_embeddings, text_embeddings], 0);

    // Pooled: [2, 1280]
    let pooled = Tensor::cat(vec![uncond_pooled, text_pooled], 0);

    // Build time_ids: [height, width, crop_top, crop_left, original_height, original_width]
    let time_ids_single: Tensor<B, 1> = Tensor::from_floats(
        [
            pipeline.height as f32,
            pipeline.width as f32,
            0.0,
            0.0,
            pipeline.height as f32,
            pipeline.width as f32,
        ],
        device,
    );
    let time_ids: Tensor<B, 2> = time_ids_single.unsqueeze_dim(0);
    let time_ids = Tensor::cat(vec![time_ids.clone(), time_ids], 0); // [2, 6] for guidance

    // Initialize random latents
    let latent_height = pipeline.height / 8;
    let latent_width = pipeline.width / 8;
    let mut latents: Tensor<B, 4> = Tensor::random(
        [1, 4, latent_height, latent_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );

    // Scale initial noise by scheduler's init_noise_sigma
    latents = latents * scheduler.init_noise_sigma();

    // Diffusion loop
    for &timestep in scheduler.timesteps().iter() {
        // Duplicate latents for classifier-free guidance (uncond + cond)
        let latent_model_input = Tensor::cat(vec![latents.clone(), latents.clone()], 0);

        // Scale model input
        let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);

        // Predict noise with additional SDXL conditioning
        let noise_pred = pipeline.unet.forward_with_additional_residuals(
            latent_model_input,
            timestep as f64,
            combined_emb.clone(),
            None,
            None,
            Some(&AddedCondKwargs {
                text_embeds: pooled.clone(),
                time_ids: time_ids.clone(),
            }),
        );

        // Split predictions for guidance
        let [noise_pred_uncond, noise_pred_text] = noise_pred.chunk(2, 0).try_into().unwrap();

        // Apply classifier-free guidance
        let noise_pred =
            noise_pred_uncond.clone() + (noise_pred_text - noise_pred_uncond) * guidance_scale;

        // Scheduler step
        latents = scheduler.step(&noise_pred, timestep, &latents);
    }

    // Decode latents to image using SDXL VAE scale
    let latents = latents / VAE_SCALE_XL;
    let image = pipeline.vae.decode(latents);
    (image / 2.0 + 0.5).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdxl_config() {
        let config = StableDiffusionXLConfig::xl(None, None);
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 1024);
        assert_eq!(config.unet.blocks.len(), 3);
        assert_eq!(config.unet.cross_attention_dim, 2048);
        assert_eq!(config.unet.addition_time_embed_dim, Some(256));
        assert_eq!(config.unet.blocks[2].transformer_layers_per_block, 10);
        assert!(config.unet.use_linear_projection);
    }

    #[test]
    fn test_sdxl_custom_dimensions() {
        let config = StableDiffusionXLConfig::xl(Some(768), Some(1024));
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
    }

    #[test]
    #[should_panic(expected = "height must be divisible by 8")]
    fn test_sdxl_invalid_height() {
        let _ = StableDiffusionXLConfig::xl(Some(513), None);
    }

    #[test]
    #[should_panic(expected = "width must be divisible by 8")]
    fn test_sdxl_invalid_width() {
        let _ = StableDiffusionXLConfig::xl(None, Some(513));
    }
}
