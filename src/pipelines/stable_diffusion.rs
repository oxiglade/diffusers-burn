//! Stable Diffusion Pipeline
//!
//! This module provides the Stable Diffusion pipeline for text-to-image generation.
//! The pipeline combines a CLIP text encoder, a VAE, a UNet, and a noise scheduler
//! to generate images from text prompts.

use alloc::vec;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::models::unet_2d::{BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig};
use crate::models::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use crate::schedulers::{
    BetaSchedule, DDIMScheduler, DDIMSchedulerConfig, DDPMScheduler, DDPMSchedulerConfig,
    DPMSolverMultistepScheduler, DPMSolverMultistepSchedulerConfig,
    EulerAncestralDiscreteScheduler, EulerAncestralDiscreteSchedulerConfig, EulerDiscreteScheduler,
    EulerDiscreteSchedulerConfig, PNDMScheduler, PNDMSchedulerConfig, PredictionType,
};
use crate::transformers::clip::{ClipConfig, ClipTextTransformer};

/// The guidance scale for classifier-free guidance.
/// Higher values give more weight to the text prompt.
pub const GUIDANCE_SCALE: f64 = 7.5;

/// The scaling factor for the VAE latent space.
/// Latents are scaled by 1/0.18215 when encoding and 0.18215 when decoding.
pub const VAE_SCALE: f64 = 0.18215;

/// Configuration for the Stable Diffusion pipeline.
#[derive(Debug, Clone)]
pub struct StableDiffusionConfig {
    /// Width of the generated image in pixels.
    pub width: usize,
    /// Height of the generated image in pixels.
    pub height: usize,
    /// CLIP text encoder configuration.
    pub clip: ClipConfig,
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

impl StableDiffusionConfig {
    /// Create a configuration for Stable Diffusion v1.5.
    ///
    /// Reference: https://huggingface.co/runwayml/stable-diffusion-v1-5
    pub fn v1_5(
        sliced_attention_size: Option<usize>,
        height: Option<usize>,
        width: Option<usize>,
    ) -> Self {
        let height = height.unwrap_or(512);
        let width = width.unwrap_or(512);
        assert!(height % 8 == 0, "height must be divisible by 8");
        assert!(width % 8 == 0, "width must be divisible by 8");

        // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json
        let unet = UNet2DConditionModelConfig {
            blocks: vec![
                BlockConfig::new(320)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
                BlockConfig::new(640)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
                BlockConfig::new(1280)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
                BlockConfig::new(1280)
                    .with_use_cross_attn(false)
                    .with_attention_head_dim(8),
            ],
            center_input_sample: false,
            cross_attention_dim: 768,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            layers_per_block: 2,
            mid_block_scale_factor: 1.0,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size,
            use_linear_projection: false,
        };

        // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
        let vae = AutoEncoderKLConfig::new()
            .with_block_out_channels(vec![128, 256, 512, 512])
            .with_layers_per_block(2)
            .with_latent_channels(4)
            .with_norm_num_groups(32);

        Self {
            width,
            height,
            clip: ClipConfig::v1_5(),
            vae,
            unet,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }

    /// Create a configuration for Stable Diffusion v2.1.
    ///
    /// Reference: https://huggingface.co/stabilityai/stable-diffusion-2-1
    pub fn v2_1(
        sliced_attention_size: Option<usize>,
        height: Option<usize>,
        width: Option<usize>,
    ) -> Self {
        let height = height.unwrap_or(768);
        let width = width.unwrap_or(768);
        assert!(height % 8 == 0, "height must be divisible by 8");
        assert!(width % 8 == 0, "width must be divisible by 8");

        // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/unet/config.json
        let unet = UNet2DConditionModelConfig {
            blocks: vec![
                BlockConfig::new(320)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(5),
                BlockConfig::new(640)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(10),
                BlockConfig::new(1280)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(20),
                BlockConfig::new(1280)
                    .with_use_cross_attn(false)
                    .with_attention_head_dim(20),
            ],
            center_input_sample: false,
            cross_attention_dim: 1024,
            downsample_padding: 1,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            layers_per_block: 2,
            mid_block_scale_factor: 1.0,
            norm_eps: 1e-5,
            norm_num_groups: 32,
            sliced_attention_size,
            use_linear_projection: true,
        };

        // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/vae/config.json
        let vae = AutoEncoderKLConfig::new()
            .with_block_out_channels(vec![128, 256, 512, 512])
            .with_layers_per_block(2)
            .with_latent_channels(4)
            .with_norm_num_groups(32);

        Self {
            width,
            height,
            clip: ClipConfig::v2_1(),
            vae,
            unet,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            prediction_type: PredictionType::VPrediction,
            train_timesteps: 1000,
        }
    }

    /// Initialize the CLIP text transformer.
    pub fn build_clip_transformer<B: Backend>(&self, device: &B::Device) -> ClipTextTransformer<B> {
        self.clip.init_text_transformer(device)
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

    /// Build a DDPM scheduler.
    pub fn build_ddpm_scheduler<B: Backend>(
        &self,
        n_steps: usize,
        device: &B::Device,
    ) -> DDPMScheduler {
        let config = DDPMSchedulerConfig {
            beta_start: self.beta_start,
            beta_end: self.beta_end,
            beta_schedule: self.beta_schedule,
            prediction_type: self.prediction_type,
            train_timesteps: self.train_timesteps,
            ..DDPMSchedulerConfig::default()
        };
        DDPMScheduler::new::<B>(n_steps, config, device)
    }

    /// Build a DPM-Solver++ Multistep scheduler.
    pub fn build_dpm_solver_scheduler<B: Backend>(
        &self,
        n_steps: usize,
        device: &B::Device,
    ) -> DPMSolverMultistepScheduler<B> {
        let config = DPMSolverMultistepSchedulerConfig {
            beta_start: self.beta_start,
            beta_end: self.beta_end,
            beta_schedule: self.beta_schedule,
            prediction_type: self.prediction_type,
            train_timesteps: self.train_timesteps,
            ..DPMSolverMultistepSchedulerConfig::default()
        };
        DPMSolverMultistepScheduler::new(n_steps, config, device)
    }

    /// Build an Euler Discrete scheduler.
    pub fn build_euler_discrete_scheduler(&self, n_steps: usize) -> EulerDiscreteScheduler {
        let config = EulerDiscreteSchedulerConfig {
            beta_start: self.beta_start,
            beta_end: self.beta_end,
            beta_schedule: self.beta_schedule,
            prediction_type: self.prediction_type,
            train_timesteps: self.train_timesteps,
            ..EulerDiscreteSchedulerConfig::default()
        };
        EulerDiscreteScheduler::new(n_steps, config)
    }

    /// Build an Euler Ancestral Discrete scheduler.
    pub fn build_euler_ancestral_scheduler(
        &self,
        n_steps: usize,
    ) -> EulerAncestralDiscreteScheduler {
        let config = EulerAncestralDiscreteSchedulerConfig {
            beta_start: self.beta_start,
            beta_end: self.beta_end,
            beta_schedule: self.beta_schedule,
            prediction_type: self.prediction_type,
            train_timesteps: self.train_timesteps,
        };
        EulerAncestralDiscreteScheduler::new(n_steps, config)
    }

    /// Build a PNDM scheduler.
    pub fn build_pndm_scheduler<B: Backend>(
        &self,
        n_steps: usize,
        device: &B::Device,
    ) -> PNDMScheduler<B> {
        let config = PNDMSchedulerConfig {
            beta_start: self.beta_start,
            beta_end: self.beta_end,
            beta_schedule: self.beta_schedule,
            prediction_type: self.prediction_type,
            train_timesteps: self.train_timesteps,
            ..PNDMSchedulerConfig::default()
        };
        PNDMScheduler::new(n_steps, config, device)
    }

    /// Initialize the complete Stable Diffusion pipeline.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StableDiffusion<B> {
        StableDiffusion {
            clip: self.build_clip_transformer(device),
            vae: self.build_vae(device),
            unet: self.build_unet(device, 4),
            width: self.width,
            height: self.height,
        }
    }
}

/// The Stable Diffusion pipeline.
///
/// This struct holds all the models needed for text-to-image generation.
#[derive(Module, Debug)]
pub struct StableDiffusion<B: Backend> {
    /// The CLIP text encoder.
    pub clip: ClipTextTransformer<B>,
    /// The VAE for encoding/decoding images.
    pub vae: AutoEncoderKL<B>,
    /// The UNet for denoising.
    pub unet: UNet2DConditionModel<B>,
    /// Width of the generated image.
    pub width: usize,
    /// Height of the generated image.
    pub height: usize,
}

impl<B: Backend> StableDiffusion<B> {
    /// Encode text tokens to embeddings using the CLIP model.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs from the tokenizer [batch_size, seq_len]
    ///
    /// # Returns
    /// Text embeddings [batch_size, seq_len, embed_dim]
    pub fn encode_text(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.clip.forward(tokens)
    }

    /// Encode an image to latent space.
    ///
    /// # Arguments
    /// * `image` - Input image tensor [batch_size, 3, height, width] with values in [0, 1]
    ///
    /// # Returns
    /// Latent tensor [batch_size, 4, height/8, width/8]
    pub fn encode_image(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        // Scale image to [-1, 1]
        let image = image * 2.0 - 1.0;
        // Encode and sample from the distribution
        let dist = self.vae.encode(image);
        // Scale latent
        dist.sample() * VAE_SCALE
    }

    /// Decode latent vectors to images.
    ///
    /// # Arguments
    /// * `latents` - Latent tensor [batch_size, 4, height/8, width/8]
    ///
    /// # Returns
    /// Image tensor [batch_size, 3, height, width] with values in [0, 1]
    pub fn decode_latents(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        // Scale latent
        let latents = latents / VAE_SCALE;
        // Decode
        let image = self.vae.decode(latents);
        // Scale back to [0, 1]
        (image / 2.0 + 0.5).clamp(0.0, 1.0)
    }

    /// Predict the noise for a given noisy latent and timestep.
    ///
    /// # Arguments
    /// * `latents` - Noisy latent tensor [batch_size, 4, height/8, width/8]
    /// * `timestep` - Current diffusion timestep
    /// * `encoder_hidden_states` - Text embeddings [batch_size, seq_len, embed_dim]
    ///
    /// # Returns
    /// Predicted noise tensor [batch_size, 4, height/8, width/8]
    pub fn predict_noise(
        &self,
        latents: Tensor<B, 4>,
        timestep: f64,
        encoder_hidden_states: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        self.unet.forward(latents, timestep, encoder_hidden_states)
    }

    /// Get text embeddings with classifier-free guidance.
    ///
    /// Concatenates unconditional (empty prompt) and conditional embeddings.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Token IDs for the prompt [1, seq_len]
    /// * `uncond_tokens` - Token IDs for empty/negative prompt [1, seq_len]
    ///
    /// # Returns
    /// Combined embeddings [2, seq_len, embed_dim] (uncond first, then cond)
    pub fn encode_prompt_with_guidance(
        &self,
        prompt_tokens: Tensor<B, 2, Int>,
        uncond_tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let text_embeddings = self.encode_text(prompt_tokens);
        let uncond_embeddings = self.encode_text(uncond_tokens);
        Tensor::cat(vec![uncond_embeddings, text_embeddings], 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_5_config() {
        let config = StableDiffusionConfig::v1_5(None, None, None);

        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert_eq!(config.unet.cross_attention_dim, 768);
        assert_eq!(config.unet.blocks.len(), 4);
        assert!(!config.unet.use_linear_projection);
    }

    #[test]
    fn test_v2_1_config() {
        let config = StableDiffusionConfig::v2_1(None, None, None);

        assert_eq!(config.width, 768);
        assert_eq!(config.height, 768);
        assert_eq!(config.unet.cross_attention_dim, 1024);
        assert_eq!(config.unet.blocks.len(), 4);
        assert!(config.unet.use_linear_projection);
    }

    #[test]
    fn test_custom_dimensions() {
        let config = StableDiffusionConfig::v1_5(None, Some(768), Some(1024));

        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
    }

    #[test]
    #[should_panic(expected = "height must be divisible by 8")]
    fn test_invalid_height() {
        let _ = StableDiffusionConfig::v1_5(None, Some(513), None);
    }

    #[test]
    #[should_panic(expected = "width must be divisible by 8")]
    fn test_invalid_width() {
        let _ = StableDiffusionConfig::v1_5(None, None, Some(513));
    }
}
