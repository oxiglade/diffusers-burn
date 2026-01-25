//! 2D UNet Denoising Models
//!
//! The 2D UNet models take as input a noisy sample and the current diffusion
//! timestep and return a denoised version of the input.

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, PaddingConfig2d};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use alloc::vec;
use alloc::vec::Vec;

use super::embeddings::{get_timestep_embedding, TimestepEmbedding, TimestepEmbeddingConfig};
use super::unet_2d_blocks::{
    CrossAttnDownBlock2D, CrossAttnDownBlock2DConfig, CrossAttnUpBlock2D, CrossAttnUpBlock2DConfig,
    DownBlock2D, DownBlock2DConfig, UNetMidBlock2DCrossAttn, UNetMidBlock2DCrossAttnConfig,
    UpBlock2D, UpBlock2DConfig,
};

/// Configuration for a single UNet block.
#[derive(Debug, Clone, burn::serde::Serialize, burn::serde::Deserialize)]
pub struct BlockConfig {
    /// Output channels for this block.
    pub out_channels: usize,
    /// Whether to use cross-attention in this block.
    pub use_cross_attn: bool,
    /// Number of attention heads.
    pub attention_head_dim: usize,
}

impl BlockConfig {
    /// Create a new block configuration.
    pub fn new(out_channels: usize) -> Self {
        Self {
            out_channels,
            use_cross_attn: true,
            attention_head_dim: 8,
        }
    }

    /// Set whether to use cross-attention.
    pub fn with_use_cross_attn(mut self, use_cross_attn: bool) -> Self {
        self.use_cross_attn = use_cross_attn;
        self
    }

    /// Set the attention head dimension.
    pub fn with_attention_head_dim(mut self, attention_head_dim: usize) -> Self {
        self.attention_head_dim = attention_head_dim;
        self
    }
}

/// Configuration for the UNet2DConditionModel.
#[derive(Config, Debug)]
pub struct UNet2DConditionModelConfig {
    /// Whether to center the input sample.
    #[config(default = false)]
    pub center_input_sample: bool,
    /// Whether to flip sin to cos in timestep embedding.
    #[config(default = true)]
    pub flip_sin_to_cos: bool,
    /// Frequency shift for timestep embedding.
    #[config(default = 0.0)]
    pub freq_shift: f64,
    /// Configuration for each block.
    pub blocks: Vec<BlockConfig>,
    /// Number of ResNet layers per block.
    #[config(default = 2)]
    pub layers_per_block: usize,
    /// Padding for downsampling convolutions.
    #[config(default = 1)]
    pub downsample_padding: usize,
    /// Scale factor for mid block.
    #[config(default = 1.0)]
    pub mid_block_scale_factor: f64,
    /// Number of groups for group normalization.
    #[config(default = 32)]
    pub norm_num_groups: usize,
    /// Epsilon for normalization layers.
    #[config(default = 1e-5)]
    pub norm_eps: f64,
    /// Dimension of cross-attention context.
    #[config(default = 1280)]
    pub cross_attention_dim: usize,
    /// Size for sliced attention (None for full attention).
    pub sliced_attention_size: Option<usize>,
    /// Whether to use linear projection in attention.
    #[config(default = false)]
    pub use_linear_projection: bool,
}

impl Default for UNet2DConditionModelConfig {
    fn default() -> Self {
        Self {
            center_input_sample: false,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
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
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.0,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 1280,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

/// Down block types for UNet.
#[derive(Module, Debug)]
pub enum UNetDownBlock<B: Backend> {
    Basic(DownBlock2D<B>),
    CrossAttn(CrossAttnDownBlock2D<B>),
}

/// Up block types for UNet.
#[derive(Module, Debug)]
pub enum UNetUpBlock<B: Backend> {
    Basic(UpBlock2D<B>),
    CrossAttn(CrossAttnUpBlock2D<B>),
}

/// UNet2D Conditional Model for denoising diffusion.
///
/// This model takes a noisy sample, timestep, and encoder hidden states
/// (from text conditioning) and predicts the noise to be removed.
#[derive(Module, Debug)]
pub struct UNet2DConditionModel<B: Backend> {
    conv_in: Conv2d<B>,
    time_embedding: TimestepEmbedding<B>,
    down_blocks: Vec<UNetDownBlock<B>>,
    mid_block: UNetMidBlock2DCrossAttn<B>,
    up_blocks: Vec<UNetUpBlock<B>>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
    #[module(skip)]
    time_proj_channels: usize,
    #[module(skip)]
    flip_sin_to_cos: bool,
    #[module(skip)]
    freq_shift: f64,
    #[module(skip)]
    center_input_sample: bool,
}

impl UNet2DConditionModelConfig {
    /// Initialize the UNet2DConditionModel.
    pub fn init<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &B::Device,
    ) -> UNet2DConditionModel<B> {
        let n_blocks = self.blocks.len();
        let b_channels = self.blocks[0].out_channels;
        let bl_channels = self.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = self.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;

        // Input convolution
        let conv_in = Conv2dConfig::new([in_channels, b_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // Time embeddings
        let time_embedding = TimestepEmbeddingConfig::new(b_channels, time_embed_dim).init(device);

        // Down blocks
        let down_blocks = (0..n_blocks)
            .map(|i| {
                let block_config = &self.blocks[i];
                let out_channels = block_config.out_channels;
                let attention_head_dim = block_config.attention_head_dim;

                // Enable automatic attention slicing if config sliced_attention_size is 0
                let sliced_attention_size = match self.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    other => other,
                };

                let in_ch = if i > 0 {
                    self.blocks[i - 1].out_channels
                } else {
                    b_channels
                };

                let db_config = DownBlock2DConfig::new(in_ch, out_channels)
                    .with_temb_channels(Some(time_embed_dim))
                    .with_n_layers(self.layers_per_block)
                    .with_resnet_eps(self.norm_eps)
                    .with_resnet_groups(self.norm_num_groups)
                    .with_add_downsample(i < n_blocks - 1)
                    .with_downsample_padding(self.downsample_padding);

                if block_config.use_cross_attn {
                    let config = CrossAttnDownBlock2DConfig::new(in_ch, out_channels, db_config)
                        .with_temb_channels(Some(time_embed_dim))
                        .with_attn_num_head_channels(attention_head_dim)
                        .with_cross_attention_dim(self.cross_attention_dim)
                        .with_sliced_attention_size(sliced_attention_size)
                        .with_use_linear_projection(self.use_linear_projection);
                    UNetDownBlock::CrossAttn(config.init(device))
                } else {
                    UNetDownBlock::Basic(db_config.init(device))
                }
            })
            .collect();

        // Mid block
        let mid_config = UNetMidBlock2DCrossAttnConfig::new(bl_channels)
            .with_temb_channels(Some(time_embed_dim))
            .with_resnet_eps(self.norm_eps)
            .with_output_scale_factor(self.mid_block_scale_factor)
            .with_cross_attn_dim(self.cross_attention_dim)
            .with_attn_num_head_channels(bl_attention_head_dim)
            .with_resnet_groups(Some(self.norm_num_groups))
            .with_use_linear_projection(self.use_linear_projection);
        let mid_block = mid_config.init(device);

        // Up blocks
        let up_blocks = (0..n_blocks)
            .map(|i| {
                let block_config = &self.blocks[n_blocks - 1 - i];
                let out_channels = block_config.out_channels;
                let attention_head_dim = block_config.attention_head_dim;

                // Enable automatic attention slicing if config sliced_attention_size is 0
                let sliced_attention_size = match self.sliced_attention_size {
                    Some(0) => Some(attention_head_dim / 2),
                    other => other,
                };

                let prev_out_channels = if i > 0 {
                    self.blocks[n_blocks - i].out_channels
                } else {
                    bl_channels
                };

                let in_ch = {
                    let index = if i == n_blocks - 1 {
                        0
                    } else {
                        n_blocks - i - 2
                    };
                    self.blocks[index].out_channels
                };

                let ub_config = UpBlock2DConfig::new(in_ch, prev_out_channels, out_channels)
                    .with_temb_channels(Some(time_embed_dim))
                    .with_n_layers(self.layers_per_block + 1)
                    .with_resnet_eps(self.norm_eps)
                    .with_resnet_groups(self.norm_num_groups)
                    .with_add_upsample(i < n_blocks - 1);

                if block_config.use_cross_attn {
                    let config = CrossAttnUpBlock2DConfig::new(
                        in_ch,
                        prev_out_channels,
                        out_channels,
                        ub_config,
                    )
                    .with_temb_channels(Some(time_embed_dim))
                    .with_attn_num_head_channels(attention_head_dim)
                    .with_cross_attention_dim(self.cross_attention_dim)
                    .with_sliced_attention_size(sliced_attention_size)
                    .with_use_linear_projection(self.use_linear_projection);
                    UNetUpBlock::CrossAttn(config.init(device))
                } else {
                    UNetUpBlock::Basic(ub_config.init(device))
                }
            })
            .collect();

        // Output layers
        let conv_norm_out = GroupNormConfig::new(self.norm_num_groups, b_channels)
            .with_epsilon(self.norm_eps)
            .init(device);

        let conv_out = Conv2dConfig::new([b_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        UNet2DConditionModel {
            conv_in,
            time_embedding,
            down_blocks,
            mid_block,
            time_proj_channels: b_channels,
            flip_sin_to_cos: self.flip_sin_to_cos,
            freq_shift: self.freq_shift,
            up_blocks,
            conv_norm_out,
            conv_out,
            center_input_sample: self.center_input_sample,
        }
    }
}

impl<B: Backend> UNet2DConditionModel<B> {
    /// Forward pass through the UNet.
    ///
    /// # Arguments
    /// * `xs` - Noisy input tensor [batch, channels, height, width]
    /// * `timestep` - Current diffusion timestep
    /// * `encoder_hidden_states` - Encoder hidden states for cross-attention [batch, seq_len, dim]
    pub fn forward(
        &self,
        xs: Tensor<B, 4>,
        timestep: f64,
        encoder_hidden_states: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        self.forward_with_additional_residuals(xs, timestep, encoder_hidden_states, None, None)
    }

    /// Forward pass with additional residuals (for ControlNet support).
    pub fn forward_with_additional_residuals(
        &self,
        xs: Tensor<B, 4>,
        timestep: f64,
        encoder_hidden_states: Tensor<B, 3>,
        down_block_additional_residuals: Option<&[Tensor<B, 4>]>,
        mid_block_additional_residual: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let [bsize, _channels, height, width] = xs.dims();
        let device = xs.device();
        let n_blocks = self.down_blocks.len();
        let num_upsamplers = n_blocks - 1;
        let default_overall_up_factor = 2usize.pow(num_upsamplers as u32);
        let forward_upsample_size =
            height % default_overall_up_factor != 0 || width % default_overall_up_factor != 0;

        // 0. Center input if necessary
        let xs = if self.center_input_sample {
            xs * 2.0 - 1.0
        } else {
            xs
        };

        // 1. Time embedding
        let timesteps: Tensor<B, 1> = Tensor::full([bsize], timestep as f32, &device);
        let emb = get_timestep_embedding(
            timesteps,
            self.time_proj_channels,
            self.flip_sin_to_cos,
            self.freq_shift,
        );
        let emb = self.time_embedding.forward(emb);

        // 2. Pre-process
        let xs = self.conv_in.forward(xs);

        // 3. Down blocks
        let mut down_block_res_xs = vec![xs.clone()];
        let mut xs = xs;
        for down_block in self.down_blocks.iter() {
            let (new_xs, res_xs) = match down_block {
                UNetDownBlock::Basic(b) => b.forward(xs, Some(emb.clone())),
                UNetDownBlock::CrossAttn(b) => {
                    b.forward(xs, Some(emb.clone()), Some(encoder_hidden_states.clone()))
                }
            };
            down_block_res_xs.extend(res_xs);
            xs = new_xs;
        }

        // Add additional residuals if provided (for ControlNet)
        let mut down_block_res_xs = if let Some(additional) = down_block_additional_residuals {
            down_block_res_xs
                .iter()
                .zip(additional.iter())
                .map(|(r, a)| r.clone() + a.clone())
                .collect()
        } else {
            down_block_res_xs
        };

        // 4. Mid block
        let xs = self
            .mid_block
            .forward(xs, Some(emb.clone()), Some(encoder_hidden_states.clone()));
        let xs = match mid_block_additional_residual {
            Some(m) => xs + m.clone(),
            None => xs,
        };

        // 5. Up blocks
        let mut xs = xs;
        let mut upsample_size = None;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up_block {
                UNetUpBlock::Basic(b) => b.resnets.len(),
                UNetUpBlock::CrossAttn(b) => b.upblock.resnets.len(),
            };
            let res_xs: Vec<_> = down_block_res_xs
                .drain(down_block_res_xs.len() - n_resnets..)
                .collect();

            if i < n_blocks - 1 && forward_upsample_size {
                let last = down_block_res_xs.last().unwrap();
                let [_, _, h, w] = last.dims();
                upsample_size = Some((h, w));
            }

            xs = match up_block {
                UNetUpBlock::Basic(b) => b.forward(xs, &res_xs, Some(emb.clone()), upsample_size),
                UNetUpBlock::CrossAttn(b) => b.forward(
                    xs,
                    &res_xs,
                    Some(emb.clone()),
                    upsample_size,
                    Some(encoder_hidden_states.clone()),
                ),
            };
        }

        // 6. Post-process
        let xs = self.conv_norm_out.forward(xs);
        let xs = silu(xs);
        self.conv_out.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::module::{Module, ModuleMapper, Param};
    use burn::tensor::{Shape, TensorData};

    /// A ModuleMapper that sets weights to one value and biases to another.
    /// Uses the field name from enter_module to distinguish weight vs bias.
    struct WeightBiasMapper<'a, B: Backend> {
        weight_val: f32,
        bias_val: f32,
        device: &'a B::Device,
        current_field: String,
    }

    impl<'a, B: Backend> ModuleMapper<B> for WeightBiasMapper<'a, B> {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.current_field = name.to_string();
        }

        fn map_float<const D: usize>(
            &mut self,
            tensor: Param<Tensor<B, D>>,
        ) -> Param<Tensor<B, D>> {
            let shape = tensor.shape();
            let dims: [usize; D] = shape.dims();

            // Use field name to distinguish: "bias" and "beta" get bias_val,
            // "weight" and "gamma" get weight_val
            let is_bias = self.current_field.contains("bias") || self.current_field == "beta";
            let val = if is_bias {
                self.bias_val
            } else {
                self.weight_val
            };

            Param::initialized(tensor.id, Tensor::full(dims, val, self.device))
        }
    }

    /// Set weights and biases to different constant values.
    fn set_weights_and_biases<B: Backend>(
        unet: UNet2DConditionModel<B>,
        weight_val: f32,
        bias_val: f32,
        device: &B::Device,
    ) -> UNet2DConditionModel<B> {
        let mut mapper = WeightBiasMapper::<B> {
            weight_val,
            bias_val,
            device,
            current_field: String::new(),
        };
        unet.map(&mut mapper)
    }

    #[test]
    fn test_unet2d_output_shape() {
        let device = Default::default();

        // Create a small UNet for testing
        let config = UNet2DConditionModelConfig {
            blocks: vec![
                BlockConfig::new(32)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
                BlockConfig::new(64)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
            ],
            layers_per_block: 1,
            norm_num_groups: 32,
            cross_attention_dim: 64,
            ..Default::default()
        };

        let unet = config.init::<TestBackend>(4, 4, &device);

        // Input: [batch=1, channels=4, height=32, width=32]
        let xs: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 32, 32], &device);
        // Encoder hidden states: [batch=1, seq_len=8, dim=64]
        let encoder_hidden_states: Tensor<TestBackend, 3> = Tensor::zeros([1, 8, 64], &device);

        let output = unet.forward(xs, 1.0, encoder_hidden_states);

        // Output should have same spatial dimensions as input
        assert_eq!(output.shape(), Shape::from([1, 4, 32, 32]));
    }

    /// Test UNet2D forward with fixed weights matches diffusers-rs
    /// Reference values from diffusers-rs v0.3.1
    #[test]
    fn test_unet2d_fixed_weights_matches_diffusers_rs() {
        let device = Default::default();

        // Create a small UNet with same config as diffusers-rs test
        let config = UNet2DConditionModelConfig {
            blocks: vec![
                BlockConfig::new(32)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
                BlockConfig::new(64)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
            ],
            layers_per_block: 1,
            norm_num_groups: 32,
            cross_attention_dim: 64,
            ..Default::default()
        };

        let unet = config.init::<TestBackend>(4, 4, &device);

        // Set weights to 0.1, biases to 0.0 (matching diffusers-rs test)
        let unet = set_weights_and_biases(unet, 0.1, 0.0, &device);

        // Input: normalized arange values for reproducibility
        let input_data: Vec<f32> = (0..(4 * 32 * 32))
            .map(|i| i as f32 / (4.0 * 32.0 * 32.0))
            .collect();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(input_data.as_slice()), &device);
        let xs: Tensor<TestBackend, 4> = xs.reshape([1, 4, 32, 32]);

        // Encoder hidden states: normalized values
        let enc_data: Vec<f32> = (0..(8 * 64)).map(|i| i as f32 / (8.0 * 64.0)).collect();
        let encoder_hidden_states: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(enc_data.as_slice()), &device);
        let encoder_hidden_states: Tensor<TestBackend, 3> =
            encoder_hidden_states.reshape([1, 8, 64]);

        let output = unet.forward(xs, 1.0, encoder_hidden_states);

        // Verify output shape
        assert_eq!(output.shape(), Shape::from([1, 4, 32, 32]));

        // Get result values
        let result_flat = output.clone().flatten::<1>(0, 3);
        let result_data = result_flat.to_data();
        let result_vec: Vec<f32> = result_data.to_vec().unwrap();

        // Reference values from diffusers-rs v0.3.1
        let expected_first_16 = [
            -1.7083509_f32,
            -2.3742673,
            -1.9932699,
            -1.836023,
            -1.7530675,
            -1.7501539,
            -1.7482271,
            -1.7476634,
            -1.7470942,
            -1.7467214,
            -1.7462531,
            -1.7458805,
            -1.7454054,
            -1.7450254,
            -1.7445545,
            -1.7441819,
        ];

        for (i, (actual, expected)) in result_vec
            .iter()
            .take(16)
            .zip(expected_first_16.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-3,
                "Mismatch at index {}: actual={}, expected={}",
                i,
                actual,
                expected
            );
        }

        // Check overall mean (reference: 0.1854093372821808)
        let mean = output.clone().mean().into_scalar();
        let expected_mean = 0.18540934_f32;
        assert!(
            (mean - expected_mean).abs() < 1e-4,
            "Mean mismatch: actual={}, expected={}",
            mean,
            expected_mean
        );
    }
}
