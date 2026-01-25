//! ControlNet Model
//!
//! ControlNet is a neural network structure to control diffusion models by adding extra conditions.
//! https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py

use alloc::vec;
use alloc::vec::Vec;
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::embeddings::{TimestepEmbedding, TimestepEmbeddingConfig, Timesteps};
use super::unet_2d::{BlockConfig, UNetDownBlock};
use super::unet_2d_blocks::{
    CrossAttnDownBlock2DConfig, DownBlock2DConfig, UNetMidBlock2DCrossAttn,
    UNetMidBlock2DCrossAttnConfig,
};

/// ControlNet conditioning embedding module.
///
/// Processes the conditioning image (e.g., canny edges, depth map) into
/// an embedding that can be added to the UNet features.
#[derive(Module, Debug)]
pub struct ControlNetConditioningEmbedding<B: Backend> {
    conv_in: Conv2d<B>,
    conv_out: Conv2d<B>,
    blocks: Vec<(Conv2d<B>, Conv2d<B>)>,
}

/// Configuration for ControlNetConditioningEmbedding.
#[derive(Config, Debug)]
pub struct ControlNetConditioningEmbeddingConfig {
    /// Output channels for the conditioning embedding.
    conditioning_embedding_channels: usize,
    /// Input channels from the conditioning image.
    #[config(default = 3)]
    conditioning_channels: usize,
    /// Channel progression for the embedding blocks.
    block_out_channels: Vec<usize>,
}

impl ControlNetConditioningEmbeddingConfig {
    /// Initialize the ControlNetConditioningEmbedding module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ControlNetConditioningEmbedding<B> {
        let b_channels = self.block_out_channels[0];
        let bl_channels = *self.block_out_channels.last().unwrap();

        let conv_in = Conv2dConfig::new([self.conditioning_channels, b_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv_out =
            Conv2dConfig::new([bl_channels, self.conditioning_embedding_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device);

        let blocks = (0..self.block_out_channels.len() - 1)
            .map(|i| {
                let channel_in = self.block_out_channels[i];
                let channel_out = self.block_out_channels[i + 1];

                let c1 = Conv2dConfig::new([channel_in, channel_in], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device);

                let c2 = Conv2dConfig::new([channel_in, channel_out], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device);

                (c1, c2)
            })
            .collect();

        ControlNetConditioningEmbedding {
            conv_in,
            conv_out,
            blocks,
        }
    }
}

impl<B: Backend> ControlNetConditioningEmbedding<B> {
    /// Forward pass through the conditioning embedding.
    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut xs = silu(self.conv_in.forward(xs));

        for (c1, c2) in &self.blocks {
            xs = silu(c1.forward(xs));
            xs = silu(c2.forward(xs));
        }

        self.conv_out.forward(xs)
    }
}

/// Configuration for the ControlNet model.
#[derive(Config, Debug)]
pub struct ControlNetConfig {
    /// Whether to flip sin to cos in timestep embedding.
    #[config(default = true)]
    pub flip_sin_to_cos: bool,
    /// Frequency shift for timestep embedding.
    #[config(default = 0.0)]
    pub freq_shift: f64,
    /// Configuration for each block.
    pub blocks: Vec<BlockConfig>,
    /// Output channels for conditioning embedding blocks.
    pub conditioning_embedding_out_channels: Vec<usize>,
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
    #[config(default = 768)]
    pub cross_attention_dim: usize,
    /// Whether to use linear projection in attention.
    #[config(default = false)]
    pub use_linear_projection: bool,
}

impl Default for ControlNetConfig {
    fn default() -> Self {
        Self {
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
            conditioning_embedding_out_channels: vec![16, 32, 96, 256],
            layers_per_block: 2,
            downsample_padding: 1,
            mid_block_scale_factor: 1.0,
            norm_num_groups: 32,
            norm_eps: 1e-5,
            cross_attention_dim: 768,
            use_linear_projection: false,
        }
    }
}

/// ControlNet model for adding spatial conditioning to diffusion models.
///
/// ControlNet copies the weights of a pretrained UNet and adds zero convolution
/// layers to inject conditioning information. The output consists of residuals
/// that are added to the corresponding layers of the UNet.
#[derive(Module, Debug)]
pub struct ControlNet<B: Backend> {
    conv_in: Conv2d<B>,
    controlnet_mid_block: Conv2d<B>,
    controlnet_cond_embedding: ControlNetConditioningEmbedding<B>,
    time_proj: Timesteps<B>,
    time_embedding: TimestepEmbedding<B>,
    down_blocks: Vec<UNetDownBlock<B>>,
    controlnet_down_blocks: Vec<Conv2d<B>>,
    mid_block: UNetMidBlock2DCrossAttn<B>,
}

impl ControlNetConfig {
    /// Initialize the ControlNet model.
    pub fn init<B: Backend>(&self, in_channels: usize, device: &B::Device) -> ControlNet<B> {
        let n_blocks = self.blocks.len();
        let b_channels = self.blocks[0].out_channels;
        let bl_channels = self.blocks.last().unwrap().out_channels;
        let bl_attention_head_dim = self.blocks.last().unwrap().attention_head_dim;
        let time_embed_dim = b_channels * 4;

        // Time embeddings
        let time_proj = Timesteps::new(b_channels, self.flip_sin_to_cos, self.freq_shift);
        let time_embedding = TimestepEmbeddingConfig::new(b_channels, time_embed_dim).init(device);

        // Input convolution
        let conv_in = Conv2dConfig::new([in_channels, b_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // ControlNet mid block (1x1 conv, zero initialized in practice)
        let controlnet_mid_block =
            Conv2dConfig::new([bl_channels, bl_channels], [1, 1]).init(device);

        // Conditioning embedding
        let controlnet_cond_embedding = ControlNetConditioningEmbeddingConfig::new(
            b_channels,
            self.conditioning_embedding_out_channels.clone(),
        )
        .init(device);

        // Down blocks
        let down_blocks: Vec<UNetDownBlock<B>> = (0..n_blocks)
            .map(|i| {
                let block_config = &self.blocks[i];
                let out_channels = block_config.out_channels;
                let attention_head_dim = block_config.attention_head_dim;

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

        // ControlNet down blocks (1x1 convs, zero initialized in practice)
        let mut controlnet_down_blocks =
            vec![Conv2dConfig::new([b_channels, b_channels], [1, 1]).init(device)];

        for (i, block) in self.blocks.iter().enumerate() {
            let out_channels = block.out_channels;
            for _ in 0..self.layers_per_block {
                controlnet_down_blocks
                    .push(Conv2dConfig::new([out_channels, out_channels], [1, 1]).init(device));
            }
            if i + 1 != self.blocks.len() {
                controlnet_down_blocks
                    .push(Conv2dConfig::new([out_channels, out_channels], [1, 1]).init(device));
            }
        }

        ControlNet {
            conv_in,
            controlnet_mid_block,
            controlnet_cond_embedding,
            time_proj,
            time_embedding,
            down_blocks,
            controlnet_down_blocks,
            mid_block,
        }
    }
}

impl<B: Backend> ControlNet<B> {
    /// Forward pass through ControlNet.
    ///
    /// # Arguments
    /// * `xs` - Noisy input tensor [batch, channels, height, width]
    /// * `timestep` - Current diffusion timestep
    /// * `encoder_hidden_states` - Encoder hidden states for cross-attention [batch, seq_len, dim]
    /// * `controlnet_cond` - Conditioning image [batch, 3, height, width]
    /// * `conditioning_scale` - Scale factor for the conditioning (typically 1.0)
    ///
    /// # Returns
    /// A tuple of (down_block_residuals, mid_block_residual) to be added to the UNet.
    pub fn forward(
        &self,
        xs: Tensor<B, 4>,
        timestep: f64,
        encoder_hidden_states: Tensor<B, 3>,
        controlnet_cond: Tensor<B, 4>,
        conditioning_scale: f64,
    ) -> (Vec<Tensor<B, 4>>, Tensor<B, 4>) {
        let [bsize, _channels, _height, _width] = xs.dims();
        let device = xs.device();

        // 1. Time embedding
        let timesteps: Tensor<B, 1> = Tensor::full([bsize], timestep as f32, &device);
        let emb = self.time_proj.forward(timesteps);
        let emb = self.time_embedding.forward(emb);

        // 2. Pre-process
        let xs = self.conv_in.forward(xs);
        let controlnet_cond = self.controlnet_cond_embedding.forward(controlnet_cond);
        let xs = xs + controlnet_cond;

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

        // 4. Mid block
        let xs = self
            .mid_block
            .forward(xs, Some(emb.clone()), Some(encoder_hidden_states.clone()));

        // 5. ControlNet blocks - apply 1x1 convs and scale
        let controlnet_down_block_res_xs: Vec<Tensor<B, 4>> = self
            .controlnet_down_blocks
            .iter()
            .enumerate()
            .map(|(i, block)| block.forward(down_block_res_xs[i].clone()) * conditioning_scale)
            .collect();

        let mid_block_res = self.controlnet_mid_block.forward(xs) * conditioning_scale;

        (controlnet_down_block_res_xs, mid_block_res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::module::{Module, ModuleMapper, Param};
    use burn::tensor::{Shape, TensorData};

    #[test]
    fn test_controlnet_conditioning_embedding_shape() {
        let device = Default::default();

        let config = ControlNetConditioningEmbeddingConfig::new(320, vec![16, 32, 96, 256]);
        let embedding = config.init::<TestBackend>(&device);

        // Input: [batch=1, channels=3, height=64, width=64]
        let xs: Tensor<TestBackend, 4> = Tensor::zeros([1, 3, 64, 64], &device);
        let output = embedding.forward(xs);

        // Output should have 320 channels and be downsampled by 2^3 = 8
        // 64 / 8 = 8
        assert_eq!(output.shape(), Shape::from([1, 320, 8, 8]));
    }

    #[test]
    fn test_controlnet_output_shape() {
        let device = Default::default();

        // Create a small ControlNet for testing
        // Note: conditioning_embedding_out_channels length determines downsampling factor
        // With [16] (1 element), there's no downsampling in the conditioning embedding
        let config = ControlNetConfig {
            blocks: vec![
                BlockConfig::new(32)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
                BlockConfig::new(64)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
            ],
            conditioning_embedding_out_channels: vec![16], // Single element = no downsampling
            layers_per_block: 1,
            norm_num_groups: 32,
            cross_attention_dim: 64,
            ..Default::default()
        };

        let controlnet = config.init::<TestBackend>(4, &device);

        // Input: [batch=1, channels=4, height=32, width=32]
        let xs: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 32, 32], &device);
        // Encoder hidden states: [batch=1, seq_len=8, dim=64]
        let encoder_hidden_states: Tensor<TestBackend, 3> = Tensor::zeros([1, 8, 64], &device);
        // Conditioning image: [batch=1, channels=3, height=32, width=32]
        let controlnet_cond: Tensor<TestBackend, 4> = Tensor::zeros([1, 3, 32, 32], &device);

        let (down_residuals, mid_residual) =
            controlnet.forward(xs, 1.0, encoder_hidden_states, controlnet_cond, 1.0);

        // Should have residuals for each down block output
        // With 2 blocks and 1 layer each, plus initial conv_in:
        // - 1 from conv_in (32 channels)
        // - 1 from block 0 resnet (32 channels)
        // - 1 from block 0 downsample (32 channels)
        // - 1 from block 1 resnet (64 channels)
        // Total: 4 residuals
        assert_eq!(down_residuals.len(), 4);

        // Mid block residual should match mid block channels
        assert_eq!(mid_residual.dims()[1], 64);
    }

    /// A ModuleMapper that sets weights to one value and biases to another.
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

            let is_bias = self.current_field.contains("bias") || self.current_field == "beta";
            let val = if is_bias {
                self.bias_val
            } else {
                self.weight_val
            };

            Param::initialized(tensor.id, Tensor::full(dims, val, self.device))
        }
    }

    fn set_weights_and_biases<B: Backend>(
        controlnet: ControlNet<B>,
        weight_val: f32,
        bias_val: f32,
        device: &B::Device,
    ) -> ControlNet<B> {
        let mut mapper = WeightBiasMapper::<B> {
            weight_val,
            bias_val,
            device,
            current_field: String::new(),
        };
        controlnet.map(&mut mapper)
    }

    /// Test ControlNet with fixed weights matches diffusers-rs
    #[test]
    fn test_controlnet_fixed_weights_matches_diffusers_rs() {
        let device = Default::default();

        // Create a small ControlNet matching the diffusers-rs test config
        let config = ControlNetConfig {
            blocks: vec![
                BlockConfig::new(32)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
                BlockConfig::new(64)
                    .with_use_cross_attn(true)
                    .with_attention_head_dim(8),
            ],
            conditioning_embedding_out_channels: vec![16],
            layers_per_block: 1,
            norm_num_groups: 32,
            cross_attention_dim: 64,
            ..Default::default()
        };

        let controlnet = config.init::<TestBackend>(4, &device);

        // Set weights to 0.1, biases to 0.0
        let controlnet = set_weights_and_biases(controlnet, 0.1, 0.0, &device);

        // Input tensor [batch=1, channels=4, height=32, width=32]
        let input_data: Vec<f32> = (0..(4 * 32 * 32))
            .map(|i| i as f32 / (4.0 * 32.0 * 32.0))
            .collect();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(input_data.as_slice()), &device);
        let xs: Tensor<TestBackend, 4> = xs.reshape([1, 4, 32, 32]);

        // Encoder hidden states [batch=1, seq_len=8, dim=64]
        let enc_data: Vec<f32> = (0..(8 * 64)).map(|i| i as f32 / (8.0 * 64.0)).collect();
        let encoder_hidden_states: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(enc_data.as_slice()), &device);
        let encoder_hidden_states: Tensor<TestBackend, 3> =
            encoder_hidden_states.reshape([1, 8, 64]);

        // Conditioning image [batch=1, channels=3, height=32, width=32]
        let cond_data: Vec<f32> = (0..(3 * 32 * 32))
            .map(|i| i as f32 / (3.0 * 32.0 * 32.0))
            .collect();
        let controlnet_cond: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(cond_data.as_slice()), &device);
        let controlnet_cond: Tensor<TestBackend, 4> = controlnet_cond.reshape([1, 3, 32, 32]);

        let (down_residuals, mid_residual) =
            controlnet.forward(xs, 1.0, encoder_hidden_states, controlnet_cond, 1.0);

        // Reference values from diffusers-rs
        assert_eq!(down_residuals.len(), 4);

        // Down residual 0: shape=[1, 32, 32, 32], mean=51.52621078491211
        let mean0: f32 = down_residuals[0].clone().mean().into_scalar();
        assert!(
            (mean0 - 51.526210).abs() < 0.1,
            "Down residual 0 mean mismatch: actual={}, expected=51.526210",
            mean0
        );

        // Down residual 1: shape=[1, 32, 32, 32], mean=156.5426025390625
        let mean1: f32 = down_residuals[1].clone().mean().into_scalar();
        assert!(
            (mean1 - 156.54260).abs() < 0.1,
            "Down residual 1 mean mismatch: actual={}, expected=156.54260",
            mean1
        );

        // Down residual 2: shape=[1, 32, 16, 16], mean=4349.87353515625
        let mean2: f32 = down_residuals[2].clone().mean().into_scalar();
        assert!(
            (mean2 - 4349.8735).abs() < 1.0,
            "Down residual 2 mean mismatch: actual={}, expected=4349.8735",
            mean2
        );

        // Down residual 3: shape=[1, 64, 16, 16], mean=28677.99609375
        let mean3: f32 = down_residuals[3].clone().mean().into_scalar();
        assert!(
            (mean3 - 28677.996).abs() < 10.0,
            "Down residual 3 mean mismatch: actual={}, expected=28677.996",
            mean3
        );

        // Mid residual: shape=[1, 64, 16, 16], mean=29518.357421875
        let mid_mean: f32 = mid_residual.clone().mean().into_scalar();
        assert!(
            (mid_mean - 29518.357).abs() < 10.0,
            "Mid residual mean mismatch: actual={}, expected=29518.357",
            mid_mean
        );
    }
}
