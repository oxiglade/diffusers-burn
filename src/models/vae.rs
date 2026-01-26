//! # Variational Auto-Encoder (VAE) Models.
//!
//! Auto-encoder models compress their input to a usually smaller latent space
//! before expanding it back to its original shape. This results in the latent values
//! compressing the original information.

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, PaddingConfig2d};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use alloc::vec;
use alloc::vec::Vec;

use super::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig,
    UpDecoderBlock2D, UpDecoderBlock2DConfig,
};

/// Configuration for the VAE Encoder.
#[derive(Config, Debug)]
pub struct EncoderConfig {
    /// Number of input channels (e.g., 3 for RGB images).
    in_channels: usize,
    /// Number of output channels (latent channels).
    out_channels: usize,
    /// Output channels for each block.
    block_out_channels: Vec<usize>,
    /// Number of resnet layers per block.
    #[config(default = 2)]
    layers_per_block: usize,
    /// Number of groups for group normalization.
    #[config(default = 32)]
    norm_num_groups: usize,
    /// Whether to output double channels for mean and logvar.
    #[config(default = true)]
    double_z: bool,
}

/// VAE Encoder - compresses images to latent space.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv_in: Conv2d<B>,
    down_blocks: Vec<DownEncoderBlock2D<B>>,
    mid_block: UNetMidBlock2D<B>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl EncoderConfig {
    /// Initialize the Encoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let conv_in = Conv2dConfig::new([self.in_channels, self.block_out_channels[0]], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let mut down_blocks = vec![];
        for index in 0..self.block_out_channels.len() {
            let out_channels = self.block_out_channels[index];
            let in_channels = if index > 0 {
                self.block_out_channels[index - 1]
            } else {
                self.block_out_channels[0]
            };
            let is_final = index + 1 == self.block_out_channels.len();

            let down_block = DownEncoderBlock2DConfig::new(in_channels, out_channels)
                .with_n_layers(self.layers_per_block)
                .with_resnet_eps(1e-6)
                .with_resnet_groups(self.norm_num_groups)
                .with_add_downsample(!is_final)
                .with_downsample_padding(0)
                .init(device);

            down_blocks.push(down_block);
        }

        let last_block_out_channels = *self.block_out_channels.last().unwrap();

        let mid_block = UNetMidBlock2DConfig::new(last_block_out_channels)
            .with_resnet_eps(1e-6)
            .with_output_scale_factor(1.0)
            .with_attn_num_head_channels(None)
            .with_resnet_groups(Some(self.norm_num_groups))
            .with_n_layers(1)
            .init(device);

        let conv_norm_out = GroupNormConfig::new(self.norm_num_groups, last_block_out_channels)
            .with_epsilon(1e-6)
            .init(device);

        let conv_out_channels = if self.double_z {
            2 * self.out_channels
        } else {
            self.out_channels
        };

        let conv_out = Conv2dConfig::new([last_block_out_channels, conv_out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Encoder {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        }
    }
}

impl<B: Backend> Encoder<B> {
    /// Forward pass through the encoder.
    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut xs = self.conv_in.forward(xs);

        for down_block in self.down_blocks.iter() {
            xs = down_block.forward(xs);
        }

        let xs = self.mid_block.forward(xs, None);
        let xs = self.conv_norm_out.forward(xs);
        let xs = silu(xs);
        self.conv_out.forward(xs)
    }
}

/// Configuration for the VAE Decoder.
#[derive(Config, Debug)]
pub struct DecoderConfig {
    /// Number of input channels (latent channels).
    in_channels: usize,
    /// Number of output channels (e.g., 3 for RGB images).
    out_channels: usize,
    /// Output channels for each block.
    block_out_channels: Vec<usize>,
    /// Number of resnet layers per block.
    #[config(default = 2)]
    layers_per_block: usize,
    /// Number of groups for group normalization.
    #[config(default = 32)]
    norm_num_groups: usize,
}

/// VAE Decoder - expands latent space back to images.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv_in: Conv2d<B>,
    up_blocks: Vec<UpDecoderBlock2D<B>>,
    mid_block: UNetMidBlock2D<B>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl DecoderConfig {
    /// Initialize the Decoder.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let n_block_out_channels = self.block_out_channels.len();
        let last_block_out_channels = *self.block_out_channels.last().unwrap();

        let conv_in = Conv2dConfig::new([self.in_channels, last_block_out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let mid_block = UNetMidBlock2DConfig::new(last_block_out_channels)
            .with_resnet_eps(1e-6)
            .with_output_scale_factor(1.0)
            .with_attn_num_head_channels(None)
            .with_resnet_groups(Some(self.norm_num_groups))
            .with_n_layers(1)
            .init(device);

        let mut up_blocks = vec![];
        let reversed_block_out_channels: Vec<_> =
            self.block_out_channels.iter().copied().rev().collect();

        for index in 0..n_block_out_channels {
            let out_channels = reversed_block_out_channels[index];
            let in_channels = if index > 0 {
                reversed_block_out_channels[index - 1]
            } else {
                reversed_block_out_channels[0]
            };
            let is_final = index + 1 == n_block_out_channels;

            let up_block = UpDecoderBlock2DConfig::new(in_channels, out_channels)
                .with_n_layers(self.layers_per_block + 1)
                .with_resnet_eps(1e-6)
                .with_resnet_groups(self.norm_num_groups)
                .with_add_upsample(!is_final)
                .init(device);

            up_blocks.push(up_block);
        }

        let conv_norm_out = GroupNormConfig::new(self.norm_num_groups, self.block_out_channels[0])
            .with_epsilon(1e-6)
            .init(device);

        let conv_out = Conv2dConfig::new([self.block_out_channels[0], self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Decoder {
            conv_in,
            up_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        }
    }
}

impl<B: Backend> Decoder<B> {
    /// Forward pass through the decoder.
    pub fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let xs = self.conv_in.forward(xs);
        let mut xs = self.mid_block.forward(xs, None);

        for up_block in self.up_blocks.iter() {
            xs = up_block.forward(xs);
        }

        let xs = self.conv_norm_out.forward(xs);
        let xs = silu(xs);
        self.conv_out.forward(xs)
    }
}

/// Diagonal Gaussian Distribution for VAE latent space.
///
/// Represents the posterior distribution q(z|x) as a diagonal Gaussian
/// with learned mean and variance.
pub struct DiagonalGaussianDistribution<B: Backend> {
    mean: Tensor<B, 4>,
    std: Tensor<B, 4>,
}

impl<B: Backend> DiagonalGaussianDistribution<B> {
    /// Create a new distribution from the encoder output parameters.
    ///
    /// The parameters tensor is expected to have shape [batch, 2*latent_channels, height, width]
    /// where the first half contains the mean and the second half contains the log variance.
    pub fn new(parameters: Tensor<B, 4>) -> Self {
        // Split along channel dimension
        let [batch, channels, height, width] = parameters.dims();
        let half_channels = channels / 2;

        // Get mean (first half of channels)
        let mean = parameters
            .clone()
            .slice([0..batch, 0..half_channels, 0..height, 0..width]);

        // Get logvar (second half of channels)
        let logvar = parameters.slice([0..batch, half_channels..channels, 0..height, 0..width]);

        // std = exp(0.5 * logvar)
        let std = (logvar * 0.5).exp();

        DiagonalGaussianDistribution { mean, std }
    }

    /// Sample from the distribution using the reparameterization trick.
    ///
    /// z = mean + std * epsilon, where epsilon ~ N(0, 1)
    pub fn sample(&self) -> Tensor<B, 4> {
        let noise: Tensor<B, 4> = Tensor::random(
            self.mean.shape(),
            Distribution::Normal(0.0, 1.0),
            &self.mean.device(),
        );
        self.mean.clone() + self.std.clone() * noise
    }

    /// Get the mean of the distribution.
    pub fn mean(&self) -> Tensor<B, 4> {
        self.mean.clone()
    }

    /// Get the mode of the distribution (same as mean for Gaussian).
    pub fn mode(&self) -> Tensor<B, 4> {
        self.mean.clone()
    }
}

/// Configuration for the AutoEncoder KL model.
#[derive(Config, Debug)]
pub struct AutoEncoderKLConfig {
    /// Output channels for each block.
    #[config(default = "vec![128, 256, 512, 512]")]
    pub block_out_channels: Vec<usize>,
    /// Number of resnet layers per block.
    #[config(default = 2)]
    pub layers_per_block: usize,
    /// Number of latent channels.
    #[config(default = 4)]
    pub latent_channels: usize,
    /// Number of groups for group normalization.
    #[config(default = 32)]
    pub norm_num_groups: usize,
}

/// AutoEncoder with KL divergence loss (VAE).
///
/// This model compresses images to a latent space and can reconstruct them.
/// It's used in Stable Diffusion to work in a compressed latent space.
#[derive(Module, Debug)]
pub struct AutoEncoderKL<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
    quant_conv: Conv2d<B>,
    post_quant_conv: Conv2d<B>,
}

impl AutoEncoderKLConfig {
    /// Initialize the AutoEncoderKL model.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input image channels (e.g., 3 for RGB)
    /// * `out_channels` - Number of output image channels (e.g., 3 for RGB)
    /// * `device` - The device to create the model on
    pub fn init<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &B::Device,
    ) -> AutoEncoderKL<B> {
        let encoder = EncoderConfig::new(
            in_channels,
            self.latent_channels,
            self.block_out_channels.clone(),
        )
        .with_layers_per_block(self.layers_per_block)
        .with_norm_num_groups(self.norm_num_groups)
        .with_double_z(true)
        .init(device);

        let decoder = DecoderConfig::new(
            self.latent_channels,
            out_channels,
            self.block_out_channels.clone(),
        )
        .with_layers_per_block(self.layers_per_block)
        .with_norm_num_groups(self.norm_num_groups)
        .init(device);

        // 1x1 convolutions for quantization
        let quant_conv =
            Conv2dConfig::new([2 * self.latent_channels, 2 * self.latent_channels], [1, 1])
                .init(device);

        let post_quant_conv =
            Conv2dConfig::new([self.latent_channels, self.latent_channels], [1, 1]).init(device);

        AutoEncoderKL {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
        }
    }
}

impl<B: Backend> AutoEncoderKL<B> {
    /// Encode an image to the latent space distribution.
    ///
    /// Returns a DiagonalGaussianDistribution that can be sampled from.
    pub fn encode(&self, xs: Tensor<B, 4>) -> DiagonalGaussianDistribution<B> {
        let parameters = self.encoder.forward(xs);
        let parameters = self.quant_conv.forward(parameters);
        DiagonalGaussianDistribution::new(parameters)
    }

    /// Decode latent vectors back to images.
    pub fn decode(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let xs = self.post_quant_conv.forward(xs);
        self.decoder.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use alloc::string::{String, ToString};
    use burn::module::Module;
    use burn::tensor::{Shape, TensorData};

    use burn::module::{ModuleMapper, Param};

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
        vae: AutoEncoderKL<B>,
        weight_val: f32,
        bias_val: f32,
        device: &B::Device,
    ) -> AutoEncoderKL<B> {
        let mut mapper = WeightBiasMapper::<B> {
            weight_val,
            bias_val,
            device,
            current_field: String::new(),
        };
        vae.map(&mut mapper)
    }

    #[test]
    fn test_encoder_output_shape() {
        let device = Default::default();

        let encoder = EncoderConfig::new(3, 4, vec![32, 64])
            .with_layers_per_block(1)
            .with_norm_num_groups(32)
            .with_double_z(true)
            .init::<TestBackend>(&device);

        // Input: [batch=1, channels=3, height=64, width=64]
        let xs: Tensor<TestBackend, 4> = Tensor::zeros([1, 3, 64, 64], &device);
        let output = encoder.forward(xs);

        // Output should have 2*latent_channels due to double_z
        // Spatial dimensions reduced by factor of 2 for each downsample (1 downsample here)
        // 64 -> 32
        assert_eq!(output.shape(), Shape::from([1, 8, 32, 32]));
    }

    #[test]
    fn test_decoder_output_shape() {
        let device = Default::default();

        let decoder = DecoderConfig::new(4, 3, vec![32, 64])
            .with_layers_per_block(1)
            .with_norm_num_groups(32)
            .init::<TestBackend>(&device);

        // Input: [batch=1, channels=4, height=32, width=32]
        let xs: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 32, 32], &device);
        let output = decoder.forward(xs);

        // Output should upsample back
        // 32 -> 64
        assert_eq!(output.shape(), Shape::from([1, 3, 64, 64]));
    }

    #[test]
    fn test_diagonal_gaussian_distribution() {
        let device = Default::default();

        // Parameters with 8 channels (4 for mean, 4 for logvar)
        let parameters: Tensor<TestBackend, 4> = Tensor::zeros([1, 8, 4, 4], &device);
        let dist = DiagonalGaussianDistribution::new(parameters);

        let sample = dist.sample();
        assert_eq!(sample.shape(), Shape::from([1, 4, 4, 4]));

        let mean = dist.mean();
        assert_eq!(mean.shape(), Shape::from([1, 4, 4, 4]));
    }

    #[test]
    fn test_autoencoder_kl_shapes() {
        let device = Default::default();

        let vae = AutoEncoderKLConfig::new()
            .with_block_out_channels(vec![32, 64])
            .with_layers_per_block(1)
            .with_latent_channels(4)
            .with_norm_num_groups(32)
            .init::<TestBackend>(3, 3, &device);

        // Input image: [batch=1, channels=3, height=64, width=64]
        let xs: Tensor<TestBackend, 4> = Tensor::zeros([1, 3, 64, 64], &device);

        // Encode
        let dist = vae.encode(xs);
        let latent = dist.sample();

        // Latent should be [1, 4, 32, 32] (4 channels, 2x downsampled spatially)
        assert_eq!(latent.shape(), Shape::from([1, 4, 32, 32]));

        // Decode
        let reconstructed = vae.decode(latent);

        // Reconstructed should match input shape
        assert_eq!(reconstructed.shape(), Shape::from([1, 3, 64, 64]));
    }

    /// Test VAE decode with fixed weights matches diffusers-rs
    /// Reference values from diffusers-rs v0.3.1
    #[test]
    fn test_vae_decode_fixed_weights_matches_diffusers_rs() {
        let device = Default::default();

        // Create VAE with same config as diffusers-rs test
        let vae = AutoEncoderKLConfig::new()
            .with_block_out_channels(vec![32, 64])
            .with_layers_per_block(1)
            .with_latent_channels(4)
            .with_norm_num_groups(32)
            .init::<TestBackend>(3, 3, &device);

        // Set weights to 0.1, biases to 0.0 (matching diffusers-rs test)
        let vae = set_weights_and_biases(vae, 0.1, 0.0, &device);

        // Input: arange(4*16*16).reshape([1, 4, 16, 16]) / (4*16*16)
        let input_data: Vec<f32> = (0..(4 * 16 * 16))
            .map(|i| i as f32 / (4.0 * 16.0 * 16.0))
            .collect();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(input_data.as_slice()), &device);
        let xs: Tensor<TestBackend, 4> = xs.reshape([1, 4, 16, 16]);

        let decoded = vae.decode(xs);

        // Verify output shape
        assert_eq!(decoded.shape(), Shape::from([1, 3, 32, 32]));

        // Get result values
        let result_flat = decoded.clone().flatten::<1>(0, 3);
        let result_data = result_flat.to_data();
        let result_vec: Vec<f32> = result_data.to_vec().unwrap();

        // Reference values from diffusers-rs (all weights=0.1, biases=0.0)
        let expected_first_16 = [
            -0.42073298_f32,
            -0.689827,
            -0.8111322,
            -0.85588384,
            -0.884616,
            -0.8894976,
            -0.8884052,
            -0.8844389,
            -0.88121814,
            -0.8796178,
            -0.8778493,
            -0.8766395,
            -0.87531906,
            -0.8743709,
            -0.87319934,
            -0.8722713,
        ];

        for (i, (actual, expected)) in result_vec
            .iter()
            .take(16)
            .zip(expected_first_16.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-4,
                "Mismatch at index {}: actual={}, expected={}",
                i,
                actual,
                expected
            );
        }

        // Check overall mean matches
        let mean = decoded.mean().into_scalar();
        assert!(
            (mean - 0.09089245647192001).abs() < 1e-4,
            "Mean mismatch: actual={}, expected=0.09089245647192001",
            mean
        );
    }
}
