//! ResNet Building Blocks
//!
//! Some Residual Network blocks used in UNet models.
//!
//! Denoising Diffusion Implicit Models, K. He and al, 2015.
//! https://arxiv.org/abs/1512.03385

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for a ResNet block.
#[derive(Config, Debug)]
pub struct ResnetBlock2DConfig {
    pub in_channels: usize,
    /// The number of output channels, defaults to the number of input channels.
    pub out_channels: Option<usize>,
    pub temb_channels: Option<usize>,
    /// The number of groups to use in group normalization.
    #[config(default = 32)]
    pub groups: usize,
    pub groups_out: Option<usize>,
    /// The epsilon to be used in the group normalization operations.
    #[config(default = 1e-6)]
    pub eps: f64,
    /// Whether to use a 2D convolution in the skip connection. When using None,
    /// such a convolution is used if the number of input channels is different from
    /// the number of output channels.
    pub use_in_shortcut: Option<bool>,
    // non_linearity: silu
    /// The final output is scaled by dividing by this value.
    #[config(default = 1.)]
    pub output_scale_factor: f64,
}

impl ResnetBlock2DConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResnetBlock2D<B> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);
        let norm1 = GroupNormConfig::new(self.groups, self.in_channels)
            .with_epsilon(self.eps)
            .init(device);
        let conv1 = Conv2dConfig::new([self.in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let groups_out = self.groups_out.unwrap_or(self.groups);
        let norm2 = GroupNormConfig::new(groups_out, out_channels)
            .with_epsilon(self.eps)
            .init(device);
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let use_in_shortcut = self
            .use_in_shortcut
            .unwrap_or(self.in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = Conv2dConfig::new([self.in_channels, out_channels], [1, 1]);
            Some(conv_cfg.init(device))
        } else {
            None
        };
        let time_emb_proj = self.temb_channels.map(|temb_channels| {
            let linear_cfg = LinearConfig::new(temb_channels, out_channels);
            linear_cfg.init(device)
        });

        ResnetBlock2D {
            norm1,
            conv1,
            norm2,
            conv2,
            time_emb_proj,
            conv_shortcut,
            output_scale_factor: self.output_scale_factor,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResnetBlock2D<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    time_emb_proj: Option<Linear<B>>,
    conv_shortcut: Option<Conv2d<B>>,
    output_scale_factor: f64,
}

impl<B: Backend> ResnetBlock2D<B> {
    pub fn forward(&self, xs: Tensor<B, 4>, temb: Option<Tensor<B, 2>>) -> Tensor<B, 4> {
        let shortcut_xs = match &self.conv_shortcut {
            Some(conv_shortcut) => conv_shortcut.forward(xs.clone()),
            None => xs.clone(),
        };

        let xs = self.norm1.forward(xs.clone());
        let xs = self.conv1.forward(silu(xs));
        let xs = match (temb, &self.time_emb_proj) {
            (Some(temb), Some(time_emb_proj)) => {
                time_emb_proj
                    .forward(silu(temb))
                    .unsqueeze_dim::<3>(3 - 1)
                    .unsqueeze_dim::<4>(4 - 1)
                    + xs.clone()
            }
            _ => xs.clone(),
        };
        let xs = self.norm2.forward(xs);
        let xs = silu(xs);
        let xs = self.conv2.forward(xs);
        (shortcut_xs + xs) / self.output_scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use alloc::vec::Vec;
    use burn::tensor::{Distribution, Shape, TensorData, Tolerance};

    #[test]
    fn test_resnet_block_2d_no_temb() {
        let device = Default::default();
        let block = ResnetBlock2DConfig::new(128).init::<TestBackend>(&device);
        let xs = Tensor::<TestBackend, 4>::random([2, 128, 64, 64], Distribution::Default, &device);
        let output = block.forward(xs, None);

        assert_eq!(output.shape(), Shape::from([2, 128, 64, 64]));
    }

    #[test]
    fn test_resnet_block_2d_with_temb() {
        let device = Default::default();
        let block = ResnetBlock2DConfig::new(128).init::<TestBackend>(&device);
        let xs = Tensor::<TestBackend, 4>::random([2, 128, 64, 64], Distribution::Default, &device);
        let temb = Tensor::<TestBackend, 2>::random([2, 128], Distribution::Default, &device);
        let output = block.forward(xs, Some(temb));

        assert_eq!(output.shape(), Shape::from([2, 128, 64, 64]));
    }

    /// Test SiLU activation matches diffusers-rs (tch silu)
    /// Reference values from diffusers-rs v0.3.1
    #[test]
    fn test_silu_matches_diffusers_rs() {
        let device = Default::default();
        let xs: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::from([-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
            &device,
        );

        let result = silu(xs);

        // Reference values from diffusers-rs: tensor.silu()
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                -0.23840584,
                -0.26894143,
                -0.18877034,
                0.0,
                0.31122968,
                0.7310586,
                1.761594,
            ]),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }

    /// Test GroupNorm matches diffusers-rs
    /// Reference values from diffusers-rs v0.3.1 with weight=1, bias=0
    #[test]
    fn test_group_norm_matches_diffusers_rs() {
        let device = Default::default();

        // Create GroupNorm: 2 groups, 4 channels
        let norm = GroupNormConfig::new(2, 4)
            .with_epsilon(1e-6)
            .init::<TestBackend>(&device);

        // Set weight to 1 and bias to 0 (default initialization)
        // GroupNorm in Burn initializes gamma=1, beta=0 by default

        // Input: [batch=1, channels=4, height=2, width=2] with sequential values 1-16
        let xs: Tensor<TestBackend, 1> = Tensor::from_data(
            TensorData::from([
                1.0f32, 2.0, 3.0, 4.0, // channel 0
                5.0, 6.0, 7.0, 8.0, // channel 1
                9.0, 10.0, 11.0, 12.0, // channel 2
                13.0, 14.0, 15.0, 16.0, // channel 3
            ]),
            &device,
        );
        let xs: Tensor<TestBackend, 4> = xs.reshape([1, 4, 2, 2]);

        let result = norm.forward(xs);

        // Reference values from diffusers-rs GroupNorm
        let result_flat = result.flatten::<1>(0, 3);
        result_flat.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                -1.5275251,
                -1.0910892,
                -0.65465355,
                -0.21821785,
                0.21821797,
                0.65465367,
                1.0910894,
                1.5275251,
                -1.5275252,
                -1.0910892,
                -0.65465355,
                -0.21821785,
                0.21821785,
                0.65465355,
                1.0910892,
                1.527525,
            ]),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }

    /// Helper function to set all weights in a Conv2d to a constant value
    fn set_conv2d_weights<B: Backend>(
        conv: Conv2d<B>,
        weight_val: f32,
        bias_val: f32,
        device: &B::Device,
    ) -> Conv2d<B> {
        let weight_shape = conv.weight.shape();
        let [out_ch, in_ch, kh, kw] = weight_shape.dims();

        // Use Param::map to transform the weight tensor
        let new_weight = conv
            .weight
            .map(|_| Tensor::full([out_ch, in_ch, kh, kw], weight_val, device));

        let new_bias = conv
            .bias
            .map(|b| b.map(|_| Tensor::full([out_ch], bias_val, device)));

        Conv2d {
            weight: new_weight,
            bias: new_bias,
            stride: conv.stride,
            kernel_size: conv.kernel_size,
            dilation: conv.dilation,
            groups: conv.groups,
            padding: conv.padding,
        }
    }

    /// Helper function to set GroupNorm weights
    fn set_group_norm_weights<B: Backend>(
        norm: GroupNorm<B>,
        gamma_val: f32,
        beta_val: f32,
        device: &B::Device,
    ) -> GroupNorm<B> {
        let num_channels = norm.num_channels;

        let new_gamma = norm
            .gamma
            .map(|g| g.map(|_| Tensor::full([num_channels], gamma_val, device)));

        let new_beta = norm
            .beta
            .map(|b| b.map(|_| Tensor::full([num_channels], beta_val, device)));

        GroupNorm {
            gamma: new_gamma,
            beta: new_beta,
            num_groups: norm.num_groups,
            num_channels: norm.num_channels,
            epsilon: norm.epsilon,
            affine: norm.affine,
        }
    }

    /// Helper function to set Linear weights
    fn set_linear_weights<B: Backend>(
        linear: Linear<B>,
        weight_val: f32,
        bias_val: f32,
        device: &B::Device,
    ) -> Linear<B> {
        let weight_shape = linear.weight.shape();
        let [d_input, d_output] = weight_shape.dims();

        let new_weight = linear
            .weight
            .map(|_| Tensor::full([d_input, d_output], weight_val, device));

        let new_bias = linear
            .bias
            .map(|b| b.map(|_| Tensor::full([d_output], bias_val, device)));

        Linear {
            weight: new_weight,
            bias: new_bias,
        }
    }

    /// Test ResnetBlock2D with fixed weights matches diffusers-rs
    /// Reference values from diffusers-rs v0.3.1
    #[test]
    fn test_resnet_block_2d_fixed_weights_matches_diffusers_rs() {
        let device = Default::default();

        // Create ResnetBlock2D: in_channels=4, out_channels=4, groups=2
        let config = ResnetBlock2DConfig::new(4)
            .with_out_channels(Some(4))
            .with_groups(2)
            .with_groups_out(Some(2))
            .with_eps(1e-6)
            .with_use_in_shortcut(Some(false));

        let mut block = config.init::<TestBackend>(&device);

        // Set all weights to 0.1 and biases to 0.0 to match diffusers-rs test
        // NOTE: diffusers-rs sets ALL "weight" params to 0.1, including GroupNorm gamma
        block.norm1 = set_group_norm_weights(block.norm1, 0.1, 0.0, &device);
        block.norm2 = set_group_norm_weights(block.norm2, 0.1, 0.0, &device);
        block.conv1 = set_conv2d_weights(block.conv1, 0.1, 0.0, &device);
        block.conv2 = set_conv2d_weights(block.conv2, 0.1, 0.0, &device);

        // Input: arange(64).reshape([1, 4, 4, 4]) / 64.0
        let input_data: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(input_data.as_slice()), &device);
        let xs: Tensor<TestBackend, 4> = xs.reshape([1, 4, 4, 4]);

        let result = block.forward(xs, None);

        // Reference values from diffusers-rs
        let result_flat = result.clone().flatten::<1>(0, 3);
        let result_data = result_flat.to_data();
        let result_vec: Vec<f32> = result_data.to_vec().unwrap();

        // Check first 16 values match diffusers-rs reference
        let expected_first_16 = [
            -0.087926514,
            -0.106717095,
            -0.07181215,
            -0.0077848956,
            -0.0029995441,
            0.005315691,
            0.054237127,
            0.10205648,
            0.1431311,
            0.20416035,
            0.25529763,
            0.25261912,
            0.2456759,
            0.31968236,
            0.35902193,
            0.33475745,
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
        let mean = result.mean().into_scalar();
        assert!(
            (mean - 0.499_919_65).abs() < 1e-4,
            "Mean mismatch: actual={}, expected=0.49991965",
            mean
        );
    }

    /// Test ResnetBlock2D with time embedding and fixed weights matches diffusers-rs
    /// Reference values from diffusers-rs v0.3.1
    #[test]
    fn test_resnet_block_2d_with_temb_fixed_weights_matches_diffusers_rs() {
        let device = Default::default();

        // Create ResnetBlock2D with time embedding: in_channels=4, out_channels=4, temb_channels=8
        let config = ResnetBlock2DConfig::new(4)
            .with_out_channels(Some(4))
            .with_temb_channels(Some(8))
            .with_groups(2)
            .with_groups_out(Some(2))
            .with_eps(1e-6)
            .with_use_in_shortcut(Some(false));

        let mut block = config.init::<TestBackend>(&device);

        // Set all weights to 0.1 and biases to 0.0 to match diffusers-rs test
        // NOTE: diffusers-rs sets ALL "weight" params to 0.1, including GroupNorm gamma
        block.norm1 = set_group_norm_weights(block.norm1, 0.1, 0.0, &device);
        block.norm2 = set_group_norm_weights(block.norm2, 0.1, 0.0, &device);
        block.conv1 = set_conv2d_weights(block.conv1, 0.1, 0.0, &device);
        block.conv2 = set_conv2d_weights(block.conv2, 0.1, 0.0, &device);
        block.time_emb_proj = block
            .time_emb_proj
            .map(|proj| set_linear_weights(proj, 0.1, 0.0, &device));

        // Input: arange(64).reshape([1, 4, 4, 4]) / 64.0
        let input_data: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(input_data.as_slice()), &device);
        let xs: Tensor<TestBackend, 4> = xs.reshape([1, 4, 4, 4]);

        // Time embedding: arange(8).reshape([1, 8]) / 8.0
        let temb_data: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let temb: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from(temb_data.as_slice()), &device);
        let temb: Tensor<TestBackend, 2> = temb.reshape([1, 8]);

        let result = block.forward(xs, Some(temb));

        // Reference values from diffusers-rs (same as without temb due to the specific input values)
        let result_flat = result.flatten::<1>(0, 3);
        let result_data = result_flat.to_data();
        let result_vec: Vec<f32> = result_data.to_vec().unwrap();

        // Check first 16 values match diffusers-rs reference
        let expected_first_16 = [
            -0.08792652,
            -0.10671712,
            -0.07181221,
            -0.007784918,
            -0.0029995516,
            0.005315639,
            0.0542371,
            0.10205645,
            0.14313108,
            0.2041603,
            0.2552976,
            0.2526191,
            0.24567588,
            0.3196823,
            0.3590219,
            0.33475742,
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
    }
}
