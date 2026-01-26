use crate::utils::pad_with_zeros;
use alloc::vec;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

#[derive(Config, Debug)]
pub struct TimestepEmbeddingConfig {
    channel: usize,
    time_embed_dim: usize,
}

#[derive(Module, Debug)]
pub struct TimestepEmbedding<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
}

impl TimestepEmbeddingConfig {
    /// Initialize a new [embedding](TimestepEmbedding) module.
    /// Uses activating function: "silu".
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimestepEmbedding<B> {
        let linear_1 = LinearConfig::new(self.channel, self.time_embed_dim).init(device);
        let linear_2 = LinearConfig::new(self.time_embed_dim, self.time_embed_dim).init(device);
        TimestepEmbedding { linear_1, linear_2 }
    }
}

impl<B: Backend> TimestepEmbedding<B> {
    pub fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let xs = silu(self.linear_1.forward(xs));
        self.linear_2.forward(xs)
    }
}

/// Computes sinusoidal timestep embeddings.
///
/// This is a pure function with no learnable parameters.
/// It generates positional embeddings for diffusion timesteps using sinusoidal encoding.
///
/// # Arguments
/// * `timesteps` - 1D tensor of timestep values
/// * `num_channels` - Number of embedding channels to produce
/// * `flip_sin_to_cos` - If true, output is [cos, sin]; if false, output is [sin, cos]
/// * `downscale_freq_shift` - Frequency shift applied to the exponent denominator
///
/// # Returns
/// A 2D tensor of shape [batch_size, num_channels] containing the timestep embeddings.
pub fn get_timestep_embedding<B: Backend>(
    timesteps: Tensor<B, 1>,
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
) -> Tensor<B, 2> {
    let half_dim = num_channels / 2;
    let exponent =
        Tensor::arange(0..half_dim as i64, &timesteps.device()).float() * -f64::ln(10000.);
    let exponent = exponent / (half_dim as f64 - downscale_freq_shift);
    let emb = exponent.exp();
    // emb = timesteps[:, None].float() * emb[None, :]
    let emb: Tensor<B, 2> = timesteps.unsqueeze_dim(1) * emb.unsqueeze();
    // Concatenate along the last dimension (-1 in PyTorch, which is dim 1 for 2D tensor)
    let emb: Tensor<B, 2> = if flip_sin_to_cos {
        Tensor::cat(vec![emb.clone().cos(), emb.clone().sin()], 1)
    } else {
        Tensor::cat(vec![emb.clone().sin(), emb.clone().cos()], 1)
    };

    if num_channels % 2 == 1 {
        pad_with_zeros(emb, 1, 0, 1)
    } else {
        emb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Shape, TensorData, Tolerance};

    /// Test get_timestep_embedding with even channels - validated against diffusers-rs v0.3.1
    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_timesteps_even_channels() {
        let device = Default::default();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from([1., 2., 3., 4.]), &device);

        let emb: Tensor<TestBackend, 2> = get_timestep_embedding(xs, 4, true, 0.);

        assert_eq!(emb.shape(), Shape::from([4, 4]));
        // Reference values from diffusers-rs v0.3.1:
        emb.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.5403023, 0.99995, 0.84147096, 0.009999833],
                [-0.41614684, 0.9998, 0.9092974, 0.019998666],
                [-0.9899925, 0.99955004, 0.14112, 0.0299955],
                [-0.6536436, 0.9992001, -0.7568025, 0.039989334],
            ]),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }

    /// Test get_timestep_embedding with odd channels (padding) - validated against diffusers-rs v0.3.1
    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_timesteps_odd_channels() {
        let device = Default::default();
        let xs: Tensor<TestBackend, 1> = Tensor::from_data(TensorData::from([1., 2., 3.]), &device);

        let emb: Tensor<TestBackend, 2> = get_timestep_embedding(xs, 5, true, 0.);

        assert_eq!(emb.shape(), Shape::from([3, 5]));
        // Reference values from diffusers-rs v0.3.1:
        emb.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.5403023, 0.99995, 0.84147096, 0.009999833, 0.0],
                [-0.41614684, 0.9998, 0.9092974, 0.019998666, 0.0],
                [-0.9899925, 0.99955004, 0.14112, 0.0299955, 0.0],
            ]),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }

    /// Test get_timestep_embedding with flip_sin_to_cos=false - validated against diffusers-rs v0.3.1
    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_timesteps_no_flip() {
        let device = Default::default();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from([1., 2., 3., 4.]), &device);

        let emb: Tensor<TestBackend, 2> = get_timestep_embedding(xs, 4, false, 0.);

        assert_eq!(emb.shape(), Shape::from([4, 4]));
        // Reference values from diffusers-rs v0.3.1 with flip_sin_to_cos=false:
        emb.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.84147096, 0.009999833, 0.5403023, 0.99995],
                [0.9092974, 0.019998666, -0.41614684, 0.9998],
                [0.14112, 0.0299955, -0.9899925, 0.99955004],
                [-0.7568025, 0.039989334, -0.6536436, 0.9992001],
            ]),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }

    /// Test get_timestep_embedding with downscale_freq_shift - validated against diffusers-rs v0.3.1
    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_timesteps_with_downscale() {
        let device = Default::default();
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from([1., 2., 3., 4.]), &device);

        let emb: Tensor<TestBackend, 2> = get_timestep_embedding(xs, 4, true, 1.0);

        assert_eq!(emb.shape(), Shape::from([4, 4]));
        // Reference values from diffusers-rs v0.3.1 with downscale_freq_shift=1.0:
        emb.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.5403023, 1.0, 0.84147096, 9.999999e-5],
                [-0.41614684, 1.0, 0.9092974, 0.00019999998],
                [-0.9899925, 0.99999994, 0.14112, 0.00029999996],
                [-0.6536436, 0.99999994, -0.7568025, 0.00039999996],
            ]),
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }
}
