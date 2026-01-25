use crate::utils::pad_with_zeros;
use alloc::vec;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use core::marker::PhantomData;

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
    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let xs = silu(self.linear_1.forward(xs));
        self.linear_2.forward(xs)
    }
}

#[derive(Module, Debug)]
pub struct Timesteps<B: Backend> {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
    _backend: PhantomData<B>,
}

impl<B: Backend> Timesteps<B> {
    pub fn new(num_channels: usize, flip_sin_to_cos: bool, downscale_freq_shift: f64) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,

            downscale_freq_shift,
            _backend: PhantomData,
        }
    }

    pub fn forward<const D1: usize, const D2: usize>(&self, xs: Tensor<B, D1>) -> Tensor<B, D2> {
        let half_dim = self.num_channels / 2;
        let exponent = Tensor::arange(0..half_dim as i64, &xs.device()).float() * -f64::ln(10000.);
        let exponent = exponent / (half_dim as f64 - self.downscale_freq_shift);
        let emb = exponent.exp();
        // emb = timesteps[:, None].float() * emb[None, :]
        let emb: Tensor<B, D2> = xs.unsqueeze_dim(D1) * emb.unsqueeze();
        let emb: Tensor<B, D2> = if self.flip_sin_to_cos {
            Tensor::cat(vec![emb.clone().cos(), emb.clone().sin()], D1)
        } else {
            Tensor::cat(vec![emb.clone().sin(), emb.clone().cos()], D1)
        };

        if self.num_channels % 2 == 1 {
            pad_with_zeros(emb, D1 - 1, 0, 1)
        } else {
            emb
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Shape, TensorData, Tolerance};

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_timesteps_even_channels() {
        let device = Default::default();
        let timesteps = Timesteps::<TestBackend>::new(4, true, 0.);
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from([1., 2., 3., 4.]), &device);

        let emb: Tensor<TestBackend, 2> = timesteps.forward(xs);

        assert_eq!(emb.shape(), Shape::from([4, 4]));
        emb.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.5403, 1.0000, 0.8415, 0.0100],
                [-0.4161, 0.9998, 0.9093, 0.0200],
                [-0.9900, 0.9996, 0.1411, 0.0300],
                [-0.6536, 0.9992, -0.7568, 0.0400],
            ]),
            Tolerance::rel_abs(1e-3, 1e-3),
        );
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_timesteps_odd_channels() {
        let device = Default::default();
        let timesteps = Timesteps::<TestBackend>::new(5, true, 0.);
        let xs: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::from([1., 2., 3., 4., 5.]), &device);

        let emb: Tensor<TestBackend, 2> = timesteps.forward(xs);

        assert_eq!(emb.shape(), Shape::from([6, 4]));
        emb.into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.5403, 1.0000, 0.8415, 0.0100],
                [-0.4161, 0.9998, 0.9093, 0.0200],
                [-0.9900, 0.9996, 0.1411, 0.0300],
                [-0.6536, 0.9992, -0.7568, 0.0400],
                [0.2837, 0.9988, -0.9589, 0.0500],
                [0.0000, 0.0000, 0.0000, 0.0000],
            ]),
            Tolerance::rel_abs(1e-3, 1e-3),
        );
    }
}
