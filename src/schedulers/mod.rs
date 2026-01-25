//! # Noise Schedulers
//!
//! Noise schedulers can be used to set the trade-off between
//! inference speed and quality.

use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Tensor};
use core::f64::consts::FRAC_PI_2;

pub mod ddim;

pub use ddim::{DDIMScheduler, DDIMSchedulerConfig};

/// This represents how beta ranges from its minimum value to the maximum
/// during training.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// Linear interpolation.
    Linear,
    /// Linear interpolation of the square root of beta.
    ScaledLinear,
    /// Glide cosine schedule
    SquaredcosCapV2,
}

/// The type of prediction the model makes.
#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    /// Predicting the noise of the diffusion process.
    Epsilon,
    /// See section 2.4 https://imagen.research.google/video/paper.pdf
    VPrediction,
    /// Directly predicting the noisy sample.
    Sample,
}

/// Create a beta schedule that discretizes the given alpha_t_bar function,
/// which defines the cumulative product of `(1-beta)` over time from `t = [0,1]`.
///
/// Contains a function `alpha_bar` that takes an argument `t` and transforms it
/// to the cumulative product of `(1-beta)` up to that part of the diffusion process.
pub fn betas_for_alpha_bar<B: Backend>(
    num_diffusion_timesteps: usize,
    max_beta: f64,
    device: &B::Device,
) -> Tensor<B, 1> {
    let alpha_bar = |time_step: usize| -> f64 {
        let t = (time_step as f64 + 0.008) / 1.008 * FRAC_PI_2;
        t.cos().powi(2)
    };

    let mut betas: Vec<f64> = Vec::with_capacity(num_diffusion_timesteps);
    for i in 0..num_diffusion_timesteps {
        let t1 = i / num_diffusion_timesteps;
        let t2 = (i + 1) / num_diffusion_timesteps;
        let beta = (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta);
        betas.push(beta);
    }

    Tensor::from_floats(betas.as_slice(), device)
}
