//! DDPM Scheduler
//!
//! Denoising Diffusion Probabilistic Models (DDPM) scheduler.
//! Based on the paper: https://arxiv.org/abs/2006.11239

use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Distribution, Tensor};

use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// Variance type for DDPM scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DDPMVarianceType {
    /// Fixed small variance.
    FixedSmall,
    /// Fixed small variance (log).
    FixedSmallLog,
    /// Fixed large variance.
    FixedLarge,
    /// Fixed large variance (log).
    FixedLargeLog,
    /// Learned variance.
    Learned,
}

impl Default for DDPMVarianceType {
    fn default() -> Self {
        Self::FixedSmall
    }
}

/// Configuration for the DDPM Scheduler.
#[derive(Debug, Clone)]
pub struct DDPMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Option to clip the predicted sample between -1 and 1 for numerical stability.
    pub clip_sample: bool,
    /// Option to clip the variance used when adding noise to the denoised sample.
    pub variance_type: DDPMVarianceType,
    /// Prediction type of the scheduler function.
    pub prediction_type: PredictionType,
    /// Number of diffusion steps used to train the model.
    pub train_timesteps: usize,
}

impl Default for DDPMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            clip_sample: false,
            variance_type: DDPMVarianceType::FixedSmall,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

/// DDPM Scheduler for diffusion models.
///
/// This scheduler implements the DDPM algorithm for denoising diffusion models.
pub struct DDPMScheduler {
    alphas_cumprod: Vec<f64>,
    init_noise_sigma: f64,
    timesteps: Vec<usize>,
    step_ratio: usize,
    /// The scheduler configuration.
    pub config: DDPMSchedulerConfig,
}

impl DDPMScheduler {
    /// Create a new DDPM Scheduler.
    ///
    /// # Arguments
    /// * `inference_steps` - Number of inference steps
    /// * `config` - Scheduler configuration
    /// * `device` - The device to create tensors on
    pub fn new<B: Backend>(
        inference_steps: usize,
        config: DDPMSchedulerConfig,
        device: &B::Device,
    ) -> Self {
        let betas: Vec<f64> = match config.beta_schedule {
            BetaSchedule::ScaledLinear => {
                let start = config.beta_start.sqrt();
                let end = config.beta_end.sqrt();
                let step = (end - start) / (config.train_timesteps - 1) as f64;
                (0..config.train_timesteps)
                    .map(|i| {
                        let v = start + step * i as f64;
                        v * v
                    })
                    .collect()
            }
            BetaSchedule::Linear => {
                let step =
                    (config.beta_end - config.beta_start) / (config.train_timesteps - 1) as f64;
                (0..config.train_timesteps)
                    .map(|i| config.beta_start + step * i as f64)
                    .collect()
            }
            BetaSchedule::SquaredcosCapV2 => {
                let betas_tensor: Tensor<B, 1> =
                    betas_for_alpha_bar(config.train_timesteps, 0.999, device);
                let data = betas_tensor.into_data();
                data.to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(|x| x as f64)
                    .collect()
            }
        };

        // alphas = 1 - betas
        let alphas: Vec<f64> = betas.iter().map(|b| 1.0 - b).collect();

        // alphas_cumprod = cumprod(alphas)
        let mut alphas_cumprod: Vec<f64> = Vec::with_capacity(config.train_timesteps);
        let mut cumprod = 1.0;
        for alpha in &alphas {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        // min(train_timesteps, inference_steps)
        let inference_steps = inference_steps.min(config.train_timesteps);

        // Timesteps: arange(0, inference_steps) * step_ratio, reversed
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps: Vec<usize> = (0..inference_steps).map(|s| s * step_ratio).rev().collect();

        Self {
            alphas_cumprod,
            init_noise_sigma: 1.0,
            timesteps,
            step_ratio,
            config,
        }
    }

    /// Compute the variance for a given timestep.
    fn get_variance(&self, timestep: usize) -> f64 {
        let prev_t = timestep as isize - self.step_ratio as isize;
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        let current_beta_t = 1.0 - alpha_prod_t / alpha_prod_t_prev;

        // For t > 0, compute predicted variance βt (see formula (6) and (7) from paper)
        let variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t;

        // Retrieve variance based on type
        match self.config.variance_type {
            DDPMVarianceType::FixedSmall => variance.max(1e-20),
            DDPMVarianceType::FixedSmallLog => {
                let variance = variance.max(1e-20).ln();
                (variance * 0.5).exp()
            }
            DDPMVarianceType::FixedLarge => current_beta_t,
            DDPMVarianceType::FixedLargeLog => current_beta_t.ln(),
            DDPMVarianceType::Learned => variance,
        }
    }

    /// Get the timesteps for the scheduler.
    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    /// Get the initial noise sigma value.
    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    /// Scale the model input (identity for DDPM).
    pub fn scale_model_input<B: Backend>(
        &self,
        sample: Tensor<B, 4>,
        _timestep: usize,
    ) -> Tensor<B, 4> {
        sample
    }

    /// Perform one step of the DDPM.
    pub fn step<B: Backend>(
        &self,
        model_output: &Tensor<B, 4>,
        timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let prev_t = timestep as isize - self.step_ratio as isize;

        // 1. Compute alphas, betas
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            1.0
        };
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        let current_alpha_t = alpha_prod_t / alpha_prod_t_prev;
        let current_beta_t = 1.0 - current_alpha_t;

        // 2. Compute predicted original sample from predicted noise (formula (15))
        let mut pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => {
                (sample.clone() - model_output.clone() * beta_prod_t.sqrt()) / alpha_prod_t.sqrt()
            }
            PredictionType::Sample => model_output.clone(),
            PredictionType::VPrediction => {
                sample.clone() * alpha_prod_t.sqrt() - model_output.clone() * beta_prod_t.sqrt()
            }
        };

        // 3. Clip predicted x_0
        if self.config.clip_sample {
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0);
        }

        // 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        // See formula (7) from paper
        let pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t;
        let current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t;

        // 5. Compute predicted previous sample µ_t (formula (7))
        let pred_prev_sample = pred_original_sample * pred_original_sample_coeff
            + sample.clone() * current_sample_coeff;

        // 6. Add noise
        if timestep > 0 {
            let device = sample.device();
            let variance_noise: Tensor<B, 4> = Tensor::random(
                model_output.shape(),
                Distribution::Normal(0.0, 1.0),
                &device,
            );

            let variance = if self.config.variance_type == DDPMVarianceType::FixedSmallLog {
                self.get_variance(timestep) * variance_noise
            } else {
                self.get_variance(timestep).sqrt() * variance_noise
            };
            pred_prev_sample + variance
        } else {
            pred_prev_sample
        }
    }

    /// Add noise to original samples.
    pub fn add_noise<B: Backend>(
        &self,
        original_samples: &Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: usize,
    ) -> Tensor<B, 4> {
        let sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt();
        let sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[timestep]).sqrt();

        original_samples.clone() * sqrt_alpha_prod + noise * sqrt_one_minus_alpha_prod
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::Shape;

    #[test]
    fn test_ddpm_scheduler_creation() {
        let device = Default::default();
        let config = DDPMSchedulerConfig::default();
        let scheduler = DDPMScheduler::new::<TestBackend>(20, config, &device);

        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.init_noise_sigma(), 1.0);
    }

    #[test]
    fn test_ddpm_timesteps() {
        let device = Default::default();
        let config = DDPMSchedulerConfig::default();
        let scheduler = DDPMScheduler::new::<TestBackend>(20, config, &device);

        let timesteps = scheduler.timesteps();
        // First timestep should be high, last should be 0
        assert!(timesteps[0] > timesteps[timesteps.len() - 1]);
        assert_eq!(timesteps[timesteps.len() - 1], 0);

        // Should be monotonically decreasing
        for i in 1..timesteps.len() {
            assert!(timesteps[i] < timesteps[i - 1]);
        }
    }

    #[test]
    fn test_ddpm_scale_model_input() {
        let device = Default::default();
        let config = DDPMSchedulerConfig::default();
        let scheduler = DDPMScheduler::new::<TestBackend>(20, config, &device);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        // DDPM doesn't scale input
        let scaled = scheduler.scale_model_input(sample.clone(), timestep);
        let diff: f32 = (scaled - sample).abs().mean().into_scalar();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_ddpm_step() {
        let device = Default::default();
        let config = DDPMSchedulerConfig::default();
        let scheduler = DDPMScheduler::new::<TestBackend>(20, config, &device);

        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);

        // Step at timestep 0 (no noise added)
        let result = scheduler.step(&model_output, 0, &sample);
        assert_eq!(result.shape(), Shape::from([1, 4, 8, 8]));

        // Result should be finite
        let result_data = result.into_data();
        let values: Vec<f32> = result_data.to_vec().unwrap();
        for v in &values {
            assert!(v.is_finite(), "Result contains non-finite values");
        }
    }

    /// Test DDPM scheduler values match diffusers-rs
    #[test]
    fn test_ddpm_matches_diffusers_rs() {
        let device = Default::default();
        let config = DDPMSchedulerConfig::default();
        let scheduler = DDPMScheduler::new::<TestBackend>(20, config, &device);

        // Reference values from diffusers-rs
        // Timesteps: [950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 0]
        let expected_timesteps = [
            950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150,
            100, 50, 0,
        ];

        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), expected_timesteps.len());
        for (i, (actual, expected)) in timesteps.iter().zip(expected_timesteps.iter()).enumerate() {
            assert_eq!(
                *actual, *expected,
                "Timestep mismatch at {}: actual={}, expected={}",
                i, actual, expected
            );
        }

        // Check init_noise_sigma (reference: 1.0)
        assert_eq!(scheduler.init_noise_sigma(), 1.0);

        // Check scale_model_input (reference mean: 1.0 - identity)
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];
        let scaled = scheduler.scale_model_input(sample, timestep);
        let scaled_mean: f32 = scaled.mean().into_scalar();
        assert!(
            (scaled_mean - 1.0).abs() < 1e-6,
            "scale_model_input mean mismatch: actual={}, expected=1.0",
            scaled_mean
        );

        // Check step at timestep 0 (reference mean: 1.0004253387451172)
        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let result = scheduler.step(&model_output, 0, &sample);
        let result_mean: f32 = result.mean().into_scalar();
        assert!(
            (result_mean as f64 - 1.0004253387451172).abs() < 1e-4,
            "step mean mismatch: actual={}, expected=1.0004253387451172",
            result_mean
        );

        // Check add_noise (reference mean: 1.0866814851760864 at timestep 950)
        let original: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let noise: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let noisy = scheduler.add_noise(&original, noise, 950);
        let noisy_mean: f32 = noisy.mean().into_scalar();
        assert!(
            (noisy_mean as f64 - 1.0866814851760864).abs() < 1e-4,
            "add_noise mean mismatch: actual={}, expected=1.0866814851760864",
            noisy_mean
        );
    }
}
