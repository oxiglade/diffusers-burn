//! # Denoising Diffusion Implicit Models
//!
//! The Denoising Diffusion Implicit Models (DDIM) is a simple scheduler
//! similar to Denoising Diffusion Probabilistic Models (DDPM). The DDPM
//! generative process is the reverse of a Markovian process, DDIM generalizes
//! this to non-Markovian guidance.
//!
//! Denoising Diffusion Implicit Models, J. Song et al, 2020.
//! <https://arxiv.org/abs/2010.02502>

use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Tensor};

use super::{betas_for_alpha_bar, BetaSchedule, PredictionType};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// The configuration for the DDIM scheduler.
#[derive(Debug, Clone, Copy)]
pub struct DDIMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// The amount of noise to be added at each step.
    pub eta: f64,
    /// Adjust the indexes of the inference schedule by this value.
    pub steps_offset: usize,
    /// Prediction type of the scheduler function, one of `epsilon` (predicting
    /// the noise of the diffusion process), `sample` (directly predicting the noisy sample)
    /// or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
    pub prediction_type: PredictionType,
    /// Number of diffusion steps used to train the model.
    pub train_timesteps: usize,
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            eta: 0.0,
            steps_offset: 1,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

/// The DDIM scheduler.
#[derive(Debug, Clone)]
pub struct DDIMScheduler {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
    init_noise_sigma: f64,
    /// The configuration used to create this scheduler.
    pub config: DDIMSchedulerConfig,
}

impl DDIMScheduler {
    /// Creates a new DDIM scheduler given the number of steps to be
    /// used for inference as well as the number of steps that was used
    /// during training.
    pub fn new<B: Backend>(
        inference_steps: usize,
        config: DDIMSchedulerConfig,
        device: &B::Device,
    ) -> Self {
        let step_ratio = config.train_timesteps / inference_steps;

        // Generate timesteps in reverse order
        let timesteps: Vec<usize> = (0..inference_steps)
            .map(|s| s * step_ratio + config.steps_offset)
            .rev()
            .collect();

        // Compute betas based on schedule type
        let betas: Tensor<B, 1> = match config.beta_schedule {
            BetaSchedule::ScaledLinear => {
                // linspace of sqrt(beta_start) to sqrt(beta_end), then squared
                let start = config.beta_start.sqrt();
                let end = config.beta_end.sqrt();
                Tensor::from_floats(
                    linspace(start, end, config.train_timesteps).as_slice(),
                    device,
                )
                .powf_scalar(2.0)
            }
            BetaSchedule::Linear => Tensor::from_floats(
                linspace(config.beta_start, config.beta_end, config.train_timesteps).as_slice(),
                device,
            ),
            BetaSchedule::SquaredcosCapV2 => {
                betas_for_alpha_bar(config.train_timesteps, 0.999, device)
            }
        };

        // alphas = 1 - betas
        let alphas = betas.neg().add_scalar(1.0);

        // Compute cumulative product of alphas
        let alphas_cumprod = cumprod_vec::<B>(alphas);

        Self {
            alphas_cumprod,
            timesteps,
            step_ratio,
            init_noise_sigma: 1.0,
            config,
        }
    }

    /// Returns the timesteps for the scheduler.
    pub fn timesteps(&self) -> &[usize] {
        self.timesteps.as_slice()
    }

    /// Ensures interchangeability with schedulers that need to scale the denoising model input
    /// depending on the current timestep.
    pub fn scale_model_input<B: Backend>(
        &self,
        sample: Tensor<B, 4>,
        _timestep: usize,
    ) -> Tensor<B, 4> {
        sample
    }

    /// Performs a backward step during inference.
    pub fn step<B: Backend>(
        &self,
        model_output: &Tensor<B, 4>,
        timestep: usize,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        // Clamp timestep if needed
        let timestep = if timestep >= self.alphas_cumprod.len() {
            timestep - 1
        } else {
            timestep
        };

        // Calculate previous timestep
        let prev_timestep = if timestep > self.step_ratio {
            timestep - self.step_ratio
        } else {
            0
        };

        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = self.alphas_cumprod[prev_timestep];
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Compute predicted original sample and epsilon based on prediction type
        let (pred_original_sample, pred_epsilon) = match self.config.prediction_type {
            PredictionType::Epsilon => {
                // pred_original_sample = (sample - sqrt(beta_prod_t) * model_output) / sqrt(alpha_prod_t)
                let pred_original_sample = sample
                    .clone()
                    .sub(model_output.clone().mul_scalar(beta_prod_t.sqrt()))
                    .div_scalar(alpha_prod_t.sqrt());
                (pred_original_sample, model_output.clone())
            }
            PredictionType::VPrediction => {
                // pred_original_sample = sqrt(alpha_prod_t) * sample - sqrt(beta_prod_t) * model_output
                let pred_original_sample = sample
                    .clone()
                    .mul_scalar(alpha_prod_t.sqrt())
                    .sub(model_output.clone().mul_scalar(beta_prod_t.sqrt()));
                // pred_epsilon = sqrt(alpha_prod_t) * model_output + sqrt(beta_prod_t) * sample
                let pred_epsilon = model_output
                    .clone()
                    .mul_scalar(alpha_prod_t.sqrt())
                    .add(sample.clone().mul_scalar(beta_prod_t.sqrt()));
                (pred_original_sample, pred_epsilon)
            }
            PredictionType::Sample => {
                let pred_original_sample = model_output.clone();
                // pred_epsilon = (sample - sqrt(alpha_prod_t) * pred_original_sample) / sqrt(beta_prod_t)
                let pred_epsilon = sample
                    .clone()
                    .sub(pred_original_sample.clone().mul_scalar(alpha_prod_t.sqrt()))
                    .div_scalar(beta_prod_t.sqrt());
                (pred_original_sample, pred_epsilon)
            }
        };

        // Compute variance
        let variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.config.eta * variance.sqrt();

        // pred_sample_direction = sqrt(1 - alpha_prod_t_prev - std_dev_t^2) * pred_epsilon
        let pred_sample_direction =
            pred_epsilon.mul_scalar((1.0 - alpha_prod_t_prev - std_dev_t * std_dev_t).sqrt());

        // prev_sample = sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        let prev_sample = pred_original_sample
            .mul_scalar(alpha_prod_t_prev.sqrt())
            .add(pred_sample_direction);

        // Add noise if eta > 0
        if self.config.eta > 0.0 {
            let noise =
                Tensor::random_like(&prev_sample, burn::tensor::Distribution::Normal(0.0, 1.0));
            prev_sample.add(noise.mul_scalar(std_dev_t))
        } else {
            prev_sample
        }
    }

    /// Adds noise to original samples.
    pub fn add_noise<B: Backend>(
        &self,
        original: &Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: usize,
    ) -> Tensor<B, 4> {
        let timestep = if timestep >= self.alphas_cumprod.len() {
            timestep - 1
        } else {
            timestep
        };

        let sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt();
        let sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[timestep]).sqrt();

        // sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        original
            .clone()
            .mul_scalar(sqrt_alpha_prod)
            .add(noise.mul_scalar(sqrt_one_minus_alpha_prod))
    }

    /// Returns the initial noise sigma.
    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }
}

/// Creates a vector of linearly spaced values.
fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        return Vec::new();
    }
    if steps == 1 {
        return alloc::vec![start];
    }
    let step_size = (end - start) / (steps - 1) as f64;
    (0..steps).map(|i| start + step_size * i as f64).collect()
}

/// Computes cumulative product and returns as Vec<f64>.
fn cumprod_vec<B: Backend>(tensor: Tensor<B, 1>) -> Vec<f64> {
    let data = tensor.into_data();
    let values: Vec<f32> = data.to_vec().unwrap();

    let mut result = Vec::with_capacity(values.len());
    let mut acc = 1.0f64;
    for v in values {
        acc *= v as f64;
        result.push(acc);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::prelude::ElementConversion;

    #[test]
    fn test_linspace() {
        let result = linspace(0.0, 1.0, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[4] - 1.0).abs() < 1e-10);
        assert!((result[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ddim_scheduler_creation() {
        let device = Default::default();
        let config = DDIMSchedulerConfig::default();
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        // Check timesteps are in descending order
        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), 50);
        assert!(timesteps[0] > timesteps[timesteps.len() - 1]);

        // First timestep should be around 981 (49 * 20 + 1 with step_ratio=20, steps_offset=1)
        assert_eq!(timesteps[0], 981);

        // Last timestep should be 1 (0 * 20 + 1)
        assert_eq!(timesteps[timesteps.len() - 1], 1);
    }

    #[test]
    fn test_ddim_alphas_cumprod() {
        let device = Default::default();
        let config = DDIMSchedulerConfig::default();
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        // alphas_cumprod should have train_timesteps entries
        assert_eq!(scheduler.alphas_cumprod.len(), 1000);

        // First alpha_cumprod should be close to 1 (since beta_start is small)
        assert!(scheduler.alphas_cumprod[0] > 0.99);

        // Last alpha_cumprod should be small (accumulated product decreases)
        assert!(scheduler.alphas_cumprod[999] < 0.1);

        // Should be monotonically decreasing
        for i in 1..scheduler.alphas_cumprod.len() {
            assert!(scheduler.alphas_cumprod[i] < scheduler.alphas_cumprod[i - 1]);
        }
    }

    #[test]
    fn test_ddim_add_noise() {
        let device = Default::default();
        let config = DDIMSchedulerConfig::default();
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        let original: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 64, 64], &device);
        let noise: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 64, 64], &device);

        // With zero noise at timestep 0, result should be close to original * sqrt(alpha_cumprod[0])
        let result = scheduler.add_noise(&original, noise, 0);
        let expected_scale = scheduler.alphas_cumprod[0].sqrt();

        // Check that the result is scaled correctly
        let result_mean: f32 = result.mean().into_scalar().elem();
        assert!((result_mean as f64 - expected_scale).abs() < 1e-4);
    }

    #[test]
    fn test_ddim_step_epsilon_prediction() {
        let device = Default::default();
        let config = DDIMSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            ..Default::default()
        };
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        // Simple test: zero model output (no predicted noise) should return scaled sample
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);

        let timestep = scheduler.timesteps()[0]; // First timestep
        let result = scheduler.step(&model_output, timestep, &sample);

        // Result should not be NaN or Inf
        let result_data = result.into_data();
        let values: Vec<f32> = result_data.to_vec().unwrap();
        for v in &values {
            assert!(v.is_finite(), "Result contains non-finite values");
        }
    }

    #[test]
    fn test_linear_beta_schedule() {
        let device = <TestBackend as Backend>::Device::default();
        let config = DDIMSchedulerConfig {
            beta_schedule: BetaSchedule::Linear,
            ..Default::default()
        };
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        // With linear schedule, alphas_cumprod should still be monotonically decreasing
        for i in 1..scheduler.alphas_cumprod.len() {
            assert!(scheduler.alphas_cumprod[i] < scheduler.alphas_cumprod[i - 1]);
        }
    }

    #[test]
    fn test_init_noise_sigma() {
        let device = Default::default();
        let config = DDIMSchedulerConfig::default();
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        assert_eq!(scheduler.init_noise_sigma(), 1.0);
    }

    /// Test that alphas_cumprod values match diffusers-rs within acceptable tolerance.
    /// Reference values derived from diffusers-rs v0.3.1 using add_noise with zero noise.
    /// Note: Small differences (~1e-4) are expected due to f32 precision in tensor ops.
    #[test]
    fn test_alphas_cumprod_matches_diffusers_rs() {
        let device = Default::default();
        let config = DDIMSchedulerConfig::default();
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        // Expected values from diffusers-rs v0.3.1 with ScaledLinear beta schedule
        // beta_start=0.00085, beta_end=0.012, train_timesteps=1000
        // Values derived by running: scheduler.add_noise(ones, zeros, t) which gives sqrt(alpha_cumprod[t])
        let expected_first = 0.999149980057211; // alphas_cumprod[0]
        let expected_at_500 = 0.276332449694738; // alphas_cumprod[500]
        let expected_last = 0.004660095778424; // alphas_cumprod[999]

        assert!(
            (scheduler.alphas_cumprod[0] - expected_first).abs() < 1e-4,
            "alphas_cumprod[0]: expected {}, got {}",
            expected_first,
            scheduler.alphas_cumprod[0]
        );
        assert!(
            (scheduler.alphas_cumprod[500] - expected_at_500).abs() < 1e-4,
            "alphas_cumprod[500]: expected {}, got {}",
            expected_at_500,
            scheduler.alphas_cumprod[500]
        );
        assert!(
            (scheduler.alphas_cumprod[999] - expected_last).abs() < 1e-4,
            "alphas_cumprod[999]: expected {}, got {}",
            expected_last,
            scheduler.alphas_cumprod[999]
        );
    }

    /// Test step() produces correct output matching diffusers-rs.
    /// Reference values from diffusers-rs v0.3.1.
    #[test]
    fn test_step_matches_diffusers_rs() {
        let device = Default::default();
        let config = DDIMSchedulerConfig {
            eta: 0.0, // Deterministic (no noise added)
            ..Default::default()
        };
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        // Use a simple known input
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device);
        let model_output: Tensor<TestBackend, 4> =
            Tensor::ones([1, 1, 2, 2], &device).mul_scalar(0.5);

        // Step at timestep 981 (first timestep with 50 inference steps)
        // diffusers-rs returns: 1.061225771903992
        let result = scheduler.step(&model_output, 981, &sample);
        let result_data = result.into_data();
        let values: Vec<f32> = result_data.to_vec().unwrap();
        let first_val = values[0];

        let expected_981 = 1.061225771903992;
        assert!(
            (first_val as f64 - expected_981).abs() < 1e-3,
            "Step at 981: expected {}, got {}",
            expected_981,
            first_val
        );

        // Step at timestep 500
        // diffusers-rs returns: 1.019660353660583
        let result2 = scheduler.step(&model_output, 500, &sample);
        let result2_data = result2.into_data();
        let values2: Vec<f32> = result2_data.to_vec().unwrap();

        let expected_500 = 1.019660353660583;
        assert!(
            (values2[0] as f64 - expected_500).abs() < 1e-3,
            "Step at 500: expected {}, got {}",
            expected_500,
            values2[0]
        );
    }

    /// Test add_noise produces correct output matching diffusers-rs.
    /// Reference values from diffusers-rs v0.3.1.
    #[test]
    fn test_add_noise_matches_diffusers_rs() {
        let device = Default::default();
        let config = DDIMSchedulerConfig::default();
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        let original: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device);

        // Test at timestep 0
        // diffusers-rs returns: 1.028730034828186
        let noise0: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device);
        let result0 = scheduler.add_noise(&original, noise0, 0);
        let val0: f32 = result0.into_data().to_vec::<f32>().unwrap()[0];
        let expected_0 = 1.028730034828186;
        assert!(
            (val0 as f64 - expected_0).abs() < 1e-3,
            "add_noise at 0: expected {}, got {}",
            expected_0,
            val0
        );

        // Test at timestep 500
        // diffusers-rs returns: 1.376359820365906
        let noise500: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device);
        let result500 = scheduler.add_noise(&original, noise500, 500);
        let val500: f32 = result500.into_data().to_vec::<f32>().unwrap()[0];
        let expected_500 = 1.376359820365906;
        assert!(
            (val500 as f64 - expected_500).abs() < 1e-3,
            "add_noise at 500: expected {}, got {}",
            expected_500,
            val500
        );

        // Test at timestep 999
        // diffusers-rs returns: 1.065932154655457
        let noise999: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device);
        let result999 = scheduler.add_noise(&original, noise999, 999);
        let val999: f32 = result999.into_data().to_vec::<f32>().unwrap()[0];
        let expected_999 = 1.065932154655457;
        assert!(
            (val999 as f64 - expected_999).abs() < 1e-3,
            "add_noise at 999: expected {}, got {}",
            expected_999,
            val999
        );
    }

    /// Test V-prediction mode produces valid output.
    #[test]
    fn test_v_prediction_step() {
        let device = Default::default();
        let config = DDIMSchedulerConfig {
            prediction_type: PredictionType::VPrediction,
            eta: 0.0,
            ..Default::default()
        };
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device);
        let model_output: Tensor<TestBackend, 4> =
            Tensor::ones([1, 1, 2, 2], &device).mul_scalar(0.5);

        let result = scheduler.step(&model_output, 981, &sample);
        let result_data = result.into_data();
        let values: Vec<f32> = result_data.to_vec().unwrap();

        // Result should be finite and uniform
        for v in &values {
            assert!(v.is_finite(), "V-prediction result should be finite");
        }
    }

    /// Test Sample prediction mode produces valid output.
    #[test]
    fn test_sample_prediction_step() {
        let device = Default::default();
        let config = DDIMSchedulerConfig {
            prediction_type: PredictionType::Sample,
            eta: 0.0,
            ..Default::default()
        };
        let scheduler = DDIMScheduler::new::<TestBackend>(50, config, &device);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device);
        let model_output: Tensor<TestBackend, 4> =
            Tensor::ones([1, 1, 2, 2], &device).mul_scalar(0.5);

        let result = scheduler.step(&model_output, 981, &sample);
        let result_data = result.into_data();
        let values: Vec<f32> = result_data.to_vec().unwrap();

        // Result should be finite and uniform
        for v in &values {
            assert!(v.is_finite(), "Sample prediction result should be finite");
        }
    }
}
