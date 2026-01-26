//! Euler Discrete Scheduler
//!
//! Euler scheduler (Algorithm 2) from Karras et al. (2022) https://arxiv.org/abs/2206.00364.
//! Based on the original k-diffusion implementation by Katherine Crowson:
//! https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51

use alloc::vec;
use alloc::vec::Vec;
use burn::tensor::{backend::Backend, Distribution, Tensor};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use super::{BetaSchedule, PredictionType};

/// Configuration for the Euler Discrete Scheduler.
#[derive(Debug, Clone)]
pub struct EulerDiscreteSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// Number of diffusion steps used to train the model.
    pub train_timesteps: usize,
    /// Prediction type of the scheduler function.
    pub prediction_type: PredictionType,
}

impl Default for EulerDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            train_timesteps: 1000,
            prediction_type: PredictionType::Epsilon,
        }
    }
}

/// Euler Discrete Scheduler for diffusion models.
///
/// This scheduler implements the Euler method for solving the probability flow ODE
/// in diffusion models, as described in Karras et al. (2022).
#[derive(Debug, Clone)]
pub struct EulerDiscreteScheduler {
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
    init_noise_sigma: f64,
    /// The scheduler configuration.
    pub config: EulerDiscreteSchedulerConfig,
}

impl EulerDiscreteScheduler {
    /// Create a new Euler Discrete Scheduler.
    ///
    /// # Arguments
    /// * `inference_steps` - Number of inference steps
    /// * `config` - Scheduler configuration
    pub fn new(inference_steps: usize, config: EulerDiscreteSchedulerConfig) -> Self {
        let betas = match config.beta_schedule {
            BetaSchedule::ScaledLinear => {
                // linspace(beta_start.sqrt(), beta_end.sqrt(), train_timesteps).square()
                let start = config.beta_start.sqrt();
                let end = config.beta_end.sqrt();
                let step = (end - start) / (config.train_timesteps - 1) as f64;
                let betas: Vec<f64> = (0..config.train_timesteps)
                    .map(|i| {
                        let v = start + step * i as f64;
                        v * v
                    })
                    .collect();
                betas
            }
            BetaSchedule::Linear => {
                let step =
                    (config.beta_end - config.beta_start) / (config.train_timesteps - 1) as f64;
                (0..config.train_timesteps)
                    .map(|i| config.beta_start + step * i as f64)
                    .collect()
            }
            BetaSchedule::SquaredcosCapV2 => {
                unimplemented!(
                    "EulerDiscreteScheduler only implements linear and scaled_linear betas."
                )
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

        // timesteps = linspace(train_timesteps - 1, 0, inference_steps)
        let timesteps: Vec<f64> =
            linspace((config.train_timesteps - 1) as f64, 0.0, inference_steps);

        // sigmas = sqrt((1 - alphas_cumprod) / alphas_cumprod)
        let sigmas_full: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&acp| ((1.0 - acp) / acp).sqrt())
            .collect();

        // Interpolate sigmas at timestep positions
        let xp: Vec<f64> = (0..sigmas_full.len()).map(|i| i as f64).collect();
        let sigmas = interp(&timesteps, &xp, &sigmas_full);

        // Append 0.0 to sigmas
        let mut sigmas = sigmas;
        sigmas.push(0.0);

        // init_noise_sigma = max(sigmas)
        let init_noise_sigma = sigmas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            timesteps,
            sigmas,
            init_noise_sigma,
            config,
        }
    }

    /// Get the timesteps for the scheduler.
    pub fn timesteps(&self) -> &[f64] {
        self.timesteps.as_slice()
    }

    /// Get the initial noise sigma value.
    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
    }

    /// Scale the model input by the appropriate sigma value.
    ///
    /// # Arguments
    /// * `sample` - The input sample tensor
    /// * `timestep` - The current timestep
    pub fn scale_model_input<B: Backend>(
        &self,
        sample: Tensor<B, 4>,
        timestep: f64,
    ) -> Tensor<B, 4> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .expect("Timestep not found in scheduler timesteps");
        let sigma = self.sigmas[step_index];

        // sample / sqrt(sigma^2 + 1)
        let scale = (sigma.powi(2) + 1.0).sqrt();
        sample / scale
    }

    /// Perform one step of the Euler method.
    ///
    /// # Arguments
    /// * `model_output` - The model's predicted noise
    /// * `timestep` - The current timestep
    /// * `sample` - The current noisy sample
    pub fn step<B: Backend>(
        &self,
        model_output: &Tensor<B, 4>,
        timestep: f64,
        sample: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        // Euler method parameters (no stochasticity by default)
        let (s_churn, s_tmin, s_tmax, s_noise) = (0.0, 0.0, f64::INFINITY, 1.0);

        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .expect("Timestep not found in scheduler timesteps");
        let sigma = self.sigmas[step_index];

        let gamma = if s_tmin <= sigma && sigma <= s_tmax {
            (s_churn / (self.sigmas.len() as f64 - 1.0)).min(2.0_f64.sqrt() - 1.0)
        } else {
            0.0
        };

        let device = sample.device();
        let noise: Tensor<B, 4> = Tensor::random(
            model_output.shape(),
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let eps = noise * s_noise;
        let sigma_hat = sigma * (gamma + 1.0);

        let sample = if gamma > 0.0 {
            sample.clone() + eps * (sigma_hat.powi(2) - sigma.powi(2)).sqrt()
        } else {
            sample.clone()
        };

        // 1. Compute predicted original sample (x_0) from sigma-scaled predicted noise
        let pred_original_sample = match self.config.prediction_type {
            PredictionType::Epsilon => sample.clone() - model_output.clone() * sigma_hat,
            PredictionType::VPrediction => {
                let sigma_sq_plus_1 = sigma.powi(2) + 1.0;
                model_output.clone() * (-sigma / sigma_sq_plus_1.sqrt())
                    + sample.clone() / sigma_sq_plus_1
            }
            PredictionType::Sample => {
                unimplemented!("Prediction type must be one of `epsilon` or `v_prediction`")
            }
        };

        // 2. Convert to an ODE derivative
        let derivative = (sample.clone() - pred_original_sample) / sigma_hat;
        let dt = self.sigmas[step_index + 1] - sigma_hat;

        // Euler step: sample + derivative * dt
        sample + derivative * dt
    }

    /// Add noise to original samples.
    ///
    /// # Arguments
    /// * `original_samples` - The original clean samples
    /// * `noise` - The noise to add
    /// * `timestep` - The timestep at which to add noise
    pub fn add_noise<B: Backend>(
        &self,
        original_samples: &Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timestep: f64,
    ) -> Tensor<B, 4> {
        let step_index = self
            .timesteps
            .iter()
            .position(|&t| t == timestep)
            .expect("Timestep not found in scheduler timesteps");
        let sigma = self.sigmas[step_index];

        original_samples.clone() + noise * sigma
    }
}

/// Create a linearly spaced vector from start to end with n points.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// One-dimensional linear interpolation for monotonically increasing sample points.
/// Mimics numpy's interp() function.
///
/// # Arguments
/// * `x` - x-coordinates at which to evaluate the interpolated values
/// * `xp` - x-coordinates of the data points (must be increasing)
/// * `yp` - y-coordinates of the data points
fn interp(x: &[f64], xp: &[f64], yp: &[f64]) -> Vec<f64> {
    assert_eq!(xp.len(), yp.len());
    let sz = xp.len();

    // Compute slopes: m = (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
    let m: Vec<f64> = (0..sz - 1)
        .map(|i| (yp[i + 1] - yp[i]) / (xp[i + 1] - xp[i]))
        .collect();

    // Compute intercepts: b = yp[:-1] - m * xp[:-1]
    let b: Vec<f64> = (0..sz - 1).map(|i| yp[i] - m[i] * xp[i]).collect();

    // For each x value, find the appropriate segment and interpolate
    x.iter()
        .map(|&xi| {
            // Find index: sum(x >= xp) - 1, clamped to valid range
            let mut idx = 0;
            for (i, &xp_val) in xp.iter().enumerate() {
                if xi >= xp_val {
                    idx = i;
                }
            }
            // Clamp to valid range for m and b
            let idx = idx.min(m.len() - 1);
            m[idx] * xi + b[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::Shape;

    #[test]
    fn test_euler_discrete_scheduler_creation() {
        let config = EulerDiscreteSchedulerConfig::default();
        let scheduler = EulerDiscreteScheduler::new(20, config);

        assert_eq!(scheduler.timesteps().len(), 20);
        assert_eq!(scheduler.sigmas.len(), 21); // 20 + 1 (appended 0)
        assert!(scheduler.init_noise_sigma() > 0.0);
    }

    #[test]
    fn test_euler_discrete_timesteps() {
        let config = EulerDiscreteSchedulerConfig::default();
        let scheduler = EulerDiscreteScheduler::new(10, config);

        let timesteps = scheduler.timesteps();
        // Should be decreasing from train_timesteps-1 to 0
        assert!((timesteps[0] - 999.0).abs() < 0.1);
        assert!((timesteps[9] - 0.0).abs() < 0.1);

        // Should be monotonically decreasing
        for i in 1..timesteps.len() {
            assert!(timesteps[i] < timesteps[i - 1]);
        }
    }

    #[test]
    fn test_euler_discrete_scale_model_input() {
        let device = Default::default();
        let config = EulerDiscreteSchedulerConfig::default();
        let scheduler = EulerDiscreteScheduler::new(20, config);

        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let scaled = scheduler.scale_model_input(sample, timestep);
        assert_eq!(scaled.shape(), Shape::from([1, 4, 8, 8]));

        // Scaled values should be less than original (division by sqrt(sigma^2 + 1))
        let scaled_mean: f32 = scaled.mean().into_scalar();
        assert!(scaled_mean < 1.0);
        assert!(scaled_mean > 0.0);
    }

    #[test]
    fn test_euler_discrete_step() {
        let device = Default::default();
        let config = EulerDiscreteSchedulerConfig::default();
        let scheduler = EulerDiscreteScheduler::new(20, config);

        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];

        let result = scheduler.step(&model_output, timestep, &sample);
        assert_eq!(result.shape(), Shape::from([1, 4, 8, 8]));
    }

    #[test]
    fn test_linspace() {
        let result = linspace(0.0, 10.0, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[4] - 10.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let yp = vec![0.0, 2.0, 4.0, 6.0];
        let x = vec![0.5, 1.5, 2.5];

        let result = interp(&x, &xp, &yp);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
    }

    /// Test Euler Discrete scheduler values match diffusers-rs
    #[test]
    fn test_euler_discrete_matches_diffusers_rs() {
        let device = Default::default();
        let config = EulerDiscreteSchedulerConfig::default();
        let scheduler = EulerDiscreteScheduler::new(20, config);

        // Reference values from diffusers-rs
        let expected_timesteps = [
            999.0,
            946.4210205078125,
            893.8421020507813,
            841.26318359375,
            788.6842041015625,
            736.105224609375,
            683.5263061523438,
            630.9473876953125,
            578.368408203125,
            525.7894287109375,
            473.2105407714844,
            420.631591796875,
            368.0526428222656,
            315.47369384765625,
            262.8947448730469,
            210.3157958984375,
            157.73684692382813,
            105.15789794921875,
            52.578948974609375,
            0.0,
        ];

        // Check timesteps
        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), expected_timesteps.len());
        for (i, (actual, expected)) in timesteps.iter().zip(expected_timesteps.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-3,
                "Timestep mismatch at {}: actual={}, expected={}",
                i,
                actual,
                expected
            );
        }

        // Check init_noise_sigma (reference: 14.614646291831562)
        let init_sigma = scheduler.init_noise_sigma();
        assert!(
            (init_sigma - 14.614646291831562).abs() < 1e-4,
            "init_noise_sigma mismatch: actual={}, expected=14.614646291831562",
            init_sigma
        );

        // Check scale_model_input (reference mean: 0.06826489418745041)
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let timestep = scheduler.timesteps()[0];
        let scaled = scheduler.scale_model_input(sample, timestep);
        let scaled_mean: f32 = scaled.mean().into_scalar();
        assert!(
            (scaled_mean as f64 - 0.06826489418745041).abs() < 1e-4,
            "scale_model_input mean mismatch: actual={}, expected=0.06826489418745041",
            scaled_mean
        );

        // Check step (reference mean: 1.0 when model_output is zeros)
        let model_output: Tensor<TestBackend, 4> = Tensor::zeros([1, 4, 8, 8], &device);
        let sample: Tensor<TestBackend, 4> = Tensor::ones([1, 4, 8, 8], &device);
        let result = scheduler.step(&model_output, timestep, &sample);
        let result_mean: f32 = result.mean().into_scalar();
        assert!(
            (result_mean - 1.0).abs() < 1e-4,
            "step mean mismatch: actual={}, expected=1.0",
            result_mean
        );
    }
}
